#!/usr/bin/env python3
"""Generate, pilot, and run a gated five-stage continual-learning screen."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from continual.tasks import compact, generate, general_examples
from continual.worker import write


def worker(root, action, source, data, output, **kwargs):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    log = output.parent / (output.stem + f'-{action}.log')
    cmd = [sys.executable, '-m', 'continual.worker', action, '--config', str(root/'config.json'),
           '--source', str(source), '--output', str(output)]
    if data:
        cmd += ['--data', str(data)]
    for k, v in kwargs.items():
        if v is False or v is None:
            continue
        if v is True:
            cmd.append('--' + k.replace('_', '-'))
        else:
            cmd += ['--' + k.replace('_', '-'), str(v)]
    print('WORKER', action, source, '->', output, flush=True)
    with log.open('w') as stream:
        subprocess.run(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
    return output


def write_eval_data(root, split, skills, n=None, include_general=True, name='eval-data'):
    items = []
    for skill in skills:
        examples = json.loads((root/'archive'/split/f'{skill}.json').read_text())
        items += examples[:n] if n else examples
    if include_general:
        items += general_examples()
    path = root/'evaluation'/f'{name}.json'
    write(path, items)
    return path


def prepare(config_path, output):
    from huggingface_hub import snapshot_download
    config = json.loads(Path(config_path).read_text())
    root = Path(output).resolve()
    if (root/'manifest.json').exists():
        raise RuntimeError('Output already initialized; use pilot/run to resume.')
    if (root/'config.json').exists() and json.loads((root/'config.json').read_text()) != config:
        raise RuntimeError('Interrupted preparation has a different configuration.')
    root.mkdir(parents=True, exist_ok=True)
    write(root/'config.json', config)
    data = generate(config)
    for split, skills in data.items():
        for skill, items in skills.items():
            write(root/'archive'/split/f'{skill}.json', items)
    base = snapshot_download(config['model'], revision=config['revision'],
                             allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja', '*.model'])
    import importlib.metadata
    write(root/'manifest.json', {'base': base, 'model': config['model'], 'revision': config['revision'],
          'python': sys.version, 'packages': {p: importlib.metadata.version(p)
          for p in ['torch', 'transformers', 'peft', 'huggingface-hub']},
          'archive_sha256': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted((root/'archive').rglob('*.json'))},
          'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in [*sorted((ROOT/'continual').glob('*.py')), Path(__file__).resolve()]}})
    print('Prepared', root, flush=True)


def pilot(root, config, manifest):
    base = manifest['base']
    n = config['pilot_eval_examples']
    eval_data = write_eval_data(root, 'dev', config['order'], n, name='pilot')
    baseline = root/'pilot'/'baseline.json'
    if not baseline.exists():
        worker(root, 'eval', base, eval_data, baseline)
    base_scores = json.loads(baseline.read_text())['groups']
    summary = {'status': 'running', 'split': 'dev', 'examples_per_skill': n, 'skills': {}}
    for skill in config['order']:
        stage = root/'pilot'/skill
        train_data = stage/'train-data.json'
        write(train_data, {'current': json.loads((root/'archive'/'train'/f'{skill}.json').read_text())})
        if not (stage/'training.json').exists():
            worker(root, 'train', base, train_data, stage, half=(skill == 'E'))
        skill_data = write_eval_data(root, 'dev', [skill], n, name=f'pilot-{skill}')
        if not (stage/'eval.json').exists():
            worker(root, 'eval', stage/'checkpoint', skill_data, stage/'eval.json')
        after = json.loads((stage/'eval.json').read_text())['groups']
        before = base_scores[skill]['strict_correct']
        accuracy = after[skill]['strict_correct']
        gain = accuracy - before
        semantic_gain = after[skill]['semantic_correct'] - base_scores[skill]['semantic_correct']
        adequate = (accuracy >= config['pilot_min_accuracy'] and gain >= config['pilot_min_gain'] and
                    semantic_gain >= config['pilot_min_semantic_gain'])
        summary['skills'][skill] = {'before': before, 'after': accuracy, 'gain': gain,
                                    'semantic_before': base_scores[skill]['semantic_correct'],
                                    'semantic_after': after[skill]['semantic_correct'],
                                    'semantic_gain': semantic_gain,
                                    'passes_acquisition_gate': adequate,
                                    'general_before': base_scores['general']['strict_correct'],
                                    'general_after': after['general']['strict_correct'],
                                    'training': json.loads((stage/'training.json').read_text())}
        write(root/'pilot'/'summary.json', summary)
        print('PILOT', skill, before, '->', accuracy, 'gate', adequate, flush=True)
        if skill != 'E':
            shutil.rmtree(stage/'checkpoint', ignore_errors=True)
        elif not (stage/'half-eval.json').exists():
            worker(root, 'merge-half', base, None, stage/'half-checkpoint', adapter=stage/'half_adapter')
            worker(root, 'eval', stage/'half-checkpoint', skill_data, stage/'half-eval.json')
            shutil.rmtree(stage/'half-checkpoint')
    summary['status'] = 'completed'
    summary['passes_all_acquisition_gates'] = all(s['passes_acquisition_gate'] for s in summary['skills'].values())
    summary['next_action'] = ('freeze_and_run' if summary['passes_all_acquisition_gates'] else
                              'revise_on_development_data; final test remains unopened')
    write(root/'pilot'/'summary.json', summary)
    lines = ['# Isolated-skill development pilot', '',
             f"Status: {summary['status']}. Seed {config['seed']}; {n} held-out development examples per skill.", '',
             '| Skill | Before | After | Gain | Gate | Training seconds |',
             '| --- | ---: | ---: | ---: | --- | ---: |']
    for skill, stats in summary['skills'].items():
        lines.append(f"| {skill} | {stats['before']:.1%} | {stats['after']:.1%} | {stats['gain']:+.1%} | "
                     f"{'pass' if stats['passes_acquisition_gate'] else 'fail'} | {stats['training']['training_seconds']:.1f} |")
    lines += ['', 'Gate: at least 75% strict accuracy and 10 percentage points of both strict and semantic gain on every skill.', '',
              'Next action: ' + summary['next_action'] + '.', '',
              'These independent updates do not measure sequential forgetting. Final test examples have not been scored by this pilot. ',
              'Strict correctness requires parseable JSON with the correct value and array order. Formatting-tolerant accuracy and full responses are retained in each evaluation JSON.', '',
              'The general probe has only 16 fixed questions and does not establish broad general retention.', '',
              'Exact configurations, checkpoint identity, data/source hashes, training costs, and evaluation reports are saved alongside this report.']
    (root/'PILOT_REPORT.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def run_streams(root, config, manifest):
    from transformers import AutoTokenizer
    from continual.worker import choose_replay
    pilot_summary = json.loads((root/'pilot'/'summary.json').read_text())
    if not pilot_summary.get('passes_all_acquisition_gates'):
        raise RuntimeError('Isolated-skill gate failed. Revise using dev data before opening test.')
    # Freeze before any final test answer is scored.
    frozen = root/'frozen.json'
    if not frozen.exists():
        write(frozen, {'config': config, 'manifest': manifest,
                      'final_source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                               for p in [*sorted((ROOT/'continual').glob('*.py')), Path(__file__).resolve()]},
                      'frozen_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      'budget_tolerance_tokens': config['max_sequence_tokens'],
                      'interpretation': 'Single-seed exploratory comparison. Composition method changes replay content, not architecture.'})
    elif json.loads(frozen.read_text())['config'] != config:
        raise RuntimeError('Frozen configuration changed.')
    base = manifest['base']
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    baseline = root/'streams'/'baseline.json'
    if not baseline.exists():
        data = write_eval_data(root, 'test', config['order'] + ['BDA', 'BCE'], name='final-baseline')
        worker(root, 'eval', base, data, baseline)
    for method in ['sequential', 'replay', 'composition']:
        source = base
        memory = []
        for i, skill in enumerate(config['order']):
            stage = root/'streams'/method/skill
            current = json.loads((root/'archive'/'train'/f'{skill}.json').read_text())
            if skill == 'E' and method != 'sequential':
                # A separate current-only E branch matches fresh-E data exposure.
                # Actual replay training on E would confound plasticity with replay ratio.
                branch = root/'plasticity'/method
                branch_data = branch/'train-data.json'
                write(branch_data, {'current': current})
                if not (branch/'training.json').exists():
                    worker(root, 'train', source, branch_data, branch, half=True)
                dev = write_eval_data(root, 'dev', ['E'], config['pilot_eval_examples'], name='plasticity-E')
                if not (branch/'full-eval.json').exists():
                    worker(root, 'eval', branch/'checkpoint', dev, branch/'full-eval.json')
                if not (branch/'half-eval.json').exists():
                    worker(root, 'merge-half', source, None, branch/'half-checkpoint', adapter=branch/'half_adapter')
                    worker(root, 'eval', branch/'half-checkpoint', dev, branch/'half-eval.json')
                    shutil.rmtree(branch/'half-checkpoint')
                shutil.rmtree(branch/'checkpoint', ignore_errors=True)
            replay = [] if method == 'sequential' else memory
            if method == 'composition' and i > 0:
                composition = {1: 'BA', 2: 'BC', 3: 'BD', 4: 'BE'}[i]
                combined = json.loads((root/'archive'/'train'/f'{composition}.json').read_text())
                replay, _ = choose_replay(tokenizer, memory + combined, config)
            stats = {'examples': len(replay), 'tokens': len(tokenizer.encode(compact(replay))) if replay else 0,
                     'bytes': len(compact(replay).encode()) if replay else 0}
            train_data = stage/'train-data.json'
            write(train_data, {'current': current, 'replay': replay, 'replay_stats': stats})
            if not (stage/'training.json').exists():
                worker(root, 'train', source, train_data, stage, half=(skill == 'E'))
            skills = config['order'][:i+1]
            combinations = (['BDA'] if i >= 3 else []) + (['BCE'] if i >= 4 else [])
            data = write_eval_data(root, 'test', skills + combinations, name=f'final-{method}-{skill}')
            if not (stage/'eval.json').exists():
                worker(root, 'eval', stage/'checkpoint', data, stage/'eval.json')
            if skill == 'E' and not (stage/'half-eval.json').exists():
                worker(root, 'merge-half', source, None, stage/'half-checkpoint', adapter=stage/'half_adapter')
                dev = write_eval_data(root, 'dev', ['E'], config['pilot_eval_examples'], name='late-E')
                worker(root, 'eval', stage/'half-checkpoint', dev, stage/'half-eval.json')
                worker(root, 'eval', stage/'checkpoint', dev, stage/'plasticity-eval.json')
                shutil.rmtree(stage/'half-checkpoint')
            if method != 'sequential':
                pool = replay + current
                memory, _ = choose_replay(tokenizer, pool, config)
                write(stage/'memory.json', memory)
            previous = Path(source)
            source = stage/'checkpoint'
            # Keep only the current compact checkpoint per method; reports persist.
            if previous != Path(base) and previous.exists():
                shutil.rmtree(previous)
    summarize(root, config)


def summarize(root, config):
    summary = {'status': 'completed', 'methods': {}, 'limitations': [
        'One seed and one order; differences are exploratory.',
        'General probe is small and JSON-focused.',
        'Synthetic composition supervision is an explicit data difference.',
        'Token budgets include prompts and answers; supervised-token counts are separately reported.']}
    for method in ['sequential', 'replay', 'composition']:
        matrix, best, worst_drop, acquired = {}, {}, 0.0, []
        for skill in config['order']:
            p = root/'streams'/method/skill/'eval.json'
            groups = json.loads(p.read_text())['groups']
            matrix[skill] = {k: v['strict_correct'] for k, v in groups.items()}
            acquired.append(matrix[skill][skill])
            for k in config['order']:
                if k in matrix[skill]:
                    if k in best:
                        worst_drop = max(worst_drop, best[k] - matrix[skill][k])
                    best[k] = max(best.get(k, 0), matrix[skill][k])
        general_base = json.loads((root/'streams'/'baseline.json').read_text())['general_mc']['accuracy']
        general_mc = {s: json.loads((root/'streams'/method/s/'eval.json').read_text())['general_mc']['accuracy']
                      for s in config['order']}
        general_drop = max(0.0, max(general_base - x for x in general_mc.values()))
        final = matrix[config['order'][-1]]
        training = [json.loads((root/'streams'/method/s/'training.json').read_text()) for s in config['order']]
        token_budget_ok = all(0 <= config['stage_training_tokens'] - t['training_tokens'] < config['max_sequence_tokens']
                              for t in training)
        summary['methods'][method] = {
            'accuracy_matrix': matrix, 'worst_skill_drop': worst_drop,
            'mean_acquisition_accuracy': sum(acquired)/len(acquired),
            'final_mean_atomic_accuracy': sum(final[s] for s in config['order'])/len(config['order']),
            'final_composition_accuracy': sum(final[s] for s in ['BDA', 'BCE'])/2,
            'worst_general_drop_from_base': general_drop,
            'general_mc_after_stage': general_mc,
            'within_retention_tolerance': worst_drop <= config['retention_tolerance'],
            'within_general_tolerance': general_drop <= config['general_tolerance'],
            'comparable_training_token_budget': token_budget_ok,
            'total_training_seconds': sum(t['training_seconds'] for t in training),
            'total_training_tokens': sum(t['training_tokens'] for t in training),
            'peak_allocated_training_bytes': max(t['peak_allocated_bytes'] for t in training),
            'checkpoint_bytes': sum(p.stat().st_size for p in (root/'streams'/method/'E'/'checkpoint').rglob('*') if p.is_file()),
            'deployed_parameters': training[-1]['deployed_parameters']}
    candidate = summary['methods']['composition']
    baseline = summary['methods']['replay']
    summary['candidate_comparison'] = {
        'final_atomic_gain_over_replay': candidate['final_mean_atomic_accuracy'] - baseline['final_mean_atomic_accuracy'],
        'composition_gain_over_replay': candidate['final_composition_accuracy'] - baseline['final_composition_accuracy'],
        'acquisition_change_over_replay': candidate['mean_acquisition_accuracy'] - baseline['mean_acquisition_accuracy'],
        'interpretation': 'Exploratory; repeat seeds and orders before treating numerical gains as progress.'}
    write(root/'summary.json', summary)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'pilot', 'run', 'all'])
    p.add_argument('--config', default=str(ROOT/'configs'/'quick.json'))
    p.add_argument('--output', default=str(ROOT/'results'/'continual_quick'/'screen-001'))
    a = p.parse_args()
    root = Path(a.output).resolve()
    if a.action == 'prepare' or (a.action == 'all' and not (root/'manifest.json').exists()):
        prepare(a.config, root)
    if a.action == 'prepare':
        return
    config = json.loads((root/'config.json').read_text())
    if config['order'] != ['A', 'B', 'C', 'D', 'E']:
        raise RuntimeError('This first screen supports A to E; review composition eligibility for new orders.')
    manifest = json.loads((root/'manifest.json').read_text())
    if a.action in ['pilot', 'all']:
        outcome = pilot(root, config, manifest)
        if a.action == 'all' and not outcome['passes_all_acquisition_gates']:
            print('Development gate failed; stopped before final test evaluation.', flush=True)
            return
    if a.action in ['run', 'all']:
        run_streams(root, config, manifest)


if __name__ == '__main__':
    main()
