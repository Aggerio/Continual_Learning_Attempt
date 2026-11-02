#!/usr/bin/env python3
"""Run the frozen three-method, five-stage compact-assimilation comparison."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from continual.tasks import compact, general_examples
from continual.worker import choose_replay, write


def read(path):
    return json.loads(Path(path).read_text())


def source_paths():
    return [*sorted((ROOT/'continual').glob('*.py')), Path(__file__).resolve(), ROOT/'configs/assimilation.json']


def check_sources(root):
    frozen = read(root/'frozen.json')
    for name, digest in frozen['sources'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Frozen source changed: {name}')


def call(root, module, action, source, data, output, config=None, **extras):
    check_sources(root)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, '-m', module, action, '--config', str(config or root/'config.json'),
           '--source', str(source), '--output', str(out)]
    if data:
        cmd += ['--data', str(data)]
    for k, v in extras.items():
        if v is None or v is False:
            continue
        cmd += ['--'+k.replace('_', '-')]
        if v is not True:
            cmd.append(str(v))
    log = out.parent/(out.stem+'-'+action+'.log')
    print(action, source, '->', out, flush=True)
    started = time.perf_counter()
    with log.open('w') as f:
        subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True)
    write(log.with_suffix('.timing.json'), {'wall_seconds': time.perf_counter()-started, 'command': cmd})


def anchors():
    result = []
    for i in range(16):
        a, b = 21+i, 43+2*i
        prompt = f'What is {a} plus {b}? Reply only with a JSON integer.'
        result.append({'id': f'anchor-{i}', 'skill': 'anchor', 'split': 'train',
                       'system': 'Follow the user instruction exactly.', 'prompt': prompt, 'answer': a+b})
    return result


def acceptance(groups, mc, skill, best, base_mc, config):
    reasons = []
    if groups[skill]['strict_correct'] < config['minimum_acquisition']:
        reasons.append('new skill below acquisition threshold')
    for old, score in best.items():
        if score - groups[old]['strict_correct'] > config['retention_tolerance'] + 1e-9:
            reasons.append(f'{old} exceeds retention tolerance')
    if base_mc - mc > config['general_tolerance'] + 1e-9:
        reasons.append('general probe exceeds tolerance')
    return not reasons, reasons


def eval_data(root, split, count, stage_index, name):
    # All atomic tasks are evaluated for accepted/rejected proposal accounting.
    skills = ['A', 'B', 'C', 'D', 'E']
    if stage_index >= 3:
        skills += ['BDA']
    if stage_index >= 4:
        skills += ['BCE']
    examples = []
    for skill in skills:
        xs = read(root/'archive'/split/f'{skill}.json')
        examples += xs[:count] if count else xs
    # One legacy JSON question triggers the separate 32-question MC probe.
    examples += general_examples()[:1]
    target = root/'evaluation'/f'{name}.json'
    write(target, examples)
    return target


def prepare(root, config):
    from transformers import AutoTokenizer
    old = ROOT/config['archive_root']
    calibration = ROOT/config['calibration_root']
    # Keep both the previously fixed 32-item development subset and all 64 scores.
    calibration_summary = {}
    initial = read(old/'pilot'/'baseline.json')['groups']
    for skill in ['B', 'C']:
        report = read(calibration/skill/'eval.json')
        items = [x for x in report['results'] if x['skill'] == skill]
        first = items[:32]
        score = sum(x['strict_correct'] for x in first)/len(first)
        sem = sum(x['semantic_correct'] for x in first)/len(first)
        passed = (report['groups'][skill]['strict_correct'] >= 0.75 and score >= 0.75 and
                  score-initial[skill]['strict_correct'] >= 0.10 and
                  sem-initial[skill]['semantic_correct'] >= 0.10)
        calibration_summary[skill] = {'first_32_accuracy': score, 'all_64_accuracy': report['groups'][skill]['strict_correct'],
                                      'passes': passed}
    if not all(x['passes'] for x in calibration_summary.values()):
        raise RuntimeError('Acquisition calibration failed; final comparison not started.')
    root.mkdir(parents=True, exist_ok=True)
    write(root/'config.json', config)
    original = read(old/'manifest.json')
    shutil.copytree(old/'archive', root/'archive', dirs_exist_ok=True)
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted((root/'archive').rglob('*.json'))}
    if hashes != original['archive_sha256']:
        raise RuntimeError('Copied archive differs from original.')
    tokenizer = AutoTokenizer.from_pretrained(original['base'], local_files_only=True)
    memory, memory_stats = choose_replay(tokenizer, anchors(), config)
    write(root/'initial-memory.json', memory)
    write(root/'acquisition-config.json', {**config, 'stage_training_tokens': config['acquisition_tokens']})
    sources = {}
    for p in source_paths():
        name = str(p.relative_to(ROOT))
        target = root/'sources'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target)
        sources[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    import importlib.metadata
    write(root/'frozen.json', {'frozen_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'model': config['model'], 'revision': config['revision'], 'base': original['base'],
          'sources': sources, 'archive_sha256': hashes, 'calibration': calibration_summary,
          'initial_memory_sha256': hashlib.sha256((root/'initial-memory.json').read_bytes()).hexdigest(),
          'initial_memory': memory_stats, 'python': sys.version,
          'packages': {p: importlib.metadata.version(p) for p in ['torch', 'transformers', 'peft']},
          'scope': 'Exploratory single-seed comparison; dev acceptance gates, final untouched tests.'})


def plasticity(root, source, name, config):
    stage = root/'plasticity'/name
    data = stage/'train-data.json'
    write(data, {'current': read(root/'archive'/'train'/'E.json')})
    if not (stage/'training.json').exists():
        call(root, 'continual.worker', 'train', source, data, stage,
             config=root/'acquisition-config.json', half=True)
    dev = root/'evaluation'/'plasticity-E.json'
    write(dev, read(root/'archive'/'dev'/'E.json'))
    if not (stage/'full-eval.json').exists():
        call(root, 'continual.worker', 'eval', stage/'checkpoint', dev, stage/'full-eval.json')
    if not (stage/'half-eval.json').exists():
        call(root, 'continual.worker', 'merge-half', source, None, stage/'half-checkpoint', adapter=stage/'half_adapter')
        call(root, 'continual.worker', 'eval', stage/'half-checkpoint', dev, stage/'half-eval.json')
        shutil.rmtree(stage/'half-checkpoint')
    shutil.rmtree(stage/'checkpoint', ignore_errors=True)


def run(root, config):
    from transformers import AutoTokenizer
    frozen = read(root/'frozen.json')
    if hashlib.sha256((root/'initial-memory.json').read_bytes()).hexdigest() != frozen['initial_memory_sha256']:
        raise RuntimeError('Frozen initial replay changed.')
    base = frozen['base']
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    # Recheck test archive integrity immediately before any final evaluation.
    for name, digest in frozen['archive_sha256'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != digest:
            raise RuntimeError('Frozen archive changed.')
    baseline_path = root/'baseline-dev.json'
    if not baseline_path.exists():
        data = eval_data(root, 'dev', config['intermediate_eval_examples'], 4, 'baseline-dev')
        call(root, 'continual.worker', 'eval', base, data, baseline_path)
    baseline = read(baseline_path)
    base_mc = baseline['general_mc']['accuracy']
    if not (root/'plasticity'/'fresh'/'half-eval.json').exists():
        plasticity(root, base, 'fresh', config)
    progress = {'status': 'running', 'methods': {}}
    for method in config['methods']:
        source = base
        memory = read(root/'initial-memory.json')
        current_eval = baseline
        best = {}
        rows = []
        for i, skill in enumerate(config['order']):
            stage = root/'streams'/method/skill
            current = read(root/'archive'/'train'/f'{skill}.json')
            composition_skill = {1: 'BA', 2: 'BC', 3: 'BD', 4: 'BE'}.get(i)
            compositions = (read(root/'archive'/'train'/f'{composition_skill}.json')
                            if method == 'composition' and composition_skill else [])
            memory_stats = {'examples': len(memory), 'tokens': len(tokenizer.encode(compact(memory), add_special_tokens=False)),
                            'bytes': len(compact(memory).encode())}
            data = stage/'train-data.json'
            if not (stage/'decision.json').exists():
                write(data, {'current': current, 'memory': memory, 'replay': memory,
                             'replay_stats': memory_stats, 'compositions': compositions})
                if method == 'replay':
                    if not (stage/'training.json').exists():
                        call(root, 'continual.worker', 'train', source, data, stage)
                    checkpoint = stage/'checkpoint'
                else:
                    learner = stage/'acquire'
                    acquire_data = stage/'acquire-data.json'
                    write(acquire_data, {'current': current})
                    if not (learner/'training.json').exists():
                        call(root, 'continual.worker', 'train', source, acquire_data, learner,
                             config=root/'acquisition-config.json')
                    old_data, new_data = stage/'old-targets.json', stage/'new-targets.json'
                    write(old_data, {'role': 'old', 'examples': memory})
                    write(new_data, {'role': 'new', 'examples': current})
                    old_cache, new_cache = stage/'old-cache.pt', stage/'new-cache.pt'
                    for teacher, targets, target_cache in [(source, old_data, old_cache),
                                                           (learner/'checkpoint', new_data, new_cache)]:
                        if not target_cache.exists():
                            call(root, 'continual.assimilation', 'cache', teacher, targets, target_cache)
                    student = stage/'student'
                    if not (student/'training.json').exists():
                        call(root, 'continual.assimilation', 'train', source, data, student,
                             old_cache=old_cache, new_cache=new_cache)
                    checkpoint = student/'checkpoint'
                eval_path = stage/'proposal-eval.json'
                if not eval_path.exists():
                    dev = eval_data(root, 'dev', config['intermediate_eval_examples'], i, f'{method}-{skill}-dev')
                    call(root, 'continual.worker', 'eval', checkpoint, dev, eval_path)
                proposal = read(eval_path)
                accepted, reasons = acceptance(proposal['groups'], proposal['general_mc']['accuracy'], skill, best, base_mc, config)
                write(stage/'decision.json', {'accepted': accepted, 'reasons': reasons,
                      'old_source': str(source), 'proposal_source': str(checkpoint),
                      'new_skill_accuracy': proposal['groups'][skill]['strict_correct'],
                      'best_prior_scores': best, 'general_mc': proposal['general_mc']['accuracy']})
            decision = read(stage/'decision.json')
            proposal = read(stage/'proposal-eval.json')
            if decision['accepted']:
                previous = Path(source)
                source = Path(decision['proposal_source'])
                current_eval = proposal
                for acquired in [*best, skill]:
                    best[acquired] = max(best.get(acquired, 0), current_eval['groups'][acquired]['strict_correct'])
                if previous != Path(base) and previous != source and previous.exists():
                    shutil.rmtree(previous)
            else:
                shutil.rmtree(Path(decision['proposal_source']), ignore_errors=True)
            rows.append({'stage': skill, 'decision': decision,
                         'proposed': {k:v['strict_correct'] for k,v in proposal['groups'].items()},
                         'accepted_model': {k:v['strict_correct'] for k,v in current_eval['groups'].items()},
                         'accepted_general_mc': current_eval['general_mc']['accuracy']})
            # Same deterministic bounded atomic memory for every method.
            memory, _ = choose_replay(tokenizer, memory+current, config)
            write(stage/'memory.json', memory)
            if method != 'replay':
                for p in [stage/'old-cache.pt', stage/'new-cache.pt']:
                    p.unlink(missing_ok=True)
                shutil.rmtree(stage/'acquire'/'checkpoint', ignore_errors=True)
            if skill == 'D' and not (root/'plasticity'/method/'half-eval.json').exists():
                plasticity(root, source, method, config)
            progress['methods'][method] = {'stages': rows, 'source': str(source), 'accepted_skills': sorted(best)}
            write(root/'progress.json', progress)
            print('DECISION', method, skill, decision['accepted'], decision['reasons'], flush=True)
        final_path = root/'final'/f'{method}.json'
        if not final_path.exists():
            data = eval_data(root, 'test', None, 4, 'final-test')
            call(root, 'continual.worker', 'eval', source, data, final_path)
    baseline_final = root/'final'/'unchanged.json'
    if not baseline_final.exists():
        data = eval_data(root, 'test', None, 4, 'final-test')
        call(root, 'continual.worker', 'eval', base, data, baseline_final)
    progress['status'] = 'completed'
    write(root/'progress.json', progress)
    summarize(root, config, progress)


def summarize(root, config, progress):
    output = {'status': 'completed', 'methods': {}, 'limitations': [
        'Single seed/order; descriptive differences are not robust evidence.',
        'Composition method has additional verified composition supervision.',
        'Equal training-token ceilings do not match teacher compute or wall time.',
        'Rollback preserves scores only by refusing updates; failed acquisitions count as failures.',
        'General MC probe is narrow; no claim of universal retention.']}
    for method, state in progress['methods'].items():
        final = read(root/'final'/f'{method}.json')
        stages = state['stages']
        train_paths = sorted((root/'streams'/method).rglob('training.json'))
        trainings = [read(p) for p in train_paths]
        old_cache = sorted((root/'streams'/method).glob('*/old-cache.json'))
        new_cache = sorted((root/'streams'/method).glob('*/new-cache.json'))
        caches = [read(p) for p in old_cache+new_cache]
        peak_allocated = max(t['peak_allocated_bytes'] for t in trainings)
        output['methods'][method] = {
            'accepted_updates': sum(s['decision']['accepted'] for s in stages),
            'rejected_updates': sum(not s['decision']['accepted'] for s in stages),
            'accepted_skills': state['accepted_skills'],
            'final_atomic_accuracy': {s: final['groups'][s]['strict_correct'] for s in config['order']},
            'mean_final_atomic_accuracy': sum(final['groups'][s]['strict_correct'] for s in config['order'])/5,
            'final_composition_accuracy': {s: final['groups'][s]['strict_correct'] for s in ['BDA','BCE']},
            'general_mc': final['general_mc']['accuracy'],
            'training_tokens_including_acquisition': sum(t['training_tokens'] for t in trainings),
            'training_seconds_including_acquisition': sum(t['training_seconds'] for t in trainings),
            'teacher_cache_seconds': sum(c['seconds'] for c in caches),
            'teacher_forward_tokens': sum(c['teacher_forward_tokens'] for c in caches),
            'peak_allocated_training_bytes': peak_allocated,
            'deployed_parameters': final['deployed_parameters'],
            'checkpoint_bytes': sum(p.stat().st_size for p in Path(state['source']).rglob('*') if p.is_file()),
            'stages': stages}
    write(root/'summary.json', output)
    lines = ['# Compact assimilation experiment', '', 'Single-seed exploratory result. All scores below are final held-out test accuracy.', '',
             '| Method | Accepted updates | Atomic mean | BDA | BCE | General MC | Training minutes |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for method, r in output['methods'].items():
        lines.append(f"| {method} | {r['accepted_updates']}/5 | {r['mean_final_atomic_accuracy']:.1%} | "
                     f"{r['final_composition_accuracy']['BDA']:.1%} | {r['final_composition_accuracy']['BCE']:.1%} | "
                     f"{r['general_mc']:.1%} | {r['training_seconds_including_acquisition']/60:.1f} |")
    lines += ['', '## Interpretation', '', *['- '+x for x in output['limitations']], '',
              'See summary.json for proposed and accepted matrices, decision reasons, resource accounting, and per-skill results. ',
              'See plasticity/*/{half,full}-eval.json and training.json for matching-data fresh/late-E learning curves.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default=str(ROOT/'configs/assimilation.json'))
    p.add_argument('--output', default=str(ROOT/'results/assimilation/experiment-001'))
    a = p.parse_args()
    root = Path(a.output).resolve()
    config = read(a.config)
    if not (root/'frozen.json').exists():
        prepare(root, config)
    elif read(root/'config.json') != config:
        raise RuntimeError('Frozen configuration changed.')
    check_sources(root)
    try:
        run(root, config)
    except Exception as error:
        write(root/'error.json', {'type': type(error).__name__, 'message': str(error)})
        raise


if __name__ == '__main__':
    main()
