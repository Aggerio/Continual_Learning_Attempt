#!/usr/bin/env python3
"""Run fixed-seed replications of the nonlinear assimilation screen."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import nonlinear_assimilation_run as screen


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', nargs='+', type=int, default=[43, 44, 45])
    parser.add_argument('--output', default='results/nonlinear_assimilation/replication-001')
    args = parser.parse_args()
    root = (ROOT / args.output).resolve()
    base = screen.read(ROOT / 'configs/nonlinear_assimilation.json')
    manifest = {'seeds': args.seeds, 'control': 'direct supervised compact update; no expanded teacher',
                'arms': ['direct', 'output', 'guided', 'random'],
                'shared_training_examples': 'fixed original boundary cohort',
                'seed_scope': 'initialization and frozen example-order schedule; evaluation cohort seed varies',
                'evaluation_cohort': '16 per atomic skill and 8 per composition; exploratory screen smaller than original.',
                'note': 'Three training seeds estimate optimization variability, not data/task-order variability.'}
    write(root / 'replication.json', manifest)
    for seed in args.seeds:
        cfg = dict(base)
        cfg['dev_examples'] = 16
        cfg['test_examples'] = 16
        cfg['composition_examples'] = 8
        cfg['seed'] = seed
        cfg['evaluation_seed'] = 20261104 + seed
        cfg['notes'] = 'Independent training seed; same frozen D-to-E training cohort and token budget.'
        run = root / f'seed-{seed}'
        if not (run / 'frozen.json').exists():
            screen.prepare(run, cfg)
        elif cfg != screen.read(run / 'config.json'):
            raise RuntimeError(f'Frozen config mismatch at {run}')
        if not ((run / 'progress.json').exists() and
                screen.read(run / 'progress.json').get('status') == 'completed'):
            screen.run(run, cfg)
    rows = []
    for seed in args.seeds:
        run = root / f'seed-{seed}'
        row = {'seed': seed, 'teacher_acquired': screen.read(run / 'progress.json')['teacher_acquired']}
        for arm in manifest['arms']:
            report = screen.read(run / 'students' / arm / 'test.json')
            row[arm] = {skill: report['groups'][skill]['strict_correct'] for skill in 'ABCDE'}
            row[arm]['compositions'] = {skill: report['groups'][skill]['strict_correct']
                                        for skill in ['BA', 'BC', 'BD', 'BE', 'BDA', 'BCE']}
        rows.append(row)
    summary = {'manifest': manifest, 'runs': rows}
    for arm in manifest['arms']:
        summary[arm + '_mean_test_accuracy'] = {
            skill: sum(row[arm][skill] for row in rows) / len(rows) for skill in 'ABCDE'}
        summary[arm + '_mean_composition_accuracy'] = {
            skill: sum(row[arm]['compositions'][skill] for row in rows) / len(rows)
            for skill in ['BA', 'BC', 'BD', 'BE', 'BDA', 'BCE']}
    write(root / 'summary.json', summary)
    audit = {'seed_count': len(rows), 'all_teachers_acquired': all(r['teacher_acquired'] for r in rows),
             'runs': {}}
    for seed in args.seeds:
        run = root / f'seed-{seed}'
        screen.check(run)
        progress = screen.read(run / 'progress.json')
        audit['runs'][str(seed)] = {'completed': progress['status'] == 'completed',
                                    'teacher_acquired': progress['teacher_acquired'],
                                    'compact_parameter_counts': {}}
        for arm in manifest['arms']:
            stats = screen.read(run / 'students' / arm / 'training.json')
            audit['runs'][str(seed)]['compact_parameter_counts'][arm] = stats['deployed_parameters']
    audit['all_runs_completed'] = all(x['completed'] for x in audit['runs'].values())
    audit['all_compact_counts_match'] = all(
        count == 596049920 for run in audit['runs'].values()
        for count in run['compact_parameter_counts'].values())
    write(root / 'AUDIT.json', audit)
    arms = manifest['arms']
    lines = ['# Nonlinear assimilation matched replication', '',
             'Three training seeds (43–45) compared direct supervised compact learning with compact students trained from an expanded nonlinear teacher. All used the same fixed D-to-E training examples, replay and token schedule per seed. Seeds vary initialization and order; the evaluation cohort varies by seed.', '',
             '## Mean held-out atomic accuracy', '',
             '| Arm | A select | B filter | C sort | D group | E join | Atomic mean |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for arm in arms:
        vals = summary[arm + '_mean_test_accuracy']
        mean = sum(vals[s] for s in 'ABCDE') / 5
        lines.append('| ' + arm + ' | ' + ' | '.join(f'{100*vals[s]:.1f}%' for s in 'ABCDE') +
                     f' | {100*mean:.1f}% |')
    lines += ['', '## Mean held-out composition accuracy', '',
              '| Arm | BA | BC | BD | BE | BDA | BCE |', '|---|---:|---:|---:|---:|---:|---:|']
    for arm in arms:
        vals = summary[arm + '_mean_composition_accuracy']
        lines.append('| ' + arm + ' | ' + ' | '.join(f'{100*vals[s]:.1f}%' for s in ['BA','BC','BD','BE','BDA','BCE']) + ' |')
    lines += ['', '## What this says', '',
              f'The expanded teacher passed the development acquisition gate in {sum(r["teacher_acquired"] for r in rows)}/{len(rows)} seeds. The direct compact control also reached {100*summary["direct_mean_test_accuracy"]["E"]:.1f}% mean join accuracy, matching the teacher-based arms on this small test cohort. This gives no evidence that temporary expansion was needed for this task.',
              '', 'Mechanistic guidance did not improve the result consistently: guided matching averaged '
              f'{100*summary["guided_mean_test_accuracy"]["E"]:.1f}% on join and '
              f'{100*summary["guided_mean_test_accuracy"]["C"]:.1f}% on sorting, while the random-location arm averaged '
              f'{100*summary["random_mean_test_accuracy"]["E"]:.1f}% and '
              f'{100*summary["random_mean_test_accuracy"]["C"]:.1f}%. Harder compositions remained weak; BDA and BCE were 0% in every arm on this sample.',
              '', 'This is not proof of continual learning. It is a three-training-seed, one-boundary replication with 16 examples per atomic test skill and 8 per composition. Training examples and task order were fixed across seeds, and the tested tasks are synthetic. Results are exploratory and do not establish broad retention or factual learning.',
              '', 'Artifacts: `summary.json`, `AUDIT.json`, per-seed frozen inputs and reports. The earlier full-size screen remains at `../screen-002/REPORT.md`.']
    (root / 'REPORT.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
