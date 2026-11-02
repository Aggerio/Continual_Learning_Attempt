# Nonlinear assimilation matched replication

Three training seeds (43–45) compared direct supervised compact learning with compact students trained from an expanded nonlinear teacher. All used the same fixed D-to-E training examples, replay and token schedule per seed. Seeds vary initialization and order; the evaluation cohort varies by seed.

## Mean held-out atomic accuracy

| Arm | A select | B filter | C sort | D group | E join | Atomic mean |
|---|---:|---:|---:|---:|---:|---:|
| direct | 100.0% | 83.3% | 83.3% | 83.3% | 100.0% | 90.0% |
| output | 100.0% | 83.3% | 83.3% | 87.5% | 100.0% | 90.8% |
| guided | 100.0% | 81.2% | 83.3% | 87.5% | 100.0% | 90.4% |
| random | 100.0% | 83.3% | 85.4% | 87.5% | 100.0% | 91.2% |

## Mean held-out composition accuracy

| Arm | BA | BC | BD | BE | BDA | BCE |
|---|---:|---:|---:|---:|---:|---:|
| direct | 91.7% | 83.3% | 25.0% | 12.5% | 0.0% | 0.0% |
| output | 87.5% | 83.3% | 20.8% | 20.8% | 0.0% | 0.0% |
| guided | 95.8% | 79.2% | 16.7% | 12.5% | 0.0% | 0.0% |
| random | 95.8% | 83.3% | 16.7% | 16.7% | 0.0% | 0.0% |

## What this says

The expanded teacher passed the development acquisition gate in 3/3 seeds. The direct compact control also reached 100.0% mean join accuracy, matching the teacher-based arms on this small test cohort. This gives no evidence that temporary expansion was needed for this task.

Mechanistic guidance did not improve the result consistently: guided matching averaged 100.0% on join and 83.3% on sorting, while the random-location arm averaged 100.0% and 85.4%. Harder compositions remained weak; BDA and BCE were 0% in every arm on this sample.

This is not proof of continual learning. It is a three-training-seed, one-boundary replication with 16 examples per atomic test skill and 8 per composition. Training examples and task order were fixed across seeds, and the tested tasks are synthetic. Results are exploratory and do not establish broad retention or factual learning.

Artifacts: `summary.json`, `AUDIT.json`, per-seed frozen inputs and reports. The earlier full-size screen remains at `../screen-002/REPORT.md`.
