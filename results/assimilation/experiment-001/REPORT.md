# Compact assimilation: measured first experiment

This is a single-seed, five-stage experiment on Qwen3-0.6B. A temporary learner supplies targets to a separate compact successor initialized from the previous accepted model. All saved successors contain 596,049,920 parameters. Training and fresh-process evaluation ran with Transformers on the CUDA GPU; these results do not mean an Ollama model was updated.

## Final held-out task results

Each atomic task and reserved composition has 128 untouched test examples. Scores require the exact correct JSON value and array order. A = select/rename, B = filter, C = sort, D = grouped sum, E = join.

| Method | Accepted updates | A | B | C | D | E | Atomic mean | BDA | BCE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Unchanged | 0/5 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| replay | 3/5 | 100.00% | 81.25% | 92.19% | 0.00% | 0.00% | 54.69% | 0.78% | 0.00% |
| consolidation | 4/5 | 100.00% | 85.94% | 77.34% | 16.41% | 100.00% | 75.94% | 0.00% | 3.91% |
| composition | 3/5 | 100.00% | 78.12% | 93.75% | 0.00% | 2.34% | 54.84% | 0.00% | 0.00% |

**No method accepted all five updates. This experiment has not established continual acquisition without forgetting.**

BDA = filter → group → select, and additionally changes selection from one output field in atomic training to two fields. Its failure combines composition and an output-arity shift. BCE = filter → sort → join is the cleaner unseen-composition test. Composition training used BA/BC/BD/BE pairs; neither BDA nor BCE entered training. The composition variant has additional oracle-verified supervision, so any gain is not attributable solely to teacher architecture.

## Development proposals and accepted-state matrices

The same gate requires incoming accuracy ≥75%, every previously accepted skill within five percentage points of its best accepted development score, and the narrow general probe within five points of the unchanged model. Each task has 32 development examples. Rollback is a failed update, not successful retention plus acquisition. Rejected experience may persist in bounded replay and contribute to later incidental learning.

### replay

| Stage | Proposed A | B | C | D | E | New skill | Gain from old | Worst earlier drop | Gate | Reason |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| A | 100.00% | 0.00% | 0.00% | 0.00% | 0.00% | 100.00% | 100.00% | 0.00% | accept | passed |
| B | 100.00% | 81.25% | 21.88% | 0.00% | 0.00% | 81.25% | 81.25% | 0.00% | accept | passed |
| C | 100.00% | 84.38% | 90.62% | 0.00% | 0.00% | 90.62% | 68.75% | 0.00% | accept | passed |
| D | 100.00% | 87.50% | 84.38% | 65.62% | 0.00% | 65.62% | 65.62% | 6.25% | reject | new skill below acquisition threshold; C exceeds retention tolerance; general probe exceeds tolerance |
| E | 100.00% | 93.75% | 84.38% | 71.88% | 100.00% | 100.00% | 100.00% | 6.25% | reject | C exceeds retention tolerance |

| After attempt | Accepted A | B | C | D | E |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 100.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| B | 100.00% | 81.25% | 21.88% | 0.00% | 0.00% |
| C | 100.00% | 84.38% | 90.62% | 0.00% | 0.00% |
| D | 100.00% | 84.38% | 90.62% | 0.00% | 0.00% |
| E | 100.00% | 84.38% | 90.62% | 0.00% | 0.00% |

### consolidation

| Stage | Proposed A | B | C | D | E | New skill | Gain from old | Worst earlier drop | Gate | Reason |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| A | 100.00% | 0.00% | 0.00% | 0.00% | 0.00% | 100.00% | 100.00% | 0.00% | accept | passed |
| B | 100.00% | 84.38% | 6.25% | 0.00% | 0.00% | 84.38% | 84.38% | 0.00% | accept | passed |
| C | 100.00% | 81.25% | 81.25% | 0.00% | 0.00% | 81.25% | 75.00% | 3.12% | accept | passed |
| D | 100.00% | 71.88% | 81.25% | 93.75% | 15.62% | 93.75% | 93.75% | 12.50% | reject | B exceeds retention tolerance |
| E | 100.00% | 87.50% | 81.25% | 15.62% | 100.00% | 100.00% | 100.00% | 0.00% | accept | passed |

| After attempt | Accepted A | B | C | D | E |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 100.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| B | 100.00% | 84.38% | 6.25% | 0.00% | 0.00% |
| C | 100.00% | 81.25% | 81.25% | 0.00% | 0.00% |
| D | 100.00% | 81.25% | 81.25% | 0.00% | 0.00% |
| E | 100.00% | 87.50% | 81.25% | 15.62% | 100.00% |

### composition

| Stage | Proposed A | B | C | D | E | New skill | Gain from old | Worst earlier drop | Gate | Reason |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| A | 100.00% | 0.00% | 0.00% | 0.00% | 0.00% | 100.00% | 100.00% | 0.00% | accept | passed |
| B | 100.00% | 78.12% | 21.88% | 0.00% | 0.00% | 78.12% | 78.12% | 0.00% | accept | passed |
| C | 100.00% | 84.38% | 93.75% | 0.00% | 0.00% | 93.75% | 71.88% | 0.00% | accept | passed |
| D | 100.00% | 84.38% | 62.50% | 75.00% | 0.00% | 75.00% | 75.00% | 31.25% | reject | C exceeds retention tolerance |
| E | 100.00% | 90.62% | 84.38% | 6.25% | 100.00% | 100.00% | 100.00% | 9.38% | reject | C exceeds retention tolerance |

| After attempt | Accepted A | B | C | D | E |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 100.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| B | 100.00% | 78.12% | 21.88% | 0.00% | 0.00% |
| C | 100.00% | 84.38% | 93.75% | 0.00% | 0.00% |
| D | 100.00% | 84.38% | 93.75% | 0.00% | 0.00% |
| E | 100.00% | 84.38% | 93.75% | 0.00% | 0.00% |

## Unseen-composition development signals

These scores are from proposals, including rejected proposals, rather than the final accepted checkpoints. Each eligible composition uses 32 development examples. They are not the final 128-example test scores.

| Method | D proposal: BDA | E proposal: BCE | E proposal accepted? |
| --- | ---: | ---: | --- |
| replay | 0.00% | 0.00% | False |
| consolidation | 0.00% | 0.00% | True |
| composition | 0.00% | 81.25% | False |

The composition E proposal combined skills successfully on many development examples but was rejected for sorting loss. Its rejected weights were discarded by the declared protocol, so its transfer result has not been confirmed on the independent final test. A larger development set and repeated runs are needed to assess this tradeoff; the gate was not relaxed after observing it.

## Formatting and generation limits

| Model | Invalid JSON among 896 task answers | Generation cap hits | Format-tolerant atomic mean |
| --- | ---: | ---: | ---: |
| unchanged | 896/896 | 0 | 2.03% |
| replay | 42/896 | 22 | 54.69% |
| consolidation | 0/896 | 0 | 75.94% |
| composition | 27/896 | 22 | 54.84% |

## Teacher acquisition and assimilation

Teachers are measured on the incoming skill only, using the same 32 development examples as the successor. This read-only diagnostic was registered separately and was not used to select checkpoints or change the gate. Missing observations are marked rather than inferred. New-teacher KL eligibility used teacher-forced answer-token checks, a different and easier criterion than free generation.

| Method | Stage | Teacher free generation | Successor free generation | New training examples eligible for KL |
| --- | --- | ---: | ---: | ---: |
| consolidation | A | 100.00% | 100.00% | 256/256 |
| consolidation | B | 81.25% | 84.38% | 242/256 |
| consolidation | C | 78.12% | 81.25% | 236/256 |
| consolidation | D | 87.50% | 93.75% | 241/256 |
| consolidation | E | 100.00% | 100.00% | 256/256 |
| composition | A | 100.00% | 100.00% | 256/256 |
| composition | B | 84.38% | 78.12% | 239/256 |
| composition | C | 93.75% | 93.75% | 244/256 |
| composition | D | 93.75% | 75.00% | 245/256 |
| composition | E | 100.00% | 100.00% | 256/256 |

## Sustained learnability diagnostic

E was learned alone from the untouched base and from each accepted state after the D attempt, with matching examples and a 98,304-token ceiling. This diagnostic branch did not feed into the stream. Each point uses 64 development examples; the early ceiling limits sensitivity to slower learning.

| Starting state | Half-budget E | Full-budget E |
| --- | ---: | ---: |
| fresh | 100.00% | 100.00% |
| replay | 98.44% | 100.00% |
| consolidation | 100.00% | 100.00% |
| composition | 100.00% | 100.00% |

## General probe and resource accounting

The 32-question multiple-choice likelihood probe is reused for development gating and final checking. It is **not an independent held-out general-capability test**. The legacy one-question JSON probe has a floor effect and is not used for the general gate.

| Model | Reused general MC |
| --- | ---: |
| Unchanged | 71.88% |
| replay | 71.88% |
| consolidation | 78.12% |
| composition | 75.00% |

All streams had the same 196,608 training-token ceiling per stage. Consolidation splits it equally between temporary acquisition and successor training. Teacher forwards add compute; equal token ceilings do not imply equal wall time, supervision, or total compute. Training times below exclude model loading, checkpoint serialization and evaluation. Diagnostic and plasticity branches are separate.

| Method | Actual stream training tokens | Training minutes | Cache seconds | Teacher forward tokens | Peak training allocated GiB | Replay tokens peak | Combined cache MiB peak | Final checkpoint bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| replay | 982619 | 22.11 | 0.00 | 0 | 2.11 | 15889 | 0.00 | 1203564233 |
| consolidation | 982375 | 22.07 | 83.27 | 307487 | 2.38 | 15889 | 3.65 | 1203564233 |
| composition | 982241 | 19.12 | 81.29 | 307487 | 2.40 | 15889 | 3.65 | 1203564233 |

Temporary acquired-teacher checkpoints observed by the storage audit were up to 1,203,564,233 bytes each. They were deleted after their stage. The final learner has one ordinary compact checkpoint plus replay capped at 128 examples and 16,000 serialized tokenizer tokens. The temporary distribution-cache ceiling was 16 MiB. Offline archives and evaluation reports are excluded from learner-accessible replay, and their disk usage is not claimed to satisfy that replay cap.

Additional timing and reserved-memory fields are in [resources.json](resources.json); checkpoint observations are in [storage-audit.jsonl](storage-audit.jsonl). These component timers are not a complete end-to-end cost estimate.

## Reproducibility and limits

- Model revision: `c1899de289a04d12100db370d81485cdf75e47ca`. One seed (42), one order (A→B→C→D→E), one configuration. CUDA execution was not fully deterministic; item confidence intervals do not capture training variation.
- Current data and bounded past memory were the learner inputs. No future or held-out solutions were used in training; the host retains an offline corpus for provenance and scoring.
- Old-teacher KL can preserve incorrect behavior on examples from a rejected skill. The grouped top-32-plus-tail KL is a compressed objective, not full-vocabulary distillation.
- The pairwise composition curriculum does not rehearse every primitive at every stage. This does not test every possible composition policy.
- The direct baseline was not exhaustively tuned. There is no direct replay-with-compositions control, and the teacher variants distribute train tokens across two models. This screen cannot isolate every mechanism or establish robust superiority.
- These are synthetic record operations on a compact model, not open-domain knowledge accumulation. Longer streams, multiple seeds/orders, better replay policies and broader independent general tests are needed before stronger claims.

## Artifacts and verification

- [Written steps](../../../ASSIMILATION_STEPS.md), [configuration](config.json), and [frozen versions and corpus/source hashes](frozen.json).
- [Machine-readable summary](summary.json) and [independent artifact audit](AUDIT.json) are included with this report. Full final predictions, per-example stream files, and branch outputs remain in the local experiment store and are not included in the Git push.
- [Independent artifact audit](AUDIT.json): frozen source/corpus hashes, train/test separation, bounded memory/cache, compact parameter count, acceptance gates, proposal process IDs, and rescoring of final predictions.
- Eight unit tests passed, including order-sensitive scoring and distillation/gate checks. A CUDA component smoke test saved and reloaded an ordinary merged compact model.
