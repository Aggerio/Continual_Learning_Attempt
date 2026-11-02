# Temporary capacity and bounded external memory: measured screen

Two requested paths were tested with fresh evaluation data. The capacity comparison starts from the same saved A–C model and attempts D/group then E/join. It is a targeted two-boundary screen, not a fresh five-stage capacity pass. The external-memory experiment receives A→B→C→D→E with frozen base weights. Hybrid checks combine periodic consolidation with the same bounded store.

All atomic and composition final tests have 64 fresh examples each; development gates use 64, and chronological probes use 16. Previous-corpus fingerprints were excluded. All methods share exact JSON/order scoring and 192 output-token ceiling. The saved compact model has 596,049,920 parameters. This is one seed/order/configuration.

## Measured answer

Increasing teacher rank did not make both incoming updates pass. The directly matched grouping boundary favored rank 16 over rank 64. The later join boundary followed different retained checkpoints, so it cannot isolate rank alone.

Frozen-base memory changed final atomic accuracy from 0.00% with retrieval off to 52.81% with retrieval on. This difference is external context use, not knowledge incorporated into weights.

For each consolidated successor, inspect the per-skill on/off table below: new-skill gains can coexist with losses on previously learned skills. These are descriptive single-run results, not robust estimates of superiority.

## Path 1: change temporary learning capacity

The temporary learner used rank 16 (10,092,544 trainable parameters) or rank 64 (40,370,176). Both successors used rank 16, initialized separately from the old compact model, with no new composition supervision during D/E updates. Acquisition and consolidation each had a 98,304-token ceiling. Alpha/rank remained 2. Higher rank adds adapter capacity, not nonlinear transformer blocks. Teacher inference and larger-rank compute prevent a claim of matched total compute.

Fresh baseline scores for the common initial checkpoint: A 100.00%, B 76.56%, C 89.06%, D 0.00%, E 0.00%.

| Teacher rank | Incoming | Teacher dev | Successor A | B | C | D | E | Worst prior drop | Decision | Reason |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 16 | D | 96.88% | 98.44% | 84.38% | 85.94% | 90.62% | 1.56% | 3.12% | accept | passed |
| 16 | E | 100.00% | 100.00% | 89.06% | 82.81% | 90.62% | 100.00% | 6.25% | reject | C exceeds retention tolerance |
| 64 | D | 84.38% | 98.44% | 85.94% | 78.12% | 81.25% | 1.56% | 10.94% | reject | C exceeds retention tolerance |
| 64 | E | 100.00% | 100.00% | 85.94% | 90.62% | 6.25% | 100.00% | 0.00% | accept | passed |

The unchanged gate requires incoming accuracy ≥75%, every previously accepted skill within five percentage points of its best accepted development score, and the reused narrow MC probe within five points of the original base score. A rejected update remains a failure; the old model is retained. Rejected skill examples may enter bounded replay.

| Teacher rank | Accepted updates | Final retrieval-off A | B | C | D | E | Atomic mean | BDA | BCE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 1/2 | 100.00% | 89.06% | 92.19% | 84.38% | 6.25% | 74.38% | 0.00% | 0.00% |
| 64 | 1/2 | 100.00% | 89.06% | 93.75% | 4.69% | 100.00% | 77.50% | 0.00% | 15.62% |

BDA additionally changes selection from one output field in training to two. Its score therefore mixes composition and an arity shift. BCE is the cleaner reserved filter→sort→join check. Neither reserved composition entered training or memory.

## Path 2: bounded external memory

Memory held at most 128 examples and 16,000 compact-serialized tokenizer tokens, including answers/metadata. There was no separate growing index. The structural retriever used visible operation names, field names and row counts; it did not execute queries or read their answers/skill labels. Up to four retained demonstrations were inserted as chat turns, with a 1,536-token prompt ceiling and 2,048-token context ceiling.

Frozen-base five-stage development probes: scores below use the same 16 questions per task. "Seen" includes every received skill, regardless of whether a weight update would have passed. Future skills are shown separately. The off-reference uses the same unchanged weights and prompts and was evaluated once, then reused.

| After memory receives | Retained examples | Compact memory tokens | Seen off mean | Seen on mean | Future off mean | Future on mean | On: A | B | C | D | E |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 119 | 15780 | 0.00% | 100.00% | 0.00% | 0.00% | 100.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| B | 118 | 15841 | 0.00% | 56.25% | 0.00% | 0.00% | 100.00% | 12.50% | 0.00% | 0.00% | 0.00% |
| C | 116 | 15887 | 0.00% | 41.67% | 0.00% | 0.00% | 100.00% | 18.75% | 6.25% | 0.00% | 0.00% |
| D | 115 | 15762 | 0.00% | 37.50% | 0.00% | 0.00% | 100.00% | 12.50% | 12.50% | 25.00% | 0.00% |
| E | 111 | 15889 | 0.00% | 47.50% | n/a | n/a | 100.00% | 25.00% | 12.50% | 31.25% | 68.75% |

Final independent test, with the exact same questions and model weights within each on/off pair:

| Weights | Retrieval | A | B | C | D | E | Atomic mean | BDA | BCE | Invalid JSON / 448 task cases | Format-tolerant atomic mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| frozen base | off | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 448 | 0.94% |
| frozen base | on | 98.44% | 29.69% | 23.44% | 25.00% | 87.50% | 52.81% | 0.00% | 0.00% | 14 | 52.81% |
| rank-16 successor | off | 100.00% | 89.06% | 92.19% | 84.38% | 6.25% | 74.38% | 0.00% | 0.00% | 5 | 74.38% |
| rank-16 successor | on | 100.00% | 79.69% | 82.81% | 73.44% | 40.62% | 75.31% | 0.00% | 0.00% | 2 | 75.31% |
| rank-64 successor | off | 100.00% | 89.06% | 93.75% | 4.69% | 100.00% | 77.50% | 0.00% | 15.62% | 0 | 77.50% |
| rank-64 successor | on | 96.88% | 84.38% | 89.06% | 31.25% | 96.88% | 79.69% | 0.00% | 29.69% | 0 | 79.69% |

Retrieval-on gains measure access to external examples. Only retrieval-off scores measure behavior available from the saved weights alone. No memory-stream weight training occurred. Hybrid successors were consolidated after each incoming boundary only when the same retrieval-off gate passed; retrieval did not influence acceptance. The replay and retrieval store are the same logical memory, not two banks.

Hybrid chronological probes after each consolidation decision:

| Rank | After attempt | Seen skills off mean | Seen skills on mean | New skill off | New skill on |
| ---: | --- | ---: | ---: | ---: | ---: |
| 16 | D | 90.62% | 75.00% | 87.50% | 56.25% |
| 16 | E | 72.50% | 71.25% | 0.00% | 37.50% |
| 64 | D | 65.62% | 60.94% | 0.00% | 6.25% |
| 64 | E | 70.00% | 73.75% | 100.00% | 100.00% |

## Corrupted-memory control

The control replaces every record-program memory answer with an empty array, keeping the real memory unchanged. This deliberately obvious corruption is not a general robustness benchmark. Scores below compare the same first 32 test questions per atomic skill; shorter false demonstrations also change prompt length.

| Weights | Retrieval off | Correct-memory retrieval | Empty-answer memory retrieval |
| --- | ---: | ---: | ---: |
| frozen base | 0.00% | 51.88% | 0.00% |
| rank-16 successor | 71.88% | 75.00% | 55.62% |
| rank-64 successor | 77.50% | 75.62% | 67.50% |

## Resources and limitations

| Teacher rank | Actual acquisition + successor tokens | Training minutes | Cache preparation seconds | Peak allocated training GiB | Max combined cache MiB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 392921 | 7.33 | 36.09 | 2.38 | 3.65 |
| 64 | 392921 | 8.05 | 37.47 | 2.56 | 3.65 |

Final evaluation cost includes generation; component timers exclude loading/serialization. Retrieval context is deliberately larger than the original training context and is an additional inference resource.

| Weights / retrieval | Evaluation seconds | Prompt tokens mean / max | Retrieval CPU seconds | Peak allocated evaluation GiB |
| --- | ---: | ---: | ---: | ---: |
| frozen base off | 172.65 | 167.8 / 206 | 0.288 | 1.40 |
| frozen base on | 124.66 | 488.9 / 707 | 1.132 | 1.87 |
| rank-16 successor off | 99.13 | 167.8 / 206 | 0.258 | 1.49 |
| rank-16 successor on | 129.73 | 488.9 / 707 | 1.061 | 1.87 |
| rank-64 successor off | 89.80 | 167.8 / 206 | 0.269 | 1.36 |
| rank-64 successor on | 124.00 | 488.9 / 707 | 1.135 | 1.87 |

- Same-size successor weights were reloaded in independent CUDA evaluation processes; temporary teachers and probability caches were deleted. Rank 64 is a larger adapter, not an added-block architecture.
- After D, the arms can follow different accepted checkpoints because of rollback. D is the directly matched rank comparison; E compares gated trajectories and cannot isolate teacher rank at an identical starting model.
- The common A–C checkpoint was selected from the previous composition arm and already learned BA/BC pairs as well as atomic A–C. No new compositions entered the capacity arms. Fresh evaluation prevents exact prompt reuse; it does not make this a new full-stream replication or remove selection effects.
- One seed, small generated programs and 64 test cases per task cannot establish robust superiority, unlimited retention, or open-domain factual learning.
- Sixteen fixed arithmetic anchors share the store ceiling and are not selected for program demonstrations. This first memory policy was not exhaustively tuned.
- Memory contains examples of operations; incoming query facts/rows are already in the prompt. This tests bounded episodic demonstrations and procedural transfer, not retrieval of newly invented world facts.
- The general MC probe is narrow and reused from the first gate. Non-program general queries receive no demonstrations, so on/off MC equality does not establish broad retrieval safety.
- Conditional teacher-forced eligibility and grouped top-32-plus-tail KL are retained from the first screen; neither guarantees free-generation correctness or preservation of all old behavior.

## Verification and artifacts

Eleven unit tests and CUDA rank-64/retrieval smoke tests passed. The [independent audit](AUDIT.json) rescored 32 evaluation reports, reconstructed retrieval traces, checked bounded chronological memory and data separation, recalculated gates, and verified fixed compact weight counts.

[Protocol](../../../CAPACITY_MEMORY_STEPS.md) · [Frozen inputs/sources](frozen.json) · [Progress and raw decisions](progress.json). Per-example capacity, retrieval-memory, and control files remain in the local experiment store and are excluded from the Git push to avoid uploading bulky raw artifacts.
