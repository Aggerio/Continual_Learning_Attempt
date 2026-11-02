# Temporary nonlinear-layer assimilation screen

Status: completed; 19 tests and the independent audit passed. Results: [audited report](results/nonlinear_assimilation/screen-002/REPORT.md). The expanded teacher solved 63/64 fresh test joins versus 3/64 originally, without updating the backbone. All three original-size students solved 64/64 joins after compression; output-only and random-location matching passed development acceptance, while guided matching failed the sorting retention gate. Selected guided locations did not give a consistent acquisition/retention advantage over the random control. Composition transfer remained weak (compact students: 4/32 filter-join and 0/32 filter-sort-join); the random arm also lost filter-group performance from 12/32 to 4/32. Atomic acceptance does not establish broad retention. This remains a single-update component result.

Results directory: `results/nonlinear_assimilation/screen-002`. The first launch (`screen-001`) failed while logging a path before any teacher training began; its frozen files and error are preserved. Only path serialization in status logging was fixed before restarting the same recipe.

This is one fixed grouping-to-join learning boundary on synthetic JSON table operations. It tests temporary nonlinear capacity and subsequent compression. It does not establish continual learning, future learnability or factual knowledge editing.

## Fixed architecture and training

1. Start from the accepted rank-16 grouping checkpoint used in the previous diagnosis. Freeze every backbone parameter.
2. Add residual branches after original decoder layers 14 and 21 (zero based): `h + W2 SiLU(W1 RMSNorm(h))`. Hidden width is 1,024; bottleneck is 256. `W2` starts at zero, so the initial expanded model is exactly the original model. The 1,050,624 new parameters train in FP32; the backbone runs in BF16. The expanded model has 597,100,544 parameters.
3. Train only these added parameters for at most 98,304 input tokens, using a frozen 50:50 sampling schedule of incoming join examples and bounded old replay. The schedule, examples, source files and checkpoint hashes are frozen before training. The replay has 115 examples, including 16 arithmetic anchors, within 128-example/16,000-token ceilings. Teacher learning rate is 0.001. Cross-entropy uses executable correct answers; old examples also have old-teacher KL with weight 0.5. The old backbone digest must remain unchanged.
4. Require at least 75% autonomous join accuracy and at least 10 percentage points improvement on development data. Independently bypass each added layer and both layers. Bypassing both must exactly reproduce original responses on the same ablation cohort. Evaluate the expanded model on the separate test set as well. If acquisition fails, stop before compression and report the failed acquisition.
5. If acquisition succeeds, train three separate original-size students from the same old checkpoint, with rank-16 adapters merged at export. Each has the exact same examples, sampling schedule, token budget, learning rate 0.0002 and output-distribution targets. There are no added layers in these students.

The compression arms are:

| Arm | Extra representation objective |
|---|---|
| Output | None; cross-entropy plus old/new teacher KL |
| Guided | Hidden states after original layers 2, 5 and 17 |
| Random | Same objective after original layers 9, 21 and 23 |

The locations were fixed by the prior diagnostic's selected and random modules, with no new site search. Matching a whole decoder boundary is a surrogate for the earlier module intervention; the weak diagnostic result does not guarantee these locations help.

Both KL terms have weight 0.5, temperature 2 and 32 vocabulary bins plus a complementary bin. Guided/random hints use coefficient 0.1 on mean per-token squared distance between normalized hidden vectors, averaged over three boundaries. On old examples, targets come from the old compact teacher; on incoming examples, they come from the expanded teacher. New-teacher targets are enabled only when all teacher-forced answer predictions are correct; verified-answer cross-entropy remains active for all examples. State matching uses the same one-token-shifted answer-prediction positions as KL. At insertion boundaries, expanded-teacher states are captured after the extra branch. No gold prefixes enter autonomous evaluation.

## Data, gates and resource accounting

- Freeze 64 fresh development examples per atomic task. Test uses another 64 per atomic task plus 32 each of six held-out compositions: BA, BC, BD, BE, BDA and BCE. No composition is supplied to this boundary's training, although the earlier backbone has learned BA/BC. The incoming-join combinations BE/BCE test new transfer. Both cohorts exclude historical archive and diagnostic fingerprints. The fixed 32-question general MC probe is reported separately; it is not a fresh general benchmark.
- Student acceptance uses development data only: join at least 75%, each old skill within 5 percentage points of the best accepted historical checkpoint re-evaluated on this same development cohort, and the fixed general probe within 5 points of the before model. Final test never selects an arm or hyperparameter. All candidates are retained as research artifacts; no model is automatically deployed.
- Output and representation targets share a declared combined 512 MiB temporary cache ceiling, larger than the earlier output-only cache because hidden states are included. Record actual bytes and teacher forward tokens. Delete tensor caches after the run.
- Reload each exported student in a fresh process using the ordinary evaluator, with retrieval off and no teacher, branch hooks, training adapters, optimizer or previous context. The parameter count must return to 596,049,920.
- Record per-skill and paired-query results, random-control comparisons, trainable/temporary/final parameter counts, CUDA memory, token counts, training time, source/input hashes, teacher ablations and unchanged-backbone checks. Query resampling uncertainty does not cover training seeds; this is one seed and one boundary.

Teacher acquisition uses replay in this experiment; the earlier LoRA acquisition teacher used incoming examples alone. Comparing those teachers is exploratory rather than a clean capacity-only comparison. The three compression arms within this run are matched on data, teachers and sampling schedule; representation losses add compute even at the same token budget.

Run with `.runtime/train-env/bin/python scripts/nonlinear_assimilation_run.py --output results/nonlinear_assimilation/screen-002`. The implementation and checkpoints use direct Hugging Face/PyTorch CUDA access rather than the Ollama API.
