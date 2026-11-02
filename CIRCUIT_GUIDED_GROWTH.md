# Use related computations to place temporary neurons

Status: implemented, trained and audited in `results/circuit_growth/screen-001`. This is distinct from the earlier fixed-location nonlinear-layer screen and its representation-matching compression arms.

The experiment causally validated lookup-related contributions at zero-based MLPs 19 and 18, but growing those interfaces did not improve acquisition or fixed-size assimilation over the controls in this run. Full evidence is in [the report](results/circuit_growth/screen-001/growth/REPORT.md), [localization audit](results/circuit_growth/screen-001/LOCALIZATION_AUDIT.json) and [growth audit](results/circuit_growth/screen-001/growth/AUDIT.json).

## Completed screen

- Discovery used 24 counterfactual pairs per family and validation used 32 per family: lookup, equality, copying and constant-output controls. Lookup passed the capability prerequisite; equality did not. Selected channels at both sites passed held-out specificity, random-channel and shuffled-donor controls in both patch directions. Attention heads were inspected and patched as well. This localizes a narrow final-prompt-position lookup contribution, not a complete join circuit or a map of uniquely semantic neurons.
- Three frozen-backbone teachers each added 393,216 native gated-channel parameters and consumed 98,189 input tokens. Related sites were 19/18; random sites were 15/12; fixed sites were 14/21. Untouched join accuracy was respectively 59/64, 64/64 and 64/64. Existing backbone digests remained identical. Bypassing both added banks exactly reproduced the original development join responses.
- Each teacher was compressed into a separate original-size learner. A matched direct learner used the same verified examples and schedule without new-task teacher targets. All four compact learners reached 64/64 test joins at exactly 596,049,920 parameters, but none passed all historical retention gates. The direct learner's sorting rejection inherited a deficit already present before this update; the guided compact learner also introduced substantial additional grouping loss.
- Grouping test accuracy was 58/64 before the update, 55/64 after direct learning, 44/64 after guided compression, 57/64 after random-placement compression and 48/64 after fixed-placement compression. All 20 guided-compact grouping failures preserved the correct group keys and shape but produced wrong totals. Guided compression scored 4/32 filter-join and 0/32 filter-sort-join, versus direct learning's 7/32 and 2/32.
- The audit rescored 26 fresh-process evaluation reports, verified checkpoint size and frozen evidence, and checked development-only acceptance. All 23 unit tests passed. Replay used 115 examples / 15,762 tokenizer tokens; peak combined target caches used 3,842,422 bytes against a 16 MiB ceiling and were deleted after training. Expanded teachers and research checkpoints remain on disk for inspection; compact inference loads no added banks or retrieval.

The next useful diagnostic is to trace the grouping computations before and after compression, then test a bounded preservation constraint on those computations. Placement alone did not solve retention. A later continual-learning claim still requires repeated boundaries, task orders and seeds with sustained learning and bounded resources.

Implementation and reproducibility:

```sh
.runtime/train-env/bin/python -m unittest discover -s tests
.runtime/train-env/bin/python -m continual.circuit_localization
.runtime/train-env/bin/python scripts/circuit_growth_run.py
.runtime/train-env/bin/python scripts/circuit_growth_report.py
```

The saved source/input manifests freeze this screen; do not edit its training or localization sources and then treat the existing results as evidence for the changed implementation. Report generation audits the completed artifacts. The protocol below records the hypothesis and controls used for this screen.

## Hypothesis

If a model already implements reusable parts of a new skill, adding capacity where those parts become available may improve acquisition, composition and retention per added parameter. Semantic similarity and high activation alone do not identify such a location. The experiment must first establish a causal contribution and then validate the placement on held-out examples.

For join, candidate primitives are key equality, matching a key across records, selecting a matching value, copying fields and combining records. These are hypotheses about the current checkpoint, not established circuit labels. The successful old tasks (selection, filtering and grouping) supply comparison behaviors. The earlier sorting localization is not assumed to locate join primitives.

## Discover and validate related computations

1. Use only discovery data to measure whether the frozen pre-join model can perform each primitive. Include executable primitive probes and contrast pairs that vary matching keys or lookup values, plus controls for formatting, token copying and field names. Keep sizes, answer lengths and schema changes balanced where possible. Probe multiple paraphrases, field renamings and row orders.
2. Inspect both attention outputs/heads and gated MLP channels. Activation or gradient attribution is a screening tool. Confirm candidates with ablation and prefix-aligned activation patching. Demand effects on the intended primitive across different surface forms, distinguish generic JSON/output corruption, and measure collateral effects on unrelated skills. Use same-model, shuffled-donor and random-component controls.
3. Freeze two candidate MLP insertion sites using discovery evidence only, then confirm the primitive effects on separate validation data. If there is no validated related circuit, report failed localization instead of assigning a skill name to the most active units. Evaluate final learned skills on another untouched cohort.
4. A location must have access to the relevant information. If the missing operation is cross-token key retrieval, test the attention path. A position-wise MLP expansion cannot be assumed to supply missing information routing.

## Add neurons within selected MLPs

Installed Qwen3 uses a gated MLP:

`MLP(x) = W_down [SiLU(W_gate x) * (W_up x)]`

Add a parallel bank of gated intermediate channels at the same normalized input and output interface:

`MLP_expanded(x) = MLP_old(x) + D_new [SiLU(G_new x) * (U_new x)]`

This is functionally a widening of that MLP. It differs from the previous residual MLPs attached after entire decoder blocks. Keep the old projections frozen; use separately registered new parameters to avoid optimizer weight decay modifying nominally frozen rows. Initialize `D_new` to zero so the expanded model initially reproduces the old model exactly. Use random new input projections with the same initialization rule in every placement arm; feature-based initialization is a separate future ablation.

For hidden width 1,024, adding 64 channels costs `3 * 1024 * 64 = 196,608` parameters per site. Two sites add 393,216 parameters. The old model has 596,049,920, giving 596,443,136 while expanded. These are real gated-channel additions, not a guarantee that each new channel represents one skill.

Hidden-unit index adjacency has no intrinsic meaning: channel order can be permuted with matching projection changes. The useful meaning of "there" is the computation's input/output interface and information flow.

## Controlled learning and compression

Compare three placement arms with exactly two sites and 64 new channels per site:

| Arm | Placement |
|---|---|
| Related circuit | Two sites frozen from causal discovery |
| Random | Two sites chosen once with a fixed seed and no test access |
| Fixed baseline | Original decoder layers 14 and 21, using the same new channel bank |

All arms use the same backbone, examples, replay bounds, sampling schedule, new parameter count, initialization distribution and training-token budget. Freeze every old weight. Include bypass controls for each new bank and both together, and verify unchanged old-weight digests. If selected sites overlap a baseline, report the overlap rather than silently changing the baseline or treating duplicate conditions as independent evidence.

Measure new-skill acquisition, all prior atomic tasks and prior composition behaviors, general probes, added/trainable parameters, tokens, cache storage, GPU memory and wall time. Independent final tests must include new combinations such as filter-join and filter-sort-join. The earlier random representation arm passed atomic acceptance but lost filter-group performance; atomics alone are not an adequate retention claim.

Only after acquisition succeeds, compress each expanded teacher into a separate original-width student using the same output-only method. Hold compression constant to isolate placement rather than simultaneously changing the representation loss. Add a direct-learning student control with the same verified examples and declared training budget but no expanded-teacher targets; report any trainable-capacity or compute differences explicitly. The completed screen did not establish that expanded-teacher targets were necessary because its students also received correct answers directly.

Remove expanded teachers and added channel banks from inference, save/reload each compact student, reset context and disable retrieval. A compact model must return to 596,049,920 parameters. Added nonlinear channels cannot generally be folded algebraically into old weights. Their removal requires successful compression or forfeits their contribution. For a continual-learning claim, repeat subsequent learning boundaries and measure sustained plasticity and bounded resources.

## Research basis and limits

- [Anthropic circuit-tracing methods](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) provide circuit hypotheses and intervention-based validation, not a complete map of safe editable addresses.
- [Knowledge Neurons in Pretrained Transformers](https://aclanthology.org/2022.acl-long.581/) studies factual attribution in BERT. It does not establish that Qwen procedural skills map to individual neurons.
- [Neural Knowledge Bank for Pretrained Transformers](https://arxiv.org/abs/2208.00399) is precedent for adding trainable FFN memory capacity to a frozen T5 model. It does not establish the proposed circuit-based placement rule or fixed-size continual assimilation.

Superposition and distributed computation mean related skills can share units with unrelated skills. Freezing existing weights does not freeze behavior: new channel outputs can still change downstream computations. Any reduction in interference or improvement in sample efficiency is a hypothesis to measure, not a property guaranteed by proximity.
