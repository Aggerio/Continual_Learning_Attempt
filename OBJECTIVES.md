# Continual Learning Objectives

Status: initial research objectives; no experimental results yet.

Starting model: Qwen3-0.6B. Pin its exact checkpoint and measure its actual parameter count and resource requirements before experiments.

## Project goal

Develop a small language model that repeatedly acquires useful, durable capabilities from new experience while preserving earlier abilities and remaining able to learn, within explicit limits on deployed model size, learning memory, and computation.

Learning must persist in saved model parameters and generalize to unseen examples after a context reset and process restart. Evaluation of the compact model must work without a teacher, retrieved training examples, or a growing collection of external skill modules.

The long-term ambition is to approach the performance of larger models on a defined capability suite while retaining small-model efficiency. Matching an 8B model across all capabilities is not an initial success criterion.

## Measurable objectives

### 1. Acquire new capabilities

The model should improve on each newly introduced skill using only experience and feedback available at that point in the learning stream.

Measure held-out skill accuracy before and after learning, examples or tokens required to reach a declared accuracy target, and total update cost. Use verifiable answers or executable checks where possible. Improvements must extend beyond the examples used for training.

### 2. Retain previously learned capabilities

After every learning stage, the model should retain earlier skills without requiring renewed access to their full training datasets.

Evaluate all earlier skills on held-out examples at every stage. Report both average and worst-skill degradation relative to each skill's best earlier score, alongside final accuracy. A strong average must not conceal the loss of an individual skill.

### 3. Preserve the starting model's general abilities

Learning new skills should preserve useful language and instruction-following behavior already present in the starting checkpoint.

Establish a fixed general-capability evaluation suite before adaptation. Compare every stage with the unchanged starting checkpoint and report results by capability. Finite evaluations support retention claims only for the abilities they cover; they cannot establish that every previous capability survives.

### 4. Sustain the ability to learn

Repeated updates should preserve the model's ability to acquire subsequent capabilities.

Measure learning curves and learning cost throughout the stream. Compare a late-stage checkpoint learning a new skill with a fresh starting checkpoint learning that same skill under the same data and update budget. Where practical, vary task order to distinguish accumulated learning effects from differences in task difficulty.

Positive transfer and reduced learnability should both be reported. A comparison between an easy early skill and a harder late skill is insufficient evidence of lost plasticity.

### 5. Reuse and combine learned capabilities

The model should apply learned skills to unseen problems that require combinations of them.

Evaluate held-out combinations separately from individual skills. Distinguish combinations encountered during training from combinations reserved for evaluation. Compare methods with equivalent access to composition-training examples so that additional supervision does not masquerade as an architectural advantage.

Successful transfer supports a behavioral claim about generalization. It does not, by itself, prove that the model has learned a particular internal representation or algorithm.

### 6. Keep resource use bounded and visible

The initial target is a fixed deployed architecture and parameter count across learning stages. Temporary training capacity is permitted if it is removed before compact-model evaluation.

Declare limits before each experiment and account for:

- Deployed parameters, checkpoint bytes, precision, and inference memory.
- Response latency and throughput under fixed evaluation conditions.
- Replay storage in tokens and bytes, including generated or compressed examples.
- Persistent auxiliary state, including retained teachers, adapters, and optimizer state.
- Peak training memory, training tokens, update steps, and total computation or measured training time.

An offline archive kept for reproducibility must be distinguished from information accessible to the learner. Resource limits must not be silently increased as the stream grows.

### 7. Make learning persist independently

At evaluation boundaries, save and reload the compact checkpoint in a fresh process with cleared conversational state. Confirm that acquired capabilities remain available and that parameter count matches the declared deployment budget.

Keep task instructions consistent across comparisons. Instructions may define the task, but evaluation prompts must not supply the training demonstrations or solutions whose durable acquisition is being tested.

## Research question and candidate mechanisms

The central question is:

> Can repeated learning improve the acquisition, retention, transfer, and sustained learnability of Qwen3-0.6B beyond a strong bounded-replay baseline, within comparable resource budgets?

Temporary capacity followed by consolidation into the compact model is one candidate mechanism. A specific hypothesis is that consolidation using verified problems combining old and new skills reduces interference and improves transfer relative to consolidation using separate skill examples.

These are hypotheses to test. Architecture growth, hierarchical reasoning, distillation, and learned update rules are optional mechanisms. None is required merely to satisfy the project definition.

## Initial scope and evidence standard

Begin with a short stream of five objectively checkable skill stages, with known stage boundaries. Evaluate the starting model and all learned skills after each stage, including general-capability retention and restart persistence. This is an initial diagnostic of continual learning, not evidence of indefinite capability accumulation.

Establish sequential adaptation and bounded replay as baselines. Compare a proposed mechanism with those baselines using declared data access, memory limits, and compute budgets. Include acquisition and consolidation costs in its total. If budgets cannot be matched, report the difference and the resulting quality–cost tradeoff explicitly.

Before training, freeze numerical tolerances for retention, minimum useful acquisition, resource ceilings, and the improvement required over the strongest baseline. Select these using baseline measurements and available compute; they are not yet fixed by this document. Use development data for method selection and keep final test examples out of training, replay, checkpoint selection, and consolidation decisions.

Treat small differences as inconclusive unless supported by uncertainty estimates and repeated runs. Promising findings should be checked across multiple task orders and random seeds before broader claims.

## Conditions for claiming progress

A useful result must demonstrate acquisition while satisfying declared retention and resource constraints. A proposed mechanism should improve a measured tradeoff over the strongest relevant baseline, such as greater retention at comparable acquisition and cost, or faster acquisition at comparable retention and memory.

Preserving old scores by failing to learn new skills is not success. Neither is learning new skills while hiding substantial forgetting, increased deployment size, expanding replay storage, or additional teacher computation.

A single successful acquisition-and-consolidation cycle establishes a component check. A successful five-stage stream establishes a narrow sequential-learning result. Longer streams, different domains, and broader evaluations are required to support stronger claims.

## Deferred objectives

- Adaptation with unknown task boundaries and noisy or conflicting feedback.
- Low-latency updates during normal use, with measured feedback-to-improvement time.
- Structural adaptation under a fixed deployment budget.
- Broader natural-language and programming capabilities.
- Comparison with a pinned larger-model reference on the same evaluation suite.

The initial work does not aim to establish unlimited learning in finite capacity, universal absence of forgetting, autonomous architectural self-redesign, or equivalence to a larger model across all tasks.
