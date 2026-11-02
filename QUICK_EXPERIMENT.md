# Quick continual-learning experiment

This is a gated, single-seed development screen, followed by a five-stage comparison only if the isolated learning gate passes. It is not evidence of indefinite continual learning. Results belong to the pinned trainable Hugging Face checkpoint, not the Ollama quantization benchmark.

## Reproduce

The local environment is `.runtime/train-env` (created with system site packages). Exact package versions, Python version, model revision, source hashes, and generated-data hashes are recorded in each run's `manifest.json`. `configs/quick.json` contains the declared settings.

```sh
python3 -m venv --system-site-packages .runtime/train-env
.runtime/train-env/bin/python -m pip install peft
python3 -m unittest discover -s tests -v
.runtime/train-env/bin/python scripts/continual_quick.py all
```

PyTorch with CUDA, Transformers, and Hugging Face Hub must also be installed. This machine already provides them. The runner downloads `Qwen/Qwen3-0.6B` at revision `c1899de289a04d12100db370d81485cdf75e47ca`. An existing model download is reused. Free GPU memory before training; these measurements do not characterize concurrent Ollama serving.

Use `--output /absolute/path/to/new/run` for a new experiment and `--config /absolute/path/to/config.json` to initialize it. Existing runs use their copied configuration. `pilot` and `run` resume completed worker outputs; `run` refuses a failed acquisition gate. Re-running `all` resumes an initialized run. Do not change code or configuration during a run.

## Tasks and splits

Each prompt contains a fixed operation specification and a small JSON record program. Stages are A: select/rename, B: filter, C: ascending sort, D: grouped sum, E: matching-key join. A deterministic Python interpreter provides targets. Values, field choices, and record order vary across examples. These are bounded record programs, not evidence of broader algorithm learning or generalization to arbitrary lengths.

Each atomic skill has 256 training, 64 development, and 128 final test examples. All prompts are deduplicated across all splits and skills. A separate archive exists for reproducibility, while training workers receive only current examples and bounded replay. Test targets are used only by evaluation workers after the development gate and configuration freeze. The generator necessarily computes all targets; they are never used for checkpoint selection.

BA, BC, BD, and BE are available as synthetic composition supervision for the composition method. BDA (filter, group, select) and BCE (filter, sort, join) are held out of training entirely. The latter is evaluated only after E; the former after D. The same evaluation prompts are used for every method.

Strict parsed-JSON accuracy is primary. Correct JSON inside a Markdown fence is counted separately as a formatting-tolerant diagnostic. Array order matters; object key order does not. Booleans and numbers are distinct. Generation-cap hits and full responses are recorded.

## Development gate

First evaluate the unchanged checkpoint on the first 32 development examples per atomic skill and a fixed 16-question general probe. Then independently train each skill from the original checkpoint and evaluate on the same development subsets. Final tests remain unscored.

The provisional gate requires at least 75% strict accuracy and at least 10 percentage points of both strict and formatting-tolerant semantic gain for every skill. This prevents formatting fixes alone from passing as new skill acquisition. Skills with a strong starting score may lack headroom; failure here requires task or budget revision on development data, not a claim of forgetting. These small samples are for feasibility, not precise capability estimates. The gate does not itself require general retention; general changes are reported for diagnosis.

## Learning and comparison

Training uses bf16 LoRA, rank 16, on attention and MLP projections. Each stage begins from the previous compact checkpoint, trains a temporary adapter, merges it into the base weights, and saves the resulting ordinary model. There is no accumulated adapter bank. Optimizer state is discarded at stage boundaries for every method. Evaluation always loads the saved model in a separate process with empty conversation state.

The stage budget is capped at 49,152 non-padding training tokens, counting prompts and targets; the last example is omitted if it would exceed the cap, leaving less than one maximum sequence (512 tokens) unused. Actual tokens, supervised answer tokens, updates, losses, training time, peak allocated GPU memory, and parameter counts are recorded. These token caps approximate comparable compute; differing sequence lengths, output ratios, and replay-generation overhead prevent an exact compute match.

Methods:

1. Sequential: current skill only.
2. Replay: approximately half current and half replay examples, with replay sampled from a bounded, skill-stratified memory.
3. Composition: the same expected current/replay ratio and memory ceilings, with verified old/new combinations eligible for replay at the current stage.

The persistent replay ceiling is both 128 examples and 16,000 tokenizer tokens, counting serialized examples including answers and metadata. Byte size is also recorded. Each memory update sees only the previous bounded memory plus the current stage's data; it cannot re-read discarded old training examples. Composition examples use known operation semantics and current/previous skills. This is an explicit extra supervision assumption, not an architecture comparison. Python generation/storage overhead should be considered when interpreting quality against cost.

After the pilot passes, freeze settings before scoring the final test. Evaluate all introduced atomic skills, eligible held-out combinations, and the general probe after every stage. The unchanged checkpoint is the final-test reference. For plasticity, branch from each checkpoint after D and train on E alone, matching the fresh-E pilot's data and token budget. The sequential method's normal E run already satisfies this; replay methods require separate current-only branches. Evaluate at roughly half and full budgets on the same development subset. Exact token counts for half checkpoints are saved. Branch costs are additional diagnostic costs and must be reported separately from each stream's training cost.

The retained deployment has the original parameter count and architecture. Its numerical weights change, and bf16 adapter merges introduce rounding. Checkpoint and adapter archives are retained only for reproducibility, with only the current compact checkpoint passed to subsequent stages. Old large merged checkpoints are removed after their successor is trained and evaluated.

## Interpretation

Report acquisition, the complete after-stage accuracy matrix, worst earlier-skill degradation, general-probe change, composition accuracy, and the fresh/late-E curves. The provisional retention and general-probe tolerances are 5 percentage points; the candidate improvement target is 5 points over ordinary replay with comparable acquisition. Treat small differences as inconclusive. These tolerances must be judged alongside sample uncertainty, resource differences, and follow-up seeds/orders.

The original 16-question general probe is small and JSON-focused. Its initial strict baseline was zero, so it cannot support a useful retention comparison. During the development pilot, a supplementary 32-question multiple-choice probe was added before any final-test scoring. It uses restricted next-token likelihood over the single-token labels A/B/C/D, separating answer preference from JSON formatting. Final stream general-retention tolerances use that probe. It is still narrow and does not establish broad language retention. Evaluation reports include source hashes, and the final freeze captures updated source hashes alongside the original pilot manifest.

The first task order is fixed A to E; configuration changes to order require reviewing composition eligibility. Passing this screen motivates a longer, multi-seed experiment; it does not justify a claim that continual learning is solved.

Implementation references: [Qwen model and chat template](https://huggingface.co/Qwen/Qwen3-0.6B), [PEFT LoRA and merging](https://huggingface.co/docs/peft/package_reference/lora).
