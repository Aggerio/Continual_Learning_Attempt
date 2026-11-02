# Temporary capacity and bounded external memory: second screen

Status: completed. Eleven unit tests, both CUDA component smoke tests, and the artifact audit passed. See the [measured report](results/capacity_memory/screen-001/REPORT.md).

This screen tests two requested paths. It preserves the first experiment and changes no acceptance rule after seeing results.

## Temporary capacity

1. Start both arms from the same saved composition-C compact model (selection/filter/sort learned). Hash its weights and all inputs before execution.
2. Generate fresh, globally deduplicated development/test examples, excluding the first experiment's entire corpus. Preserve the original incoming D/group and E/join training examples. Use 64 development and 64 final test examples per atomic skill and held-out composition.
3. Compare temporary acquisition rank 16 versus 64, with alpha/rank fixed at 2, identical learning rate and 98,304-token acquisition ceiling. This changes trainable temporary capacity; it does not add nonlinear transformer blocks.
4. Distill each teacher into a separate rank-16 successor initialized from the old compact model, using the same old/new cache objective, bounded replay and 98,304-token successor ceiling. No composition supervision in these two arms.
5. Independently reload and evaluate all five skills after D and E. Keep the 75% acquisition / five-percentage-point retention / five-point reused general-probe gate. Initialize earlier-skill references from fresh 64-example baseline scores; track the best accepted scores thereafter.
6. Record acquisition teacher scores, successor scores, raw and accepted-state matrices, time, allocated GPU memory, trainable/deployed parameters, bounded replay/cache sizes, and test results. Delete stage teachers/caches and rejected proposals. No teacher or adapter is used at inference.

This is a controlled two-boundary screen, not a fresh five-stage continual-learning pass. Equal training-token ceilings do not imply equal compute for different ranks.

## Bounded external memory

1. Receive A → B → C → D → E training experiences sequentially, maintaining a single memory of at most 128 examples and 16,000 serialized tokenizer tokens. The existing skill-stratified selection policy is fixed. No unbounded index or old-corpus retrieval is permitted.
2. Use a deterministic structural retriever over the actual retained examples. It reads operation names, visible field names and row counts from the query program. It never reads query answers, evaluation skill labels, or future examples, and never executes the query. Composite queries retrieve demonstrations covering constituent operations when available.
3. Insert up to four input/output demonstrations as chat turns, within a 1,536-token complete prompt ceiling. The model's actual context ceiling is 2,048 tokens. Count and report prompt tokens and retrieval trace/latency. No examples are cut mid-answer.
4. With the unchanged base weights, evaluate the five-stage memory stream on fixed 16-example development probes, retrieval on and off. Retrieval is a data-access intervention, not weight assimilation. Report visible/current and future skills separately.
5. After capacity-arm D and E decisions, evaluate each accepted compact state on the same probes with retrieval on/off using its actual bounded memory. This tests periodic consolidation plus memory without changing the frozen training recipe or acceptance gate.
6. Finally evaluate untouched test examples with retrieval on/off for the frozen base and each accepted successor. Use a separate deliberately corrupted-answer memory control on 32 test examples per atomic skill; keep its results separate from ordinary memory. This checks dependence on memory contents, not robustness to all bad memories.

The same logical memory used for replay supplies retrieval for the hybrid; it is not an additional bank. General MC queries do not contain record programs and receive no demonstrations, so their unchanged retrieval scores cannot establish broad safety of retrieval. Rejected D experience may remain in memory as declared in the first protocol.

## Interpretation

- Higher teacher accuracy alone does not establish successful compression or retention.
- Better retrieval-on scores with low retrieval-off scores indicate external-memory dependence.
- Better successor retrieval-off scores measure learned behavior, subject to acquisition and retention checks.
- One seed and a small probe set provide a screen. Promising changes need full streams, repeated seeds/orders, and broader factual/general tests.

## Measured outcome and reproduction

Each capacity arm accepted one of two incoming updates. Rank 16 accepted grouping and rejected join; rank 64 rejected grouping and accepted join. On the directly matched grouping boundary, the rank-16 successor scored 90.62% versus rank 64's 81.25%. This does not support larger adapters as a retention fix under this recipe.

Bounded retrieval raised frozen-base final atomic accuracy from 0% to 52.81%. For the two consolidated successors, retrieval changed final means from 74.38% to 75.31% and from 77.50% to 79.69%, with losses on some retained skills. The store peaked at 119 examples and 15,889 compact-serialized tokens. These are single-seed procedural results, not factual knowledge assimilation or unlimited continual learning.

To reproduce from the repository root with the existing environment and original checkpoint available:

```sh
.runtime/train-env/bin/python scripts/capacity_memory_run.py --config configs/capacity_memory.json --output results/capacity_memory/screen-002
```

This reuses the frozen seed/settings to reproduce the screen; it is not an independent-seed validation. The source snapshots, pinned input/checkpoint hashes and package versions are in the completed run's `frozen.json`. Its `diagnostics/audit.py` and `diagnostics/build_report.py` target `screen-001` explicitly. From the repository root, use `PYTHONPATH=. .runtime/train-env/bin/python` for the audit and `python3` for the report generator.
