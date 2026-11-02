# Isolated-skill development pilot

Status: completed. Seed 42; 32 held-out development examples per skill.

| Skill | Before | After | Gain | Gate | Training seconds |
| --- | ---: | ---: | ---: | --- | ---: |
| A | 0.0% | 100.0% | +100.0% | pass | 58.4 |
| B | 0.0% | 68.8% | +68.8% | fail | 54.8 |
| C | 0.0% | 68.8% | +68.8% | fail | 54.3 |
| D | 0.0% | 84.4% | +84.4% | pass | 54.8 |
| E | 0.0% | 100.0% | +100.0% | pass | 47.7 |

Gate: at least 75% strict accuracy and 10 percentage points of both strict and semantic gain on every skill.

Next action: revise_on_development_data; final test remains unopened.

These independent updates do not measure sequential forgetting. Final test examples have not been scored by this pilot.
Strict correctness requires parseable JSON with the correct value and array order. Formatting-tolerant accuracy and full responses are retained in each evaluation JSON.

The general probe has only 16 fixed questions and does not establish broad general retention.

Exact configurations, checkpoint identity, data/source hashes, training costs, and evaluation reports are saved alongside this report.

## Findings and next experiment

The first screen fails the declared acquisition gate: filtering and sorting each score 22/32, below the required 24/32. Selection/renaming and joining score 32/32; grouped sums score 27/32. All post-training atomic answers are valid JSON, so the remaining atomic failures are content errors rather than Markdown formatting.

Filtering sometimes retains rows that fail its predicate. Sorting sometimes sorts numeric values while corrupting which categorical field belongs to each row. Full prompts, expected outputs, and responses are available in the archived development data and evaluation reports.

A useful next calibration is more training on B/C using development data, checking whether these content errors fall while keeping the final test unscored. The sequential/replay/composition runner is implemented but has not been executed or GPU-validated end to end. Do not treat it as an experimental result.

Five independent updates used 245,183 non-padding training tokens and 270.0 seconds of measured training time, excluding initial download, process startup, evaluation, and final export. The deployed checkpoint has 596,049,920 parameters before and after adapter merging. Peak PyTorch allocated training memory was 2.11 GiB; this excludes CUDA-reserved but unallocated memory.

## General-probe calibration

The original JSON-based probe has a zero strict baseline and is unsuitable for a useful retention conclusion. A supplementary 32-question multiple-choice probe was added during development, after A/B evaluations, using restricted next-token likelihood over A/B/C/D. The original probe results remain available rather than being replaced.

| Checkpoint | Multiple-choice score |
| --- | ---: |
| Unchanged model | 71.9% |
| Independently trained C | 71.9% |
| Independently trained D | 71.9% |
| Independently trained E | 75.0% |

A/B lack this supplementary measurement. These narrow, noisy samples do not establish general capability preservation. Original provenance is in `manifest.json`; the development amendment and final source snapshot are in `development_changes.json` and `source-final/`. No final-test evaluation or sequential stream was run.
