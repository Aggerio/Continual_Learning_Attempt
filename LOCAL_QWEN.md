# Local Qwen3-0.6B inference

## Current verification status

Verified on 2026-10-31 (Asia/Kolkata): NVIDIA driver and loaded module both report 615.78.08. The installed `/usr/bin/ollama` and local API report 0.40.13, with the CUDA backend detecting the RTX 3060 Laptop GPU. GPU runs offload all 29 layers; benchmark reports record residency separately for each configuration.

The system service is stopped and starting it requires interactive authentication. These tests use a user-owned server bound to `127.0.0.1:11434`, with model manifests under `.runtime/models/` and symlinks to existing `/var/lib/ollama/blobs/` files. The system model files remain the source of those links. Runtime files are ignored by Git.

`ollama` on the current PATH resolves to an older `/usr/local/bin/ollama` (0.3.14). Use `/usr/bin/ollama` explicitly for these checks. The Python script talks directly to the API.

To restart this user-owned server when port 11434 is free:

```sh
OLLAMA_HOST=127.0.0.1:11434 OLLAMA_MODELS="$PWD/.runtime/models" /usr/bin/ollama serve
```

Alternatively, start the system service with `sudo systemctl start ollama`; it uses the original system model directory. Run only one server on port 11434.

## CUDA benchmark results

Completed 2026-10-31 local time using Ollama 0.40.13, a 2048-token context, thinking disabled, and three 128-token samples per configuration. All eight configurations completed (24 measured requests), with no benchmark errors.

| Model | Placement | Threads | Median decode tokens/s |
| --- | --- | --- | --- |
| Q4_K_M | GPU | 4 | 299.8 |
| Q4_K_M | GPU | 8 | 294.8 |
| Q4_K_M | CPU | 4 | 60.5 |
| Q4_K_M | CPU | 8 | 60.4 |
| Q8_0 | GPU | 4 | 244.3 |
| Q8_0 | GPU | 8 | 228.4 |
| Q8_0 | CPU | 4 | 41.1 |
| Q8_0 | CPU | 8 | 39.3 |

Selected Q4_K_M with automatic placement and four threads: 299.8 tokens/s, 20 ms median first visible token latency on warm requests. This is approximately 4.95 times the fastest CPU candidate in this run. `/api/ps` recorded 693,423,308 bytes of GPU residency for Q4 and 925,273,947 for Q8; CPU controls recorded zero. Server logs confirm all 29 layers offloaded on GPU runs, and `/usr/bin/ollama ps` reports 100% GPU for the selected model.

Evidence: `results/local_qwen/benchmark-20261030T183147Z.json`, `selected.json`, and `demo-20261030T183147Z.json`. Filenames use UTC. The selected configuration is saved for `chat`.

Three additional deterministic inference smoke checks are saved in `results/local_qwen/smoke-20261030T183147Z.json`. All three failed strict output checks: arithmetic returned the correct result with extra text; sorting returned the correct array in a Markdown fence; record filtering returned the wrong name (`A` instead of `B`) in a Markdown fence. These are model response failures; API requests completed. The speed selection is not a correctness recommendation, and these checks do not establish continual learning.

## Run on the host

With a current Ollama server running, execute from this directory:

```sh
python3 scripts/local_qwen.py setup
```

The script uses Python's standard library and the running Ollama API. It pulls `qwen3:0.6b` (Q4_K_M) and `qwen3:0.6b-q8_0`, warms and benchmarks each using automatic device placement and CPU-only execution at four and eight threads, then runs a demonstration with the fastest tested configuration. Model files are stored by the existing Ollama server in its configured model directory. Existing services and server settings are not changed.

Results, model metadata/digests, responses, errors, and selected settings are saved under `results/local_qwen/`. The setup unloads the tested Qwen models between configurations; do not benchmark while another client is using these models. It does not unload unrelated models. Competing workloads and resident models can affect the result.

To chat afterward:

```sh
python3 scripts/local_qwen.py chat
python3 scripts/local_qwen.py chat --prompt 'Explain continual learning in two sentences.'
```

For an alternate local endpoint, pass the same `--host http://127.0.0.1:PORT` to setup and chat.

If the server is outdated, update and restart Ollama using its [official Linux instructions](https://docs.ollama.com/linux). The script checks the server rather than inferring its version from the CLI. It does not upgrade system software.

## Performance scope

- Selection favors median warm generation throughput across the tested configurations. First-visible-token latency is also recorded. This is a short comparison, not proof of the machine's absolute maximum speed.
- A 2048-token context bounds memory use for short interactions. Thinking is disabled for low-latency responses; enable it separately when evaluating reasoning quality.
- Q4 and Q8 are quantized representations of the requested model. Their speed comparison does not establish equal answer quality.
- Automatic placement can fall back to CPU. Inspect `/api/ps` in the saved report and run `ollama ps` on the host to confirm GPU placement. A positive `size_vram` indicates some GPU residency, not necessarily full GPU offload. The script leaves the selected model warm for 30 minutes after use.
- Flash Attention and other server-level options are not silently changed. Further tuning requires host GPU access and benchmark evidence.
- This installs an inference representation through Ollama. Continual-learning training will need a separate trainable checkpoint and training runtime.

Sources: [Q4 model](https://ollama.com/library/qwen3:0.6b), [Q8 model](https://ollama.com/library/qwen3:0.6b-q8_0), [chat API](https://docs.ollama.com/api/chat), [GPU placement and configuration](https://docs.ollama.com/faq).
