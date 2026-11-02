#!/usr/bin/env python3
"""Pull, benchmark, and chat with Qwen3 through a local Ollama service.

Uses only the Python standard library. Run on the host that runs Ollama.
"""

import argparse
import datetime
import json
import os
from pathlib import Path
import statistics
import sys
import time
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "local_qwen"
MODELS = ("qwen3:0.6b", "qwen3:0.6b-q8_0")


class Ollama:
    def __init__(self, base):
        self.base = base.rstrip("/")
        # Local inference should not travel through an HTTP proxy.
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def open(self, route, payload=None, timeout=600):
        body = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(
            self.base + route, data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            return self.opener.open(request, timeout=timeout)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            raise RuntimeError(f"{route}: HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Cannot reach Ollama at {self.base}: {exc.reason}. "
                "Run this script in your host terminal with Ollama running."
            ) from exc

    def request(self, route, payload=None):
        with self.open(route, payload) as response:
            result = json.load(response)
        if result.get("error"):
            raise RuntimeError(result["error"])
        return result

    def pull(self, model):
        print(f"Pulling {model}", flush=True)
        previous = None
        success = False
        with self.open("/api/pull", {"model": model, "stream": True}, 1800) as response:
            for line in response:
                event = json.loads(line)
                if event.get("error"):
                    raise RuntimeError(event["error"])
                total = event.get("total", 0)
                percent = int(100 * event.get("completed", 0) / total) if total else None
                status = event.get("status", "")
                progress = (status, percent // 10 if percent is not None else None)
                if progress != previous:
                    print(f"  {status}" + (f" {percent}%" if percent is not None else ""), flush=True)
                    previous = progress
                success |= status == "success"
        if not success:
            raise RuntimeError(f"Pull did not finish successfully: {model}")

    def chat(self, model, options, messages, max_tokens=128, emit=False):
        payload = {
            "model": model, "messages": messages, "think": False,
            "stream": True, "keep_alive": "30m",
            "options": {**options, "num_predict": max_tokens},
        }
        started = time.perf_counter()
        first = None
        chunks = []
        final = None
        with self.open("/api/chat", payload) as response:
            for line in response:
                event = json.loads(line)
                if event.get("error"):
                    raise RuntimeError(event["error"])
                content = event.get("message", {}).get("content", "")
                if content:
                    if first is None:
                        first = time.perf_counter() - started
                    chunks.append(content)
                    if emit:
                        print(content, end="", flush=True)
                if event.get("done"):
                    final = event
        if final is None or not chunks:
            raise RuntimeError("Ollama returned no completed visible response.")
        elapsed = time.perf_counter() - started
        final["text"] = "".join(chunks)
        final["wall_seconds"] = elapsed
        final["first_visible_token_seconds"] = first
        duration = final.get("eval_duration", 0) / 1e9
        final["decode_tokens_per_second"] = final.get("eval_count", 0) / duration if duration else 0
        return final


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def benchmark(api, repeats):
    version = api.request("/api/version")
    print(f"Ollama server: {version.get('version')}", flush=True)
    try:
        parts = tuple(int(p) for p in version["version"].split("-")[0].split(".")[:2])
    except (KeyError, ValueError):
        parts = (0, 0)
    if parts < (0, 9):
        raise RuntimeError(
            "This setup needs a current Ollama server supporting Qwen3 and think=false. "
            "Update Ollama and restart the server. Installed client and server versions can differ. "
            "Official instructions: https://docs.ollama.com/linux"
        )
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = RESULTS / f"benchmark-{timestamp}.json"
    report = {
        "created_at": timestamp, "endpoint": api.base, "server": version,
        "context_tokens": 2048, "repeats": repeats,
        "selection_metric": "median warm decode tokens per second",
        "limitations": [
            "Short single-request throughput screen, not an exhaustive maximum or quality benchmark.",
            "Q4 and Q8 alter weight precision; correctness is not established by speed.",
            "Automatic GPU placement is recorded from /api/ps; CPU fallback remains possible.",
            "Thinking disabled; scores do not characterize reasoning-mode performance.",
        ],
        "models": {}, "runs": [], "candidates": [],
    }
    for model in MODELS:
        api.pull(model)
        shown = api.request("/api/show", {"model": model})
        tags = api.request("/api/tags").get("models", [])
        report["models"][model] = {
            "show": shown,
            "manifest": next((item for item in tags if item.get("name") == model), None),
        }
        for placement in ("auto", "cpu"):
            for threads in (4, 8):
                options = {
                    "num_ctx": 2048, "num_thread": threads,
                    "temperature": 0.7, "top_p": 0.8, "top_k": 20,
                    "min_p": 0, "repeat_penalty": 1.0, "seed": 42,
                }
                if placement == "cpu":
                    options["num_gpu"] = 0
                label = f"{model} {placement} threads={threads}"
                print(f"\nWarming {label}", flush=True)
                try:
                    # Unload only this tested model to ensure placement changes take effect.
                    api.request("/api/generate", {"model": model, "keep_alive": 0})
                    warm = api.chat(model, options, [{"role": "user", "content": "Reply exactly READY."}], 32)
                    placement_info = api.request("/api/ps")
                    results = []
                    for repeat in range(repeats):
                        prompt = (
                            f"Trial {repeat + 1}. Write a long numbered list of practical tips "
                            "for organizing a desk. Continue for at least 80 items."
                        )
                        result = api.chat(model, options, [{"role": "user", "content": prompt}])
                        report["runs"].append({
                            "model": model, "placement": placement, "options": options,
                            "repeat": repeat, "result": result,
                        })
                        save(report_path, report)
                        if result.get("eval_count", 0) < 64:
                            raise RuntimeError("Response too short for a comparable throughput sample.")
                        results.append(result)
                        print(
                            f"  {result['decode_tokens_per_second']:.1f} tok/s; "
                            f"first visible token {result['first_visible_token_seconds']:.3f}s",
                            flush=True,
                        )
                    report["candidates"].append({
                        "model": model, "placement": placement, "options": options,
                        "median_decode_tokens_per_second": statistics.median(
                            r["decode_tokens_per_second"] for r in results),
                        "median_first_visible_token_seconds": statistics.median(
                            r["first_visible_token_seconds"] for r in results),
                        "placement_info": placement_info, "warm_response": warm["text"],
                    })
                except RuntimeError as exc:
                    report.setdefault("errors", []).append({"candidate": label, "error": str(exc)})
                    print(f"  Candidate failed: {exc}", file=sys.stderr, flush=True)
                finally:
                    api.request("/api/generate", {"model": model, "keep_alive": 0})
                    save(report_path, report)
    if not report["candidates"]:
        raise RuntimeError(f"No benchmark candidate completed. Inspect {report_path}")
    best = max(report["candidates"], key=lambda c: c["median_decode_tokens_per_second"])
    report["selected"] = best
    report["status"] = "completed"
    save(report_path, report)
    save(RESULTS / "selected.json", {**best, "benchmark_report": str(report_path), "endpoint": api.base})
    print(f"\nFastest tested: {best['model']} {best['placement']}, "
          f"{best['median_decode_tokens_per_second']:.1f} tok/s")
    demo = api.chat(best["model"], best["options"], [{
        "role": "user", "content": "Explain continual learning in two short sentences."
    }], 128)
    save(RESULTS / f"demo-{timestamp}.json", demo)
    print(f"\nLocal model response:\n{demo['text']}\n\nReport: {report_path}")
    print("Chat: python3 scripts/local_qwen.py chat")


def chat(api, prompt):
    selected_path = RESULTS / "selected.json"
    if not selected_path.exists():
        raise RuntimeError("Run 'python3 scripts/local_qwen.py setup' first.")
    config = json.loads(selected_path.read_text())
    if api.base != config["endpoint"]:
        raise RuntimeError("Endpoint differs from the benchmark. Use the benchmarked --host.")
    messages = []
    while True:
        if prompt is None:
            try:
                user = input("\nYou (Ctrl-D to exit): ").strip()
            except EOFError:
                break
        else:
            user = prompt
        if not user:
            if prompt is not None:
                break
            continue
        # Bound retained history. Ollama's 2048-token context also bounds each request.
        messages.append({"role": "user", "content": user})
        print("Qwen: ", end="", flush=True)
        result = api.chat(config["model"], config["options"], messages, 512, emit=True)
        print(f"\n[{result['decode_tokens_per_second']:.1f} tok/s]")
        messages.append({"role": "assistant", "content": result["text"]})
        messages = messages[-8:]
        if prompt is not None:
            break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("setup", "chat"))
    parser.add_argument("--host", default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--prompt", help="Single prompt instead of interactive chat")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    base = args.host if "://" in args.host else "http://" + args.host
    api = Ollama(base)
    try:
        if args.action == "setup":
            benchmark(api, args.repeats)
        else:
            chat(api, args.prompt)
    except (RuntimeError, TimeoutError, OSError) as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
