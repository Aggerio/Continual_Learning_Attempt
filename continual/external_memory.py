"""Bounded structural demonstration retrieval; never executes a query program."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from continual import worker
from continual.tasks import SPEC, compact


def structure(prompt):
    try:
        p = json.loads(prompt)
        if not isinstance(p, dict) or not isinstance(p.get('ops'), list) or not isinstance(p.get('rows'), list):
            return None
        ops = tuple(x['op'] for x in p['ops'])
        fields = set(k for row in p['rows'] for k in row)
        return ops, fields, len(p['rows'])
    except (ValueError, KeyError, TypeError):
        return None


def retrieve(prompt, memory, limit=4):
    query = structure(prompt)
    if query is None:
        return []
    ops, fields, n = query
    candidates = []
    for item in memory:
        s = structure(item['prompt'])
        if s is None or not set(ops).intersection(s[0]):
            continue
        candidate_ops, candidate_fields, size = s
        similarity = (500 * (ops == candidate_ops) + 50 * len(set(ops).intersection(candidate_ops))
                      + 10 * len(fields & candidate_fields) / max(1, len(fields | candidate_fields))
                      - abs(n-size))
        candidates.append((similarity, item['id'], item, candidate_ops))
    candidates.sort(key=lambda x: (-x[0], x[1]))
    chosen, covered = [], set()
    for op in ops:
        if op in covered or len(chosen) >= limit:
            continue
        found = next((x for x in candidates if op in x[3] and x[2] not in chosen), None)
        if found:
            chosen.append(found[2])
            covered.update(found[3])
    for _, _, item, _ in candidates:
        if len(chosen) >= limit:
            break
        if item not in chosen:
            chosen.append(item)
    return chosen


def chat(tokenizer, item, shots):
    messages = [{'role': 'system', 'content': item.get('system', SPEC)}]
    for example in shots:
        messages += [{'role': 'user', 'content': example['prompt']},
                     {'role': 'assistant', 'content': compact(example['answer'])}]
    messages.append({'role': 'user', 'content': item['prompt']})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def bounded_prompt(tokenizer, item, memory, limit, token_cap):
    shots = retrieve(item['prompt'], memory, limit)
    while True:
        text = chat(tokenizer, item, shots)
        tokens = len(tokenizer.encode(text, add_special_tokens=False))
        if tokens <= token_cap:
            return text, shots, tokens
        if not shots:
            raise RuntimeError('Query alone exceeds retrieval prompt cap; refusing truncation.')
        shots.pop()


def evaluate(args, config):
    memory = json.loads(Path(args.memory).read_text())
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.source, local_files_only=True)
    memory_tokens = len(tokenizer.encode(compact(memory), add_special_tokens=False))
    if len(memory) > config['replay_examples'] or memory_tokens > config['replay_tokens']:
        raise RuntimeError('External memory exceeds persistent budget.')
    original = worker.prompt_text
    traces, elapsed = {}, []

    def render(tok, item):
        started = time.perf_counter()
        if args.mode == 'on':
            text, shots, tokens = bounded_prompt(tok, item, memory, config['retrieval_examples'],
                                                config['retrieval_prompt_tokens'])
        else:
            text, shots = original(tok, item), []
            tokens = len(tok.encode(text, add_special_tokens=False))
        elapsed.append(time.perf_counter()-started)
        if 'id' in item:
            traces[item['id']] = {'memory_ids': [x['id'] for x in shots], 'prompt_tokens': tokens}
        return text

    worker.prompt_text = render
    worker.evaluate(args, config)
    report = json.loads(Path(args.output).read_text())
    report['external_memory'] = {'mode': args.mode, 'memory_sha256': hashlib.sha256(Path(args.memory).read_bytes()).hexdigest(),
        'examples': len(memory), 'tokens': memory_tokens, 'bytes': Path(args.memory).stat().st_size,
        'retrieval_seconds': sum(elapsed), 'traces': traces,
        'max_prompt_tokens': max([v['prompt_tokens'] for v in traces.values()], default=0),
        'worker_pid': os.getpid(), 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    worker.write(args.output, report)


def main():
    p = argparse.ArgumentParser()
    for name in ['config', 'source', 'data', 'output', 'memory']:
        p.add_argument('--'+name, required=True)
    p.add_argument('--mode', choices=['on', 'off'], required=True)
    a = p.parse_args()
    evaluate(a, json.loads(Path(a.config).read_text()))


if __name__ == '__main__':
    main()
