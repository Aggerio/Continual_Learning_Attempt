"""GPU workers; training and each checkpoint evaluation run in separate processes."""
import argparse
import hashlib
import json
import os
import math
from pathlib import Path
import random
import time

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

from continual.tasks import SPEC, compact, general_mc_examples, score

WORKER_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
TASK_SOURCE_SHA256 = hashlib.sha256(Path(__file__).with_name('tasks.py').read_bytes()).hexdigest()


def wilson(successes, n):
    if n == 0:
        return [0.0, 1.0]
    z = 1.959963984540054
    p = successes / n
    denominator = 1 + z*z/n
    center = (p + z*z/(2*n)) / denominator
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denominator
    return [max(0.0, center-half), min(1.0, center+half)]


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def load_runtime(source):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this declared experiment.')
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(
        source, local_files_only=True, dtype=torch.bfloat16, attn_implementation='sdpa')
    model.to('cuda')
    return torch, tokenizer, model


def prompt_text(tokenizer, item):
    return tokenizer.apply_chat_template(
        [{'role': 'system', 'content': item.get('system', SPEC)},
         {'role': 'user', 'content': item['prompt']}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)


def encode(tokenizer, item, maximum):
    prefix = tokenizer.encode(prompt_text(tokenizer, item), add_special_tokens=False)
    suffix = tokenizer.encode(compact(item['answer']) + tokenizer.eos_token, add_special_tokens=False)
    if len(prefix) + len(suffix) > maximum:
        raise RuntimeError(f"Example {item['id']} exceeds sequence cap; no silent truncation.")
    return {'ids': prefix + suffix, 'labels': [-100] * len(prefix) + suffix,
            'id': item['id'], 'tokens': len(prefix) + len(suffix)}


def choose_replay(tokenizer, items, config):
    rng = random.Random(config['seed'])
    items = list(items)
    rng.shuffle(items)
    chosen, tokens = [], 0
    # Stratify via cycling skills; shuffle within each skill first.
    groups = {}
    for item in items:
        groups.setdefault(item['skill'], []).append(item)
    while groups and len(chosen) < config['replay_examples']:
        for key in list(sorted(groups)):
            item = groups[key].pop()
            n = len(tokenizer.encode(compact(item), add_special_tokens=False))
            if tokens + n <= config['replay_tokens'] and len(chosen) < config['replay_examples']:
                chosen.append(item)
                tokens += n
            if not groups[key]:
                del groups[key]
    serialized = compact(chosen).encode()
    # Include JSON list delimiters/commas in the token ceiling too.
    while chosen and len(tokenizer.encode(compact(chosen), add_special_tokens=False)) > config['replay_tokens']:
        chosen.pop()
    serialized = compact(chosen).encode()
    return chosen, {'examples': len(chosen),
                    'tokens': len(tokenizer.encode(serialized.decode(), add_special_tokens=False)),
                    'bytes': len(serialized)}


def train(args, config):
    from peft import LoraConfig, get_peft_model
    torch, tokenizer, base = load_runtime(args.source)
    torch.manual_seed(config['seed'])
    random.seed(config['seed'])
    base_params = sum(p.numel() for p in base.parameters())
    model = get_peft_model(base, LoraConfig(
        r=config['lora_rank'], lora_alpha=config['lora_rank'] * 2,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        lora_dropout=0.0, task_type='CAUSAL_LM'))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config['learning_rate'])
    payload = json.loads(Path(args.data).read_text())
    current = payload['current']
    # The worker receives only the current stage and bounded previous memory.
    replay = payload.get('replay', [])
    replay_stats = payload.get('replay_stats', {'examples': 0, 'tokens': 0, 'bytes': 0})
    if replay_stats['tokens'] > config['replay_tokens'] or len(replay) > config['replay_examples']:
        raise RuntimeError('Replay exceeds declared budget.')
    current_encoded = [encode(tokenizer, x, config['max_sequence_tokens']) for x in current]
    replay_encoded = [encode(tokenizer, x, config['max_sequence_tokens']) for x in replay]
    rng = random.Random(config['seed'])
    budget = config['stage_training_tokens']
    tokens = answer_tokens = updates = micro = 0
    seen_ids = set()
    sampled_skills = {}
    skill_by_id = {x['id']: x['skill'] for x in current + replay}
    losses = []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    model.train()
    optimizer.zero_grad()
    checkpoint_half = False
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    while True:
        batch = []
        # Fixed expected 50:50 current/replay ratio when replay exists.
        for _ in range(config['batch_size']):
            pool = replay_encoded if replay_encoded and rng.random() < 0.5 else current_encoded
            item = rng.choice(pool)
            if tokens + sum(len(x['ids']) for x in batch) + len(item['ids']) <= budget:
                batch.append(item)
        if not batch:
            break
        width = max(len(x['ids']) for x in batch)
        ids = torch.tensor([x['ids'] + [tokenizer.pad_token_id] * (width - len(x['ids'])) for x in batch], device='cuda')
        labels = torch.tensor([x['labels'] + [-100] * (width - len(x['ids'])) for x in batch], device='cuda')
        mask = torch.tensor([[1] * len(x['ids']) + [0] * (width - len(x['ids'])) for x in batch], device='cuda')
        loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
        (loss / config['gradient_accumulation']).backward()
        losses.append(float(loss.detach()))
        tokens += sum(len(x['ids']) for x in batch)
        answer_tokens += sum(sum(v != -100 for v in x['labels']) for x in batch)
        for item in batch:
            seen_ids.add(item['id'])
            skill = skill_by_id[item['id']]
            sampled_skills[skill] = sampled_skills.get(skill, 0) + 1
        micro += 1
        if micro % config['gradient_accumulation'] == 0:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad()
            updates += 1
            if updates % 5 == 0:
                print(f'tokens={tokens}/{budget} loss={sum(losses[-8:])/len(losses[-8:]):.4f}', flush=True)
            if args.half and not checkpoint_half and tokens >= budget // 2:
                # Save a temporary adapter; export it separately after training.
                model.save_pretrained(out / 'half_adapter')
                write(out / 'half_budget.json', {'training_tokens': tokens, 'optimizer_updates': updates})
                checkpoint_half = True
    if micro % config['gradient_accumulation']:
        # Correct the last partial accumulation to keep average gradient scale.
        factor = config['gradient_accumulation'] / (micro % config['gradient_accumulation'])
        for p in trainable:
            if p.grad is not None:
                p.grad.mul_(factor)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        updates += 1
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated()
    merged = model.merge_and_unload()
    merged.config.use_cache = True
    merged.save_pretrained(out / 'checkpoint', safe_serialization=True)
    tokenizer.save_pretrained(out / 'checkpoint')
    stats = {'source': args.source, 'base_parameters': base_params,
             'deployed_parameters': sum(p.numel() for p in merged.parameters()),
             'trainable_parameters': sum(p.numel() for p in trainable),
             'training_tokens': tokens, 'supervised_answer_tokens': answer_tokens,
             'token_budget': budget, 'optimizer_updates': updates, 'microbatches': micro,
             'training_seconds': seconds, 'peak_allocated_bytes': peak,
             'replay': replay_stats, 'current_examples': len(current),
             'sampled_examples_by_skill': sampled_skills, 'unique_examples_seen': len(seen_ids),
             'training_data_sha256': hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
             'loss_first_8': sum(losses[:8]) / len(losses[:8]),
             'loss_last_8': sum(losses[-8:]) / len(losses[-8:]),
             'torch_version': torch.__version__, 'cuda_version': torch.version.cuda,
             'gpu': torch.cuda.get_device_name(), 'persistent_optimizer_state': False}
    stats['worker_source_sha256'] = WORKER_SOURCE_SHA256
    stats['task_source_sha256'] = TASK_SOURCE_SHA256
    write(out / 'training.json', stats)
    if stats['base_parameters'] != stats['deployed_parameters']:
        raise RuntimeError('Deployment size changed.')
    print(json.dumps(stats), flush=True)


def evaluate(args, config):
    torch, tokenizer, model = load_runtime(args.source)
    model.eval()
    data = json.loads(Path(args.data).read_text())
    results = []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for start in range(0, len(data), config['eval_batch_size']):
        batch = data[start:start + config['eval_batch_size']]
        texts = [prompt_text(tokenizer, x) for x in batch]
        inputs = tokenizer(texts, return_tensors='pt', padding=True, add_special_tokens=False).to('cuda')
        if inputs.input_ids.shape[1] > config['max_sequence_tokens']:
            raise RuntimeError('Evaluation prompt exceeds context budget.')
        torch.cuda.synchronize()
        t = time.perf_counter()
        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=config['max_new_tokens'], do_sample=False,
                                     use_cache=True, pad_token_id=tokenizer.pad_token_id)
        torch.cuda.synchronize()
        duration = time.perf_counter() - t
        generated = outputs[:, inputs.input_ids.shape[1]:]
        for item, ids in zip(batch, generated):
            response = tokenizer.decode(ids, skip_special_tokens=True)
            results.append({'id': item['id'], 'skill': item['skill'], 'answer': item['answer'],
                            'response': response, **score(response, item['answer']),
                            'hit_generation_cap': bool(len(ids) == config['max_new_tokens'] and
                                                       tokenizer.eos_token_id not in ids.tolist())})
        print(f'evaluated {min(start+len(batch),len(data))}/{len(data)} batch_seconds={duration:.2f}', flush=True)
    groups = {}
    for skill in sorted({x['skill'] for x in results}):
        items = [x for x in results if x['skill'] == skill]
        groups[skill] = {'n': len(items), **{key: sum(x[key] for x in items) / len(items)
                        for key in ['strict_correct', 'semantic_correct', 'valid_json']}}
        groups[skill]['strict_correct_wilson_95'] = wilson(sum(x['strict_correct'] for x in items), len(items))
    report = {'source': args.source, 'fresh_process_pid': os.getpid(),
              'data_sha256': hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
              'deployed_parameters': sum(p.numel() for p in model.parameters()),
              'evaluation_seconds': time.perf_counter()-started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'groups': groups, 'results': results}
    if any(x['skill'] == 'general' for x in data):
        choices = [tokenizer.encode(label, add_special_tokens=False) for label in 'ABCD']
        if any(len(ids) != 1 for ids in choices):
            raise RuntimeError('Multiple-choice probe requires single-token answer labels.')
        choice_ids = torch.tensor([ids[0] for ids in choices], device='cuda')
        mc_results = []
        probes = general_mc_examples()
        for start in range(0, len(probes), config['eval_batch_size']):
            items = probes[start:start+config['eval_batch_size']]
            texts = [prompt_text(tokenizer, {'system': 'Answer the multiple-choice question.',
                                            'prompt': x['question']}) for x in items]
            inputs = tokenizer(texts, return_tensors='pt', padding=True, add_special_tokens=False).to('cuda')
            with torch.inference_mode():
                logits = model(**inputs, logits_to_keep=1).logits[:, -1, choice_ids].float()
                probabilities = logits.softmax(dim=-1).cpu().tolist()
            for item, values in zip(items, probabilities):
                selected = 'ABCD'[max(range(4), key=lambda i: values[i])]
                mc_results.append({**item, 'selected': selected, 'correct': selected == item['answer'],
                                   'conditional_label_probabilities': dict(zip('ABCD', values))})
        successes = sum(x['correct'] for x in mc_results)
        report['general_mc'] = {'n': len(mc_results), 'accuracy': successes/len(mc_results),
                                 'wilson_95': wilson(successes, len(mc_results)),
                                 'scoring': 'Restricted next-token likelihood over A/B/C/D; no generated JSON requirement.',
                                 'results': mc_results}
    report['worker_source_sha256'] = WORKER_SOURCE_SHA256
    report['task_source_sha256'] = TASK_SOURCE_SHA256
    write(args.output, report)
    print(json.dumps(groups), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['train', 'eval', 'merge-half'])
    p.add_argument('--config', required=True)
    p.add_argument('--source', required=True)
    p.add_argument('--data')
    p.add_argument('--output', required=True)
    p.add_argument('--half', action='store_true')
    p.add_argument('--adapter')
    a = p.parse_args()
    c = json.loads(Path(a.config).read_text())
    if a.action == 'train':
        train(a, c)
    elif a.action == 'eval':
        evaluate(a, c)
    else:
        from peft import PeftModel
        _, tokenizer, base = load_runtime(a.source)
        model = PeftModel.from_pretrained(base, a.adapter).merge_and_unload()
        model.save_pretrained(a.output, safe_serialization=True)
        tokenizer.save_pretrained(a.output)


if __name__ == '__main__':
    main()
