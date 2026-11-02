"""Bounded teacher-distribution caches and separate compact consolidation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time

from continual.tasks import compact
from continual.worker import encode, load_runtime, write

SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def grouped_kl(student_logits, teacher, temperature):
    """KL on teacher top-k vocabulary bins plus the complementary bin.

    This is a coarsened KL, not full-vocabulary KL. Mean over answer positions.
    """
    import torch
    logp = (student_logits.float() / temperature).log_softmax(-1)
    indices = teacher['indices'].to(logp.device).long()
    teacher_logp = teacher['log_probs'].to(logp.device).float()
    teacher_tail = teacher['tail'].to(logp.device).float().clamp_min(1e-8)
    selected = logp.gather(-1, indices)
    student_tail = (1 - selected.exp().sum(-1)).clamp_min(1e-8)
    top = (teacher_logp.exp() * (teacher_logp - selected)).sum(-1)
    tail = teacher_tail * (teacher_tail.log() - student_tail.log())
    return (top + tail).mean() * temperature**2


def cache(args, config):
    import torch
    torch, tokenizer, model = load_runtime(args.source)
    model.eval()
    data = json.loads(Path(args.data).read_text())
    examples = data['examples']
    cached, n_tokens, enabled = {}, 0, 0
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for item in examples:
        enc = encode(tokenizer, item, config['max_sequence_tokens'])
        ids = torch.tensor([enc['ids']], device='cuda')
        labels = torch.tensor(enc['labels'], device='cuda')
        positions = (labels[1:] != -100).nonzero().flatten()
        with torch.inference_mode():
            output = model(input_ids=ids, use_cache=False).logits[0]
            selected = output[positions].float()
            correct = bool((selected.argmax(-1) == labels[positions + 1]).all())
            logs = (selected / config['distillation_temperature']).log_softmax(-1)
            values, indices = logs.topk(config['teacher_top_k'], -1)
            tail = (1 - values.exp().sum(-1)).clamp_min(1e-8)
        eligible = correct if data['role'] == 'new' else True
        enabled += int(eligible)
        cached[item['id']] = {'positions': positions.cpu(), 'indices': indices.to(torch.int32).cpu(),
                              'log_probs': values.cpu(), 'tail': tail.cpu(), 'enabled': eligible,
                              'encoding_sha256': hashlib.sha256(compact(enc['ids']).encode()).hexdigest()}
        n_tokens += len(enc['ids'])
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    torch.save(cached, tmp)
    if tmp.stat().st_size > config['teacher_cache_bytes']:
        tmp.unlink()
        raise RuntimeError('Temporary teacher cache exceeded byte ceiling.')
    tmp.replace(path)
    write(path.with_suffix('.json'), {'role': data['role'], 'examples': len(examples),
          'enabled_examples': enabled, 'teacher_forward_tokens': n_tokens,
          'cache_bytes': path.stat().st_size, 'seconds': time.perf_counter()-started,
          'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
          'source': args.source, 'source_sha256': SOURCE_SHA256,
          'data_sha256': hashlib.sha256(Path(args.data).read_bytes()).hexdigest()})
    print('cache', data['role'], len(examples), 'eligible', enabled, flush=True)


def train(args, config):
    import torch
    from peft import LoraConfig, get_peft_model
    torch, tokenizer, base = load_runtime(args.source)
    torch.manual_seed(config['seed'])
    base_count = sum(p.numel() for p in base.parameters())
    model = get_peft_model(base, LoraConfig(r=config['lora_rank'], lora_alpha=2*config['lora_rank'],
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        lora_dropout=0.0, task_type='CAUSAL_LM'))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=config['learning_rate'])
    payload = json.loads(Path(args.data).read_text())
    memory = payload.get('memory', [])
    memory_serialized = compact(memory)
    if len(memory) > config['replay_examples'] or len(tokenizer.encode(memory_serialized, add_special_tokens=False)) > config['replay_tokens']:
        raise RuntimeError('Memory budget exceeded.')
    old_cache = torch.load(args.old_cache, map_location='cpu', weights_only=True) if args.old_cache else {}
    new_cache = torch.load(args.new_cache, map_location='cpu', weights_only=True) if args.new_cache else {}
    total_cache_bytes = sum(Path(p).stat().st_size for p in [args.old_cache, args.new_cache] if p)
    if total_cache_bytes > config['teacher_cache_bytes']:
        raise RuntimeError('Combined teacher cache exceeds ceiling.')
    pools = {}
    for key, examples in [('new', payload['current']), ('old', memory), ('composition', payload.get('compositions', []))]:
        if examples:
            pools[key] = [encode(tokenizer, item, config['max_sequence_tokens']) for item in examples]
    probabilities = {'new': 0.5, 'old': 0.5, 'composition': 0.0}
    if 'composition' in pools:
        probabilities = {'new': 0.5, 'old': 0.25, 'composition': 0.25}
    active = [k for k in pools if probabilities[k] > 0]
    weights = [probabilities[k] for k in active]
    rng = random.Random(config['seed'])
    tokens = answer_tokens = updates = micro = 0
    seen = set()
    counts = {}
    losses, kls = [], []
    budget = config['consolidation_tokens']
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model.train()
    optimizer.zero_grad()
    while True:
        batch = []
        for _ in range(config['batch_size']):
            role = rng.choices(active, weights=weights)[0]
            item = rng.choice(pools[role])
            if tokens + sum(len(x[1]['ids']) for x in batch) + len(item['ids']) <= budget:
                batch.append((role, item))
        if not batch:
            break
        width = max(len(x['ids']) for _, x in batch)
        ids = torch.tensor([x['ids']+[tokenizer.pad_token_id]*(width-len(x['ids'])) for _, x in batch], device='cuda')
        labels = torch.tensor([x['labels']+[-100]*(width-len(x['ids'])) for _, x in batch], device='cuda')
        mask = torch.tensor([[1]*len(x['ids'])+[0]*(width-len(x['ids'])) for _, x in batch], device='cuda')
        outputs = model(input_ids=ids, attention_mask=mask, labels=labels)
        extra = outputs.loss * 0
        enabled = 0
        for i, (role, enc) in enumerate(batch):
            teacher = (new_cache if role == 'new' else old_cache if role == 'old' else {}).get(enc['id'])
            if teacher and teacher['enabled']:
                if teacher['encoding_sha256'] != hashlib.sha256(compact(enc['ids']).encode()).hexdigest():
                    raise RuntimeError('Teacher/student tokenization mismatch.')
                positions = teacher['positions'].to('cuda').long()
                weight = config['new_teacher_weight'] if role == 'new' else config['old_teacher_weight']
                extra = extra + weight * grouped_kl(outputs.logits[i, positions], teacher, config['distillation_temperature'])
                enabled += 1
        # Average over all batch members, including examples without a KL target.
        extra = extra / len(batch)
        loss = outputs.loss + extra
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite consolidation loss; refusing checkpoint export.')
        (loss/config['gradient_accumulation']).backward()
        losses.append(float(outputs.loss.detach()))
        kls.append(float(extra.detach()))
        tokens += sum(len(x['ids']) for _, x in batch)
        answer_tokens += sum(sum(v != -100 for v in x['labels']) for _, x in batch)
        for role, enc in batch:
            seen.add(enc['id'])
            counts[role] = counts.get(role, 0)+1
        micro += 1
        if micro % config['gradient_accumulation'] == 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            optimizer.zero_grad()
            updates += 1
            if updates % 5 == 0:
                print('tokens', tokens, '/', budget, 'CE', sum(losses[-8:])/len(losses[-8:]),
                      'KL', sum(kls[-8:])/len(kls[-8:]), flush=True)
    if micro % config['gradient_accumulation']:
        scale = config['gradient_accumulation']/(micro % config['gradient_accumulation'])
        for p in params:
            if p.grad is not None:
                p.grad.mul_(scale)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        updates += 1
    torch.cuda.synchronize()
    seconds = time.perf_counter()-started
    peak = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.max_memory_reserved()
    merged = model.merge_and_unload()
    merged.config.use_cache = True
    out = Path(args.output)
    merged.save_pretrained(out/'checkpoint', safe_serialization=True)
    tokenizer.save_pretrained(out/'checkpoint')
    deployed = sum(p.numel() for p in merged.parameters())
    if base_count != deployed:
        raise RuntimeError('Compact parameter count changed.')
    stats = {'base_parameters': base_count, 'deployed_parameters': deployed,
             'trainable_parameters': sum(p.numel() for p in params), 'training_tokens': tokens,
             'supervised_answer_tokens': answer_tokens, 'optimizer_updates': updates,
             'training_seconds': seconds, 'peak_allocated_bytes': peak, 'peak_reserved_bytes': reserved,
             'replay_examples': len(memory), 'replay_tokens': len(tokenizer.encode(memory_serialized, add_special_tokens=False)),
             'replay_bytes': len(memory_serialized.encode()), 'temporary_cache_bytes': total_cache_bytes,
             'sampled_examples_by_role': counts, 'unique_examples_seen': len(seen),
             'loss_last_8': sum(losses[-8:])/len(losses[-8:]), 'weighted_kl_last_8': sum(kls[-8:])/len(kls[-8:]),
             'source': args.source, 'source_sha256': SOURCE_SHA256,
             'payload_sha256': hashlib.sha256(Path(args.data).read_bytes()).hexdigest()}
    write(out/'training.json', stats)
    print(json.dumps(stats), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['cache', 'train'])
    for name in ['config', 'source', 'data', 'output']:
        p.add_argument('--'+name, required=True)
    p.add_argument('--old-cache')
    p.add_argument('--new-cache')
    a = p.parse_args()
    config = json.loads(Path(a.config).read_text())
    (cache if a.action == 'cache' else train)(a, config)


if __name__ == '__main__':
    main()
