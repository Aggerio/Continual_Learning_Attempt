"""Bounded activation patching and discovery-selected module reversion.

Cross-checkpoint interventions identify causal contributions, not unique fact storage.
Teacher-forced discovery and autonomous validation are explicitly separate.
"""
import argparse
from contextlib import contextmanager
import gc
import json
from pathlib import Path
import random
import subprocess
import sys
import time

import torch
import torch.nn.functional as F

from continual.tasks import score
from continual.worker import encode, load_runtime, prompt_text, write


def read(path):
    return json.loads(Path(path).read_text())


def tensor_output(output):
    return output[0] if isinstance(output, tuple) else output


def replace_output(output, value):
    original = tensor_output(output)
    if original.shape != value.shape:
        raise ValueError('Patch shape mismatch')
    return (value,) + output[1:] if isinstance(output, tuple) else value


def module(model, site):
    layer, component = site.split(':')
    block = model.model.layers[int(layer)]
    return block if component == 'residual' else getattr(block, component)


@contextmanager
def patch(model, site, donor=None, zero=False, shuffle=False):
    def hook(_module, _inputs, output):
        target = tensor_output(output)
        value = torch.zeros_like(target) if zero else donor.to(target.device)
        if shuffle:
            value = value.roll(1, 0)
        return replace_output(output, value)
    handle = module(model, site).register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def reverted(destination, source, sites):
    if any(site.endswith(':residual') for site in sites):
        raise ValueError('Residual states are not editable weight modules')
    backups = {s: {k: v.detach().clone() for k, v in module(destination, s).state_dict().items()}
               for s in sites}
    try:
        for site in sites:
            module(destination, site).load_state_dict(module(source, site).state_dict())
        yield
    finally:
        for site in sites:
            module(destination, site).load_state_dict(backups[site])


def answer_logp(logits, labels):
    target = labels[:, 1:]
    valid = target != -100
    rows = valid.nonzero()[:, 0]
    # Convert only scored positions to fp32, not the full vocabulary tensor.
    values = -F.cross_entropy(logits[:, :-1][valid].float(), target[valid], reduction='none')
    total = torch.zeros(labels.shape[0], device=logits.device).scatter_add_(0, rows, values)
    return total / valid.sum(1).clamp_min(1)


def batch_tensors(tokenizer, items):
    xs = [encode(tokenizer, x, 512) for x in items]
    width = max(len(x['ids']) for x in xs)
    device = 'cuda'
    ids = torch.tensor([x['ids'] + [tokenizer.pad_token_id]*(width-len(x['ids'])) for x in xs], device=device)
    labels = torch.tensor([x['labels'] + [-100]*(width-len(x['ids'])) for x in xs], device=device)
    lengths = torch.tensor([len(x['ids']) for x in xs], device=device)
    mask = (torch.arange(width, device=device)[None] < lengths[:, None]).long()
    return {'input_ids': ids, 'attention_mask': mask}, labels


def capture(model, inputs, sites):
    cache = {}
    handles = []
    for site in sites:
        def hook(_module, _inputs, output, key=site):
            cache[key] = tensor_output(output).detach().cpu()
        handles.append(module(model, site).register_forward_hook(hook))
    try:
        output = model(**inputs, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return output, cache


def bootstrap(values, seed=1701):
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(values, k=len(values)))/len(values) for _ in range(1000))
    return {'mean': sum(values)/len(values), 'query_bootstrap_95': [means[24], means[974]]}


@torch.inference_mode()
def sweep(before, after, tokenizer, items, cfg, root):
    sites = [f'{i}:{kind}' for i in range(len(before.model.layers)) for kind in cfg['components']]
    weight_sites = [s for s in sites if not s.endswith(':residual')]
    result = {'ids': [x['id'] for x in items], 'before': [], 'after': [],
              'old_into_new': {s: [] for s in sites}, 'new_into_old': {s: [] for s in sites},
              'weight_reversion': {s: [] for s in weight_sites}, 'peak_activation_cache_bytes': 0,
              'self_patch_max_error': 0.0}
    for start in range(0, len(items), cfg['trace_batch_size']):
        inputs, labels = batch_tensors(tokenizer, items[start:start+cfg['trace_batch_size']])
        old_out, old_cache = capture(before, inputs, sites)
        old = answer_logp(old_out.logits, labels); del old_out
        new_out, new_cache = capture(after, inputs, sites)
        new = answer_logp(new_out.logits, labels); del new_out
        size = sum(v.numel()*v.element_size() for cache in [old_cache, new_cache] for v in cache.values())
        if size > cfg['activation_cache_bytes']:
            raise RuntimeError('Activation cache ceiling exceeded')
        result['peak_activation_cache_bytes'] = max(size, result['peak_activation_cache_bytes'])
        result['before'].extend(old.tolist()); result['after'].extend(new.tolist())
        for site in sites:
            with patch(after, site, old_cache[site]):
                values = answer_logp(after(**inputs, use_cache=False).logits, labels)
                result['old_into_new'][site].extend((values-new).tolist())
            with patch(before, site, new_cache[site]):
                values = answer_logp(before(**inputs, use_cache=False).logits, labels)
                result['new_into_old'][site].extend((values-old).tolist())
        for site in weight_sites:
            with reverted(after, before, [site]):
                values = answer_logp(after(**inputs, use_cache=False).logits, labels)
                result['weight_reversion'][site].extend((values-new).tolist())
        with patch(after, sites[0], new_cache[sites[0]]):
            error = (answer_logp(after(**inputs, use_cache=False).logits, labels)-new).abs().max().item()
            result['self_patch_max_error'] = max(result['self_patch_max_error'], error)
        write(root/'trace-partial.json', result)
        print('Discovery sweep', start+len(old), '/', len(items), flush=True)
    for family in ['old_into_new', 'new_into_old', 'weight_reversion']:
        result[family+'_summary'] = {s: bootstrap(v) for s, v in result[family].items()}
    write(root/'trace.json', result)
    return result


def select(result, cfg):
    ranked = sorted(result['weight_reversion'], key=lambda s: -sum(result['weight_reversion'][s])/len(result['ids']))
    activation = max(ranked, key=lambda s: sum(result['old_into_new'][s]))
    rng = random.Random(cfg['random_control_seed'])
    variants = {}
    for n in [1, cfg['top_weight_modules']]:
        chosen = ranked[:n]
        used = set(chosen)
        controls = []
        for site in chosen:
            pool = [s for s in ranked if s.split(':')[1] == site.split(':')[1] and s not in used]
            control = rng.choice(pool); used.add(control); controls.append(control)
        variants['top'+str(n)] = chosen
        variants['random'+str(n)] = controls
    alternatives = [s for s in ranked if s != activation and s.split(':')[1] == activation.split(':')[1]]
    return {'weight_ranking': ranked, 'weight_variants': variants, 'activation_site': activation,
            'random_activation_site': rng.choice(alternatives), 'selection_data': '64 discovery C only',
            'interpretation': 'Descending mean gold-token log probability change; positive values are helpful. Residual patches excluded from selection.'}


@torch.inference_mode()
def greedy(model, tokenizer, items, maximum=192, donor=None, site=None, shuffle=False, zero=False):
    inputs = tokenizer([prompt_text(tokenizer, x) for x in items], padding=True,
                       add_special_tokens=False, return_tensors='pt').to(model.device)
    ids, mask = inputs['input_ids'], inputs['attention_mask']
    past = donor_past = None
    finished = torch.zeros(len(items), dtype=torch.bool, device=model.device)
    sequences = [[] for _ in items]
    for _ in range(maximum):
        positions = (mask.cumsum(-1)-1).masked_fill(mask == 0, 0)[:, -ids.shape[1]:]
        kwargs = {'input_ids': ids, 'attention_mask': mask, 'position_ids': positions,
                  'use_cache': True, 'logits_to_keep': 1}
        if donor is not None:
            cache = {}
            def hook(_module, _inputs, output):
                cache['value'] = tensor_output(output).detach()
            handle = module(donor, site).register_forward_hook(hook)
            try:
                output = donor(**kwargs, past_key_values=donor_past)
                donor_past = output.past_key_values
                del output
            finally:
                handle.remove()
            with patch(model, site, cache['value'], shuffle=shuffle):
                output = model(**kwargs, past_key_values=past)
        elif zero:
            with patch(model, site, zero=True):
                output = model(**kwargs, past_key_values=past)
        else:
            output = model(**kwargs, past_key_values=past)
        past = output.past_key_values
        tokens = output.logits[:, -1].argmax(-1)
        tokens = tokens.masked_fill(finished, tokenizer.pad_token_id)
        active = ~finished
        for i, token in enumerate(tokens.tolist()):
            if active[i]:
                sequences[i].append(token)
        finished |= tokens == tokenizer.eos_token_id
        if finished.all():
            break
        mask = torch.cat([mask, active.long()[:, None]], dim=1)
        ids = tokens[:, None]
    responses = tokenizer.batch_decode(sequences, skip_special_tokens=True)
    return [{**x, 'response': text, **score(text, x['answer']),
             'generated_tokens': len(tokens), 'hit_generation_cap': len(tokens) == maximum and tokenizer.eos_token_id not in tokens}
            for x, text, tokens in zip(items, responses, sequences)]


def generation_controls(before, after, tokenizer, items, selection, cfg, root):
    site = selection['activation_site']
    conditions = {'before': (before, {}), 'after': (after, {}),
        'old_patch': (after, {'donor': before, 'site': site}),
        'self_patch': (after, {'donor': after, 'site': site}),
        'random_patch': (after, {'donor': before, 'site': selection['random_activation_site']}),
        'shuffled_patch': (after, {'donor': before, 'site': site, 'shuffle': True}),
        'zero_ablation': (after, {'site': site, 'zero': True}),
        'output_copy_control': (after, {'donor': before, 'site': f'{len(before.model.layers)-1}:residual'})}
    reports = {}
    for name, (target, kwargs) in conditions.items():
        rows = []
        for start in range(0, len(items), cfg['trace_batch_size']):
            rows += greedy(target, tokenizer, items[start:start+cfg['trace_batch_size']], **kwargs)
        reports[name] = rows
        write(root/'generation'/f'{name}.json', rows)
        print('Autonomous generation', name, flush=True)
    controls = {'self_patch_same_responses': all(a['response'] == b['response'] for a,b in zip(reports['after'], reports['self_patch'])),
                'output_copy_same_responses': all(a['response'] == b['response'] for a,b in zip(reports['before'], reports['output_copy_control']))}
    write(root/'generation/controls.json', controls)
    if not all(controls.values()):
        raise RuntimeError('Generation implementation control failed: '+str(controls))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args(); root = args.root.resolve(); cfg = read(root/'config.json')
    started = time.perf_counter()
    torch.manual_seed(cfg['seed'])
    torch.cuda.reset_peak_memory_stats()
    _, tokenizer, before = load_runtime(cfg['before_checkpoint']); before.eval()
    _, _, after = load_runtime(root/'after/checkpoint'); after.eval()
    result = sweep(before, after, tokenizer, read(root/'data/discovery.json'), cfg, root)
    selection = select(result, cfg); write(root/'selection.json', selection)
    # Freeze site selection before any intervention on validation examples.
    for name, sites in selection['weight_variants'].items():
        with reverted(after, before, sites):
            target = root/'variants'/name/'checkpoint'; target.mkdir(parents=True, exist_ok=True)
            after.save_pretrained(target); tokenizer.save_pretrained(target)
    generation_controls(before, after, tokenizer, read(root/'data/activation-generation.json'), selection, cfg, root)
    resource = {'trace_wall_seconds': time.perf_counter()-started,
                'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
                'before_parameters': sum(p.numel() for p in before.parameters()),
                'after_parameters': sum(p.numel() for p in after.parameters()),
                'peak_activation_cache_bytes': result['peak_activation_cache_bytes'],
                'cuda_device': torch.cuda.get_device_name()}
    del before, after; gc.collect(); torch.cuda.empty_cache()
    for name in selection['weight_variants']:
        base = root/'variants'/name
        cmd = [sys.executable, '-m', 'continual.worker', 'eval', '--source', str(base/'checkpoint'),
               '--data', str(root/'data/validation.json'), '--output', str(base/'validation.json'),
               '--config', str(root/'student-config.json')]
        with (base/'eval.log').open('w') as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
        print('Variant evaluated', name, flush=True)
    resource['total_wall_seconds'] = time.perf_counter()-started
    write(root/'resources.json', resource)


if __name__ == '__main__':
    main()
