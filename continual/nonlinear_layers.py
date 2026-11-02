"""Temporary nonlinear layers, bounded teacher targets, and compact distillation.

Expanded checkpoints contain extra weights plus an immutable backbone reference.
Compact exports contain neither hooks, extra layers, nor training adapters.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import random
import time
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F

from continual.assimilation import grouped_kl
from continual.tasks import compact
from continual.worker import encode, load_runtime, write


def read(path):
    return json.loads(Path(path).read_text())


class TemporaryResidual(nn.Module):
    """FP32 trainable nonlinear branch; preserve the backbone activation dtype."""
    def __init__(self, hidden, bottleneck):
        super().__init__()
        self.norm = nn.RMSNorm(hidden, eps=1e-6)
        self.up = nn.Linear(hidden, bottleneck, bias=False)
        self.down = nn.Linear(bottleneck, hidden, bias=False)
        nn.init.zeros_(self.down.weight)

    def forward(self, hidden):
        branch = self.down(F.silu(self.up(self.norm(hidden.float()))))
        return hidden + branch.to(hidden.dtype)


def attach(model, layers, bottleneck):
    if hasattr(model, 'temporary_layers'):
        raise ValueError('Temporary layers already attached')
    if len(set(layers)) != len(layers) or any(i < 0 or i >= len(model.model.layers) for i in layers):
        raise ValueError('Invalid insertion layers')
    model.requires_grad_(False)
    extra = nn.ModuleDict({str(i): TemporaryResidual(model.config.hidden_size, bottleneck) for i in layers})
    extra.to(device=model.device, dtype=torch.float32)
    model.add_module('temporary_layers', extra)
    model.temporary_bypass = set()
    handles = []
    for i in layers:
        def hook(_module, _inputs, output, index=i):
            if index in model.temporary_bypass:
                return output
            if not isinstance(output, torch.Tensor):
                raise TypeError('Expected current Qwen decoder tensor output')
            return extra[str(index)](output)
        handles.append(model.model.layers[i].register_forward_hook(hook))
    model.temporary_handles = handles
    return extra


@contextmanager
def capture_states(model, layers, detach=False):
    states, handles = {}, []
    for i in layers:
        def hook(_module, _inputs, output, index=i):
            states[index] = output.detach() if detach else output
        handles.append(model.model.layers[i].register_forward_hook(hook))
    try:
        yield states
    finally:
        for handle in handles:
            handle.remove()


def normalized_state_loss(student, teacher):
    """Mean squared distance between unit vectors, summed over hidden width."""
    return (F.normalize(student.float(), dim=-1) - F.normalize(teacher.float(), dim=-1)).square().sum(-1).mean()


def backbone_digest(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if name.startswith('temporary_layers.'):
            continue
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def load_expanded(source, config, teacher=None):
    _, tokenizer, model = load_runtime(source)
    attach(model, config['insertion_layers'], config['bottleneck'])
    if teacher:
        from safetensors.torch import load_file
        model.temporary_layers.load_state_dict(load_file(str(Path(teacher)/'layers.safetensors')))
    return tokenizer, model


def make_schedule(tokenizer, payload, config):
    pools = {role: [encode(tokenizer, x, config['max_sequence_tokens']) for x in items]
             for role, items in [('new', payload['current']), ('old', payload['memory'])]}
    rng = random.Random(config['seed']); batches = []; tokens = 0
    while True:
        batch = []
        for _ in range(config['batch_size']):
            role = rng.choices(['new', 'old'], weights=[0.5, 0.5])[0]
            enc = rng.choice(pools[role])
            if tokens + sum(x['tokens'] for x in batch) + enc['tokens'] <= config['training_tokens']:
                batch.append({'role': role, 'id': enc['id'], 'tokens': enc['tokens']})
        if not batch:
            break
        tokens += sum(x['tokens'] for x in batch); batches.append(batch)
    return {'batches': batches, 'tokens': tokens,
            'encoding_sha256': {x['id']: hashlib.sha256(compact(x['ids']).encode()).hexdigest()
                                for pool in pools.values() for x in pool}}


def cache_targets(args, config):
    if args.role == 'new':
        tokenizer, model = load_expanded(args.source, config, args.teacher)
    else:
        _, tokenizer, model = load_runtime(args.source)
    model.eval(); layers = sorted(set(config['guided_layers']+config['random_layers']))
    data = read(args.data)[args.role]; targets = {}; tokens = enabled = 0
    started = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    for item in data:
        enc = encode(tokenizer, item, config['max_sequence_tokens'])
        ids = torch.tensor([enc['ids']], device='cuda')
        labels = torch.tensor(enc['labels'], device='cuda')
        positions = (labels[1:] != -100).nonzero().flatten()
        with torch.inference_mode(), capture_states(model, layers, detach=True) as states:
            output = model(input_ids=ids, use_cache=False)
            logits = output.logits[0, positions].float()
            correct = bool((logits.argmax(-1) == labels[positions+1]).all())
            values, indices = (logits/config['distillation_temperature']).log_softmax(-1).topk(config['teacher_top_k'], -1)
            target = {'positions': positions.cpu(), 'indices': indices.to(torch.int32).cpu(),
                      'log_probs': values.cpu(), 'tail': (1-values.exp().sum(-1)).clamp_min(1e-8).cpu(),
                      'enabled': correct if args.role == 'new' else True,
                      'encoding_sha256': hashlib.sha256(compact(enc['ids']).encode()).hexdigest(),
                      'states': {str(i): states[i][0, positions].to(torch.bfloat16).cpu() for i in layers}}
        targets[item['id']] = target; enabled += int(target['enabled']); tokens += enc['tokens']
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(targets, path)
    if path.stat().st_size > config['teacher_cache_bytes']:
        path.unlink(); raise RuntimeError('Teacher cache ceiling exceeded')
    write(path.with_suffix('.json'), {'role': args.role, 'examples': len(data), 'enabled_examples': enabled,
          'hint_layers': layers, 'hint_positions': 'answer-prediction positions, matching shifted KL positions',
          'teacher_forward_tokens': tokens, 'cache_bytes': path.stat().st_size,
          'seconds': time.perf_counter()-started, 'peak_cuda_bytes': torch.cuda.max_memory_allocated()})


def train(args, config):
    from safetensors.torch import save_file
    torch.manual_seed(config['seed'])
    if args.arm == 'teacher':
        tokenizer, model = load_expanded(args.source, config)
        base_count = sum(p.numel() for name,p in model.named_parameters() if not name.startswith('temporary_layers.'))
        old_digest = backbone_digest(model)
        lr = config['teacher_learning_rate']; student_base = model
    else:
        from peft import LoraConfig, get_peft_model
        _, tokenizer, base = load_runtime(args.source)
        base_count = sum(p.numel() for p in base.parameters())
        model = get_peft_model(base, LoraConfig(r=config['lora_rank'], lora_alpha=2*config['lora_rank'],
            target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
            lora_dropout=0.0, task_type='CAUSAL_LM'))
        student_base = model.get_base_model(); lr = config['learning_rate']
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads(); model.config.use_cache = False
    params = [p for p in model.parameters() if p.requires_grad]
    if args.arm == 'teacher' and any(p.requires_grad for n,p in model.named_parameters() if not n.startswith('temporary_layers.')):
        raise RuntimeError('Backbone not frozen')
    optimizer = torch.optim.AdamW(params, lr=lr)
    payload = read(args.data); schedule = read(args.schedule)
    memory_tokens = len(tokenizer.encode(compact(payload['old']), add_special_tokens=False))
    if len(payload['old']) > config['replay_examples'] or memory_tokens > config['replay_tokens']:
        raise RuntimeError('Replay exceeds fixed bound')
    encodings = {x['id']: encode(tokenizer, x, config['max_sequence_tokens']) for xs in payload.values() for x in xs}
    if any(hashlib.sha256(compact(x['ids']).encode()).hexdigest()!=schedule['encoding_sha256'][key] for key,x in encodings.items()):
        raise RuntimeError('Frozen schedule encoding mismatch')
    caches = {}
    cache_bytes = 0
    for role,path in [('old',args.old_cache),('new',args.new_cache)]:
        if path:
            caches[role] = torch.load(path,map_location='cpu',weights_only=True); cache_bytes += Path(path).stat().st_size
    if cache_bytes > config['teacher_cache_bytes']:
        raise RuntimeError('Combined teacher cache ceiling exceeded')
    layers = config['guided_layers'] if args.arm == 'guided' else config['random_layers'] if args.arm == 'random' else []
    started = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    model.train(); optimizer.zero_grad(); tokens = updates = 0; losses = []; kls = []; hints = []; roles = {}
    for micro,batch in enumerate(schedule['batches'],1):
        xs = [encodings[x['id']] for x in batch]; width = max(len(x['ids']) for x in xs)
        ids = torch.tensor([x['ids']+[tokenizer.pad_token_id]*(width-len(x['ids'])) for x in xs],device='cuda')
        labels = torch.tensor([x['labels']+[-100]*(width-len(x['ids'])) for x in xs],device='cuda')
        mask = torch.tensor([[1]*len(x['ids'])+[0]*(width-len(x['ids'])) for x in xs],device='cuda')
        with capture_states(student_base,layers) as states:
            output = model(input_ids=ids,attention_mask=mask,labels=labels)
        kl = output.loss*0; hint = output.loss*0
        for i,(sample,enc) in enumerate(zip(batch,xs)):
            role = sample['role']; roles[role] = roles.get(role,0)+1
            target = caches.get(role,{}).get(enc['id'])
            if target and target['enabled']:
                if target['encoding_sha256'] != schedule['encoding_sha256'][enc['id']]:
                    raise RuntimeError('Cached target encoding mismatch')
                positions = target['positions'].to('cuda').long()
                weight = config['old_teacher_weight'] if role == 'old' else config['new_teacher_weight']
                kl = kl+weight*grouped_kl(output.logits[i,positions],target,config['distillation_temperature'])
                if layers:
                    hint = hint+sum(normalized_state_loss(states[layer][i,positions],
                        target['states'][str(layer)].to('cuda')) for layer in layers)/len(layers)
        kl = kl/len(batch); hint = hint/len(batch)*config['hidden_weight']
        loss = output.loss+kl+hint
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite training loss')
        (loss/config['gradient_accumulation']).backward()
        tokens += sum(x['tokens'] for x in xs)
        losses.append(float(output.loss.detach())); kls.append(float(kl.detach())); hints.append(float(hint.detach()))
        if micro%config['gradient_accumulation']==0:
            torch.nn.utils.clip_grad_norm_(params,1.0);optimizer.step();optimizer.zero_grad();updates += 1
            if updates%5==0:
                print('arm',args.arm,'tokens',tokens,'CE',sum(losses[-8:])/len(losses[-8:]),
                      'KL',sum(kls[-8:])/len(kls[-8:]),'hint',sum(hints[-8:])/len(hints[-8:]),flush=True)
    remainder = micro%config['gradient_accumulation']
    if remainder:
        for p in params:
            if p.grad is not None:p.grad.mul_(config['gradient_accumulation']/remainder)
        torch.nn.utils.clip_grad_norm_(params,1.0);optimizer.step();updates += 1
    torch.cuda.synchronize(); seconds = time.perf_counter()-started
    stats = {'arm':args.arm,'base_parameters':base_count,'trainable_parameters':sum(p.numel() for p in params),
             'training_tokens':tokens,'optimizer_updates':updates,'training_seconds':seconds,
             'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(),
             'replay_examples':len(payload['old']),'replay_tokens':memory_tokens,'teacher_cache_bytes':cache_bytes,
             'hint_layers':layers,'hidden_weight':config['hidden_weight'] if layers else 0.0,
             'sampled_roles':roles,'loss_last_8':sum(losses[-8:])/len(losses[-8:]),
             'kl_last_8':sum(kls[-8:])/len(kls[-8:]),'hint_last_8':sum(hints[-8:])/len(hints[-8:]),
             'schedule_sha256':hashlib.sha256(Path(args.schedule).read_bytes()).hexdigest(),
             'torch_version':torch.__version__,'cuda_version':torch.version.cuda,'gpu':torch.cuda.get_device_name()}
    out = Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if args.arm == 'teacher':
        final_digest = backbone_digest(model)
        if final_digest != old_digest:raise RuntimeError('Frozen backbone changed')
        stats.update({'backbone_before_sha256':old_digest,'backbone_after_sha256':final_digest,
                      'expanded_parameters':sum(p.numel() for p in model.parameters())})
        save_file({k:v.detach().cpu().contiguous() for k,v in model.temporary_layers.state_dict().items()},str(out/'layers.safetensors'))
    else:
        merged = model.merge_and_unload();merged.config.use_cache = True
        if hasattr(merged,'temporary_layers'):raise RuntimeError('Extra layers leaked into compact student')
        merged.save_pretrained(out/'checkpoint',safe_serialization=True);tokenizer.save_pretrained(out/'checkpoint')
        stats['deployed_parameters'] = sum(p.numel() for p in merged.parameters())
        if stats['deployed_parameters'] != base_count:raise RuntimeError('Compact parameter count changed')
    write(out/'training.json',stats)


def evaluate(args, config):
    from continual import worker
    tokenizer,model = load_expanded(args.source,config,args.teacher)
    model.temporary_bypass = set(int(x) for x in args.bypass.split(',') if x)
    if not model.temporary_bypass.issubset(config['insertion_layers']):raise ValueError('Unknown bypass layer')
    original = worker.load_runtime
    worker.load_runtime = lambda source:(torch,tokenizer,model)
    try:
        worker.evaluate(SimpleNamespace(source=args.source,data=args.data,output=args.output),config)
    finally:
        worker.load_runtime = original
    report = read(args.output)
    report.update({'temporary_layers':config['insertion_layers'],'bypassed_layers':sorted(model.temporary_bypass),
                   'active_extra_parameters':sum(p.numel() for k,m in model.temporary_layers.items() if int(k) not in model.temporary_bypass for p in m.parameters()),
                   'teacher_layers_sha256':hashlib.sha256((Path(args.teacher)/'layers.safetensors').read_bytes()).hexdigest(),
                   'evaluation_kind':'expanded_teacher'})
    write(args.output,report)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['cache','train','eval'])
    for name in ['config','source','data','output']:p.add_argument('--'+name,required=True)
    p.add_argument('--role',choices=['old','new']);p.add_argument('--teacher');p.add_argument('--schedule')
    p.add_argument('--arm',choices=['teacher','direct','output','guided','random']);p.add_argument('--old-cache');p.add_argument('--new-cache')
    p.add_argument('--bypass',default='')
    args=p.parse_args();config=read(args.config)
    {'cache':cache_targets,'train':train,'eval':evaluate}[args.action](args,config)


if __name__=='__main__':main()
