"""Numeric-position causal tracing and module-protected compact compression."""
import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import random
import re
import time

import torch
import torch.nn.functional as F

from continual.causal_diagnosis import capture,greedy,module,patch,reverted
from continual.tasks import compact
from continual.worker import encode,load_runtime,prompt_text,write


def read(p):return json.loads(Path(p).read_text())


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def numeric_positions(tokenizer,item):
    """Return causal prediction indices, not digit token indices or prompt digits."""
    text=compact(item['answer']);suffix=tokenizer(text+tokenizer.eos_token,add_special_tokens=False,return_offsets_mapping=True)
    prefix=len(tokenizer.encode(prompt_text(tokenizer,item),add_special_tokens=False))
    pattern=r'"total":(-?\d+)' if item['skill']=='D' else r':(-?\d+)'
    spans=[m.span(1) for m in re.finditer(pattern,text)]
    positions=[prefix+j-1 for j,(a,b) in enumerate(suffix['offset_mapping'])
               if a<b and any(a<end and b>start for start,end in spans)]
    if not positions:raise ValueError('No answer-number positions: '+item['id'])
    enc=encode(tokenizer,item,512)
    if enc['ids'][prefix:]!=suffix['input_ids']:raise RuntimeError('Tokenizer alignment mismatch')
    return positions


def tensors(tokenizer,items):
    xs=[encode(tokenizer,x,512) for x in items];width=max(x['tokens'] for x in xs)
    ids=torch.tensor([x['ids']+[tokenizer.pad_token_id]*(width-x['tokens']) for x in xs],device='cuda')
    mask=torch.tensor([[1]*x['tokens']+[0]*(width-x['tokens']) for x in xs],device='cuda')
    return {'input_ids':ids,'attention_mask':mask},[numeric_positions(tokenizer,x) for x in items]


def numeric_logp(logits,ids,positions):
    return torch.stack([-F.cross_entropy(logits[i,p].float(),ids[i,torch.tensor(p,device=ids.device)+1],reduction='mean')
                        for i,p in enumerate(positions)])


def ci(xs,seed=1701,samples=2000):
    rng=random.Random(seed);values=sorted(sum(rng.choices(xs,k=len(xs)))/len(xs) for _ in range(samples))
    return {'mean':sum(xs)/len(xs),'query_bootstrap_95':[values[int(.025*samples)],values[int(.975*samples)-1]],'n':len(xs)}


def specificity(values,items):
    d=[v for v,x in zip(values,items) if x['skill']=='D'];a=[v for v,x in zip(values,items) if x['skill']=='A']
    rng=random.Random(1701)
    samples=sorted(sum(rng.choices(d,k=len(d)))/len(d)-sum(rng.choices(a,k=len(a)))/len(a) for _ in range(2000))
    return {'mean':sum(d)/len(d)-sum(a)/len(a),'query_bootstrap_95':[samples[49],samples[1949]]}


def choose(trace,items,cfg):
    ranked=sorted(trace['weight_reversion'],key=lambda s:-specificity(trace['weight_reversion'][s],items)['mean'])
    chosen=ranked[:cfg['selected_modules']];rng=random.Random(cfg['random_seed']);randoms=[]
    for s in chosen:
        pool=[x for x in ranked if x.split(':')[1]==s.split(':')[1] and x not in chosen+randoms]
        randoms.append(rng.choice(pool))
    return {'guided':chosen,'random':randoms,'ranking':ranked,
            'discovery_numeric_specificity':{s:specificity(trace['weight_reversion'][s],items) for s in ranked},
            'scope':'Module-level update contribution on numeric answer positions, not a unique arithmetic circuit.'}


@torch.inference_mode()
def trace(before,after,tok,items,cfg,root):
    sites=[f'{i}:{kind}' for i in range(len(before.model.layers)) for kind in ['mlp','self_attn']]
    allsites=sites+[f'{len(before.model.layers)-1}:residual']
    result={'ids':[x['id'] for x in items],'before':[],'after':[],
            **{k:{s:[] for s in sites} for k in ['old_into_new','new_into_old','weight_reversion']},
            'self_patch_max_error':0.0,'output_copy_max_error':0.0,'peak_activation_cache_bytes':0}
    for start in range(0,len(items),cfg['trace_batch_size']):
        xs=items[start:start+cfg['trace_batch_size']];inputs,positions=tensors(tok,xs)
        out,old=capture(before,inputs,allsites);a=numeric_logp(out.logits,inputs['input_ids'],positions);del out
        out,new=capture(after,inputs,allsites);b=numeric_logp(out.logits,inputs['input_ids'],positions);del out
        size=sum(v.numel()*v.element_size() for c in [old,new] for v in c.values())
        if size>cfg['activation_cache_bytes']:raise RuntimeError('Activation cache bound exceeded')
        result['peak_activation_cache_bytes']=max(size,result['peak_activation_cache_bytes'])
        result['before']+=a.tolist();result['after']+=b.tolist()
        for s in sites:
            for target,donor,baseline,kind in [(after,old,b,'old_into_new'),(before,new,a,'new_into_old')]:
                with patch(target,s,donor[s]):
                    out=target(**inputs,use_cache=False);values=numeric_logp(out.logits,inputs['input_ids'],positions);del out
                result[kind][s]+=(values-baseline).tolist()
            with reverted(after,before,[s]):
                out=after(**inputs,use_cache=False);values=numeric_logp(out.logits,inputs['input_ids'],positions);del out
            result['weight_reversion'][s]+=(values-b).tolist()
        for s,donor,expected,key in [(sites[0],new,b,'self_patch_max_error'),(allsites[-1],old,a,'output_copy_max_error')]:
            with patch(after,s,donor[s]):
                out=after(**inputs,use_cache=False);v=numeric_logp(out.logits,inputs['input_ids'],positions);del out
            result[key]=max(result[key],float((v-expected).abs().max()))
        print('Trace',start+len(xs),'/',len(items),flush=True)
        write(root/'trace-partial.json',result)
    return result


@torch.inference_mode()
def validate(before,after,tok,items,selected,cfg,root):
    gs=selected['guided'][0];rs=selected['random'][0]
    result={'ids':[x['id'] for x in items],'before':[],'after':[],
            **{k:[] for k in ['guided_rollback','random_rollback','guided_patch','random_patch','shuffled_patch','reverse_patch','zero_ablation']}}
    for start in range(0,len(items),cfg['trace_batch_size']):
        xs=items[start:start+cfg['trace_batch_size']];inputs,p=tensors(tok,xs)
        out,old=capture(before,inputs,[gs,rs]);a=numeric_logp(out.logits,inputs['input_ids'],p);del out
        out,new=capture(after,inputs,[gs]);b=numeric_logp(out.logits,inputs['input_ids'],p);del out
        result['before']+=a.tolist();result['after']+=b.tolist()
        for name,sites in [('guided_rollback',selected['guided']),('random_rollback',selected['random'])]:
            with reverted(after,before,sites):
                out=after(**inputs,use_cache=False);v=numeric_logp(out.logits,inputs['input_ids'],p);del out
            result[name]+=(v-b).tolist()
        for name,target,s,donor,baseline,options in [
            ('guided_patch',after,gs,old[gs],b,{}),('random_patch',after,rs,old[rs],b,{}),
            ('shuffled_patch',after,gs,old[gs],b,{'shuffle':True}),('zero_ablation',after,gs,None,b,{'zero':True}),
            ('reverse_patch',before,gs,new[gs],a,{})]:
            with patch(target,s,donor,**options):
                out=target(**inputs,use_cache=False);v=numeric_logp(out.logits,inputs['input_ids'],p);del out
            result[name]+=(v-baseline).tolist()
    write(root/'validation-numeric.json',result)
    # Donor observes the recipient's generated prefixes; gold answers are never fed to these tests.
    cohorts={s:[x for x in items if x['skill']==s] for s in ['A','D','E']}
    patch_items=sum([cohorts[s][:n] for s,n in cfg['generation_patch_counts'].items()],[])
    control_items=cohorts['D'][:4]+cohorts['E'][:4]
    outputs={}
    def generate(name,target,data,**options):
        rows=[]
        for start in range(0,len(data),cfg['generation_batch_size']):
            rows+=greedy(target,tok,data[start:start+cfg['generation_batch_size']],**options)
        write(root/'generation'/f'{name}.json',rows);outputs[name]=rows;print('Generation',name,flush=True)
    for name,target in [('before',before),('after',after)]:generate(name,target,items)
    for name,sites in [('guided_rollback',selected['guided']),('random_rollback',selected['random'])]:
        with reverted(after,before,sites):generate(name,after,items)
    for name,s,options in [('guided_patch',gs,{}),('random_patch',rs,{}),('shuffled_patch',gs,{'shuffle':True}),('zero_ablation',gs,{'zero':True})]:
        generate(name,after,patch_items,**({'site':s,'zero':True} if options.get('zero') else {'donor':before,'site':s,**options}))
    generate('self_patch',after,control_items,donor=after,site=gs)
    generate('output_copy',after,control_items,donor=before,site=f'{len(before.model.layers)-1}:residual')
    for name,reference in [('self_patch','after'),('output_copy','before')]:
        index={x['id']:x['response'] for x in outputs[reference]}
        if any(x['response']!=index[x['id']] for x in outputs[name]):raise RuntimeError('Generation control failed: '+name)
    d=[v for v,x in zip(result['guided_rollback'],items) if x['skill']=='D']
    gain=ci(d);specific=specificity(result['guided_rollback'],items)
    acc=lambda name:sum(x['strict_correct'] for x in outputs[name] if x['skill']=='D')/len(cohorts['D'])
    validated=gain['query_bootstrap_95'][0]>0 and specific['query_bootstrap_95'][0]>0 and acc('guided_rollback')>acc('after')
    write(root/'validation-decision.json',{'validated_grouping_update_contribution':validated,'numeric_gain':gain,'specificity_vs_numeric_copy':specific,
          'autonomous_D_before':acc('before'),'autonomous_D_after':acc('after'),'autonomous_D_guided_rollback':acc('guided_rollback'),
          'selection':'Discovery only; no validation reranking. Protection arms run regardless, with validation failures explicitly labeled.'})


def module_prefix(site):
    layer,kind=site.split(':')
    if kind not in ['mlp','self_attn']:raise ValueError('Cannot protect residual pseudo-module')
    return f'model.layers.{int(layer)}.{kind}.'


def protect_adapters(model,sites):
    protected=[]
    for name,p in model.named_parameters():
        if 'lora_' in name and any(module_prefix(s) in name for s in sites):
            p.requires_grad_(False);protected.append(name)
    if sites and not protected:raise RuntimeError('No adapters matched selected sites')
    return protected


def checkpoint_module_hashes(checkpoint,sites):
    from safetensors import safe_open
    hashes={s:hashlib.sha256() for s in sites};counts=Counter()
    for path in sorted(Path(checkpoint).glob('*.safetensors')):
        with safe_open(path,framework='pt',device='cpu') as f:
            for key in sorted(f.keys()):
                for s in sites:
                    if key.startswith(module_prefix(s)):
                        t=f.get_tensor(key);hashes[s].update(key.encode());hashes[s].update(t.contiguous().view(torch.uint8).numpy().tobytes());counts[s]+=t.numel()
    if any(not counts[s] for s in sites):raise RuntimeError('No exported tensors matched protected sites')
    return {s:{'sha256':h.hexdigest(),'parameters':counts[s]} for s,h in hashes.items()}


def train(args,cfg):
    import peft
    from continual import nonlinear_layers as training
    selected=read(Path(args.root)/'selection.json');sites=[] if args.constraint=='none' else selected[args.constraint]
    original=peft.get_peft_model;record={}
    def wrapped(*a,**kw):
        model=original(*a,**kw);record['protected_adapter_names']=protect_adapters(model,sites)
        record['protected_adapter_parameters']=sum(p.numel() for n,p in model.named_parameters() if n in record['protected_adapter_names'])
        return model
    peft.get_peft_model=wrapped
    try:training.train(args,cfg)
    finally:peft.get_peft_model=original
    before=checkpoint_module_hashes(args.source,sites);after=checkpoint_module_hashes(Path(args.output)/'checkpoint',sites)
    if before!=after:raise RuntimeError('Protected module weights changed during compression')
    stats=read(Path(args.output)/'training.json')
    stats.update({'constraint':args.constraint,'protected_modules':sites,**record,'protected_module_hashes':after,
                  'constraint_scope':'Exact module-weight preservation. Upstream changes may still change its inputs and outputs.'})
    write(Path(args.output)/'training.json',stats)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['trace','train']);p.add_argument('--root',required=True)
    for name in ['config','source','data','output','schedule','old-cache','new-cache']:p.add_argument('--'+name)
    p.add_argument('--constraint',choices=['none','guided','random']);p.add_argument('--arm',default='output');args=p.parse_args()
    root=Path(args.root).resolve();cfg=read(args.config or root/'config.json')
    if args.action=='train':train(args,cfg);return
    started=time.perf_counter();torch.manual_seed(cfg['seed'])
    _,tok,before=load_runtime(cfg['before_checkpoint']);_,_,after=load_runtime(cfg['after_checkpoint']);before.eval();after.eval()
    torch.cuda.reset_peak_memory_stats()
    items=read(root/'data/discovery.json');result=trace(before,after,tok,items,cfg,root);write(root/'trace.json',result)
    selected=choose(result,items,cfg);write(root/'selection.json',selected)
    validate(before,after,tok,read(root/'data/validation.json'),selected,cfg,root)
    write(root/'trace-resources.json',{'seconds':time.perf_counter()-started,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
          'gpu':torch.cuda.get_device_name(),'torch_version':torch.__version__,'cuda_version':torch.version.cuda,
          'before_parameters':sum(p.numel() for p in before.parameters()),'after_parameters':sum(p.numel() for p in after.parameters())})


if __name__=='__main__':main()
