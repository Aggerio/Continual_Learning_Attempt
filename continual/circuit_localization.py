"""Contrastive primitive probes and held-out causal validation before MLP growth."""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import random
import shutil
import time

import torch

from continual.tasks import compact, score
from continual.worker import load_runtime, prompt_text, write

ROOT=Path(__file__).resolve().parents[1]
SYSTEM='Follow the user instruction exactly. Reply with a JSON integer and nothing else.'


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def make_pair(rng,family,serial):
    a=rng.randrange(2);b=1-a;nonce=''.join(rng.choices('abcdefghjkmnpqrstuvwxyz',k=5))
    field,value=rng.sample(['team','zone','kind','code','score','level'],2)
    template=rng.randrange(3)
    if family=='lookup':
        rows=[{field:'x',value:a},{field:'y',value:b}];rng.shuffle(rows)
        key=rng.choice(['x','y']);other='y' if key=='x' else 'x'
        answer=a if key=='x' else b
        templates=[f'Records: {compact(rows)}. Return the {value} of the record whose {field} equals "KEY".',
                   f'Find the row with {field}="KEY" in {compact(rows)} and output its {value}.',
                   f'From {compact(rows)}, look up {field} "KEY". What is that row\'s {value}?']
        text=templates[template];clean=text.replace('KEY',key);corrupt=text.replace('KEY',other)
        a,b=answer,1-answer
    elif family=='equality':
        first=nonce;other=nonce[:-1]+rng.choice([x for x in 'abcdefghjkmnpqrstuvwxyz' if x!=nonce[-1]])
        second=first if a else other;different=other if a else first
        templates=[f'Compare identifiers "{first}" and "SECOND". Return 1 if they are exactly equal, otherwise 0.',
                   f'Are the strings "{first}" and "SECOND" identical? Output 1 for yes or 0 for no.',
                   f'String A is "{first}". String B is "SECOND". Output the equality indicator: 1 when equal, 0 when different.']
        text=templates[template];clean=text.replace('SECOND',second);corrupt=text.replace('SECOND',different)
    elif family=='copy':
        templates=[f'Object: {{"{value}":VALUE,"tag":"{nonce}"}}. Copy its {value} as a JSON integer.',
                   f'Output the value of {value} in {{"tag":"{nonce}","{value}":VALUE}}.',
                   f'Read {{"{value}":VALUE,"tag":"{nonce}"}} and return only the number stored under {value}.']
        text=templates[template];clean=text.replace('VALUE',str(a));corrupt=text.replace('VALUE',str(b))
    elif family=='constant':
        templates=[f'Reference code: "{nonce}". Ignore that code and output the JSON integer VALUE.',
                   f'For reference "{nonce}", respond with exactly this JSON integer: VALUE.',
                   f'Output VALUE as a JSON integer. The unrelated tag "{nonce}" must not appear in your reply.']
        text=templates[template];clean=text.replace('VALUE',str(a));corrupt=text.replace('VALUE',str(b))
    else:raise ValueError(family)
    fp=hashlib.sha256(compact([SYSTEM,*sorted([clean,corrupt])]).encode()).hexdigest()
    item=lambda text,answer,suffix:{'id':f'{serial}-{family}-{suffix}','skill':family,'system':SYSTEM,
                                    'prompt':text,'answer':answer}
    return {'id':f'{serial}-{family}','family':family,'fingerprint':fp,
            'clean':item(clean,a,'clean'),'corrupt':item(corrupt,b,'corrupt')}


def prepare(root,cfg):
    rng=random.Random(cfg['probe_seed']);seen=set();root.mkdir(parents=True,exist_ok=True)
    write(root/'config.json',cfg)
    for split,key in [('discovery','discovery_pairs_per_family'),('validation','validation_pairs_per_family')]:
        pairs=[]
        for family in cfg['families']:
            chosen=[]
            while len(chosen)<cfg[key]:
                pair=make_pair(rng,family,f'{split}-{len(chosen)}')
                if pair['fingerprint'] in seen:continue
                seen.add(pair['fingerprint']);chosen.append(pair)
            pairs+=chosen
        write(root/'data'/f'{split}.json',pairs)
    paths=[Path(__file__).resolve(),ROOT/'continual/tasks.py',ROOT/'continual/worker.py',ROOT/'configs/circuit_growth.json',
           ROOT/'tests/test_circuit_localization.py']
    sources={}
    for p in paths:
        out=root/'sources'/p.relative_to(ROOT);out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,out)
        sources[str(p.relative_to(ROOT))]=sha(p)
    write(root/'frozen-localization.json',{'sources':sources,
          'inputs':{str(p.relative_to(root)):sha(p) for p in [root/'config.json',*sorted((root/'data').glob('*.json'))]},
          'backbone':{str(p.relative_to(ROOT)):sha(p) for p in (ROOT/cfg['before_checkpoint']).glob('*.safetensors')}})


def check(root):
    frozen=read(root/'frozen-localization.json')
    for key,base in [('sources',ROOT),('inputs',root),('backbone',ROOT)]:
        for name,h in frozen[key].items():
            if sha(base/name)!=h:raise RuntimeError('Frozen '+key+' changed: '+name)


def inputs_for(tokenizer,items,device):
    inputs=tokenizer([prompt_text(tokenizer,x) for x in items],padding=True,add_special_tokens=False,return_tensors='pt').to(device)
    inputs['position_ids']=(inputs['attention_mask'].cumsum(-1)-1).masked_fill(inputs['attention_mask']==0,0)
    return dict(inputs)


@contextmanager
def observations(model,grad=False):
    values={};handles=[]
    def save(key,value):
        if grad and key[0] in ['z','head']:
            value.retain_grad();values[key]=value
        else:values[key]=value[:,-1].detach().cpu()
    for i,layer in enumerate(model.model.layers):
        handles.append(layer.mlp.register_forward_hook(lambda m,a,o,j=i:save(('mlp',j),o)))
        handles.append(layer.mlp.down_proj.register_forward_pre_hook(lambda m,a,j=i:save(('z',j),a[0])))
        handles.append(layer.self_attn.o_proj.register_forward_pre_hook(lambda m,a,j=i:save(('head',j),a[0])))
    handles.append(model.model.layers[-1].register_forward_hook(lambda m,a,o:save(('residual',len(model.model.layers)-1),o)))
    try:yield values
    finally:
        for h in handles:h.remove()


@contextmanager
def intervene(model,kind,layer,donor=None,indices=None,zero=False,shuffle=False):
    block=model.model.layers[layer]
    def changed(tensor):
        result=tensor.clone();value=torch.zeros_like(tensor[:,-1]) if zero else donor.to(tensor.device,tensor.dtype)
        if shuffle:value=value.roll(1,0)
        if indices is None:result[:,-1]=value
        else:result[:,-1,indices]=value[:,indices]
        return result
    if kind in ['z','head']:
        target=block.mlp.down_proj if kind=='z' else block.self_attn.o_proj
        h=target.register_forward_pre_hook(lambda m,a:(changed(a[0]),)+a[1:])
    else:
        target=block.mlp if kind=='mlp' else block
        h=target.register_forward_hook(lambda m,a,o:changed(o))
    try:yield
    finally:h.remove()


def odds(logits,labels,token_ids):
    rows=torch.arange(len(labels),device=logits.device)
    ids=torch.tensor([token_ids[x] for x in labels],device=logits.device)
    other=torch.tensor([token_ids[1-x] for x in labels],device=logits.device)
    return (logits[rows,-1,ids]-logits[rows,-1,other]).float()


def ci(values,cfg):
    rng=random.Random(cfg['random_seed']);n=len(values)
    if not n:return {'mean':None,'query_bootstrap_95':[None,None]}
    samples=sorted(sum(rng.choices(values,k=n))/n for _ in range(cfg['bootstrap_samples']))
    return {'mean':sum(values)/n,'query_bootstrap_95':[samples[int(.025*len(samples))],samples[int(.975*len(samples))-1]]}


def difference_ci(a,b,cfg):
    rng=random.Random(cfg['random_seed'])
    if not a or not b:return {'mean':None,'query_bootstrap_95':[None,None]}
    samples=sorted(sum(rng.choices(a,k=len(a)))/len(a)-sum(rng.choices(b,k=len(b)))/len(b)
                   for _ in range(cfg['bootstrap_samples']))
    return {'mean':sum(a)/len(a)-sum(b)/len(b),
            'query_bootstrap_95':[samples[int(.025*len(samples))],samples[int(.975*len(samples))-1]]}


def subset(values,pairs,families):return [v for v,p in zip(values,pairs) if p['family'] in families]


@torch.no_grad()
def baseline(model,tokenizer,pairs,cfg):
    items=[p[side] for p in pairs for side in ['clean','corrupt']];rows=[]
    for start in range(0,len(items),cfg['batch_size']):
        xs=items[start:start+cfg['batch_size']];inputs=inputs_for(tokenizer,xs,model.device)
        if inputs['input_ids'].shape[1]>cfg['max_prompt_tokens']:raise RuntimeError('Prompt cap exceeded')
        generated=model.generate(**inputs,max_new_tokens=cfg['max_new_tokens'],do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id,use_cache=True)[:,inputs['input_ids'].shape[1]:]
        for x,tokens in zip(xs,generated):
            response=tokenizer.decode(tokens,skip_special_tokens=True)
            rows.append({**x,'response':response,**score(response,x['answer'])})
    return rows


def family_support(pairs,rows,first,cfg):
    index={x['id']:x for x in rows};result={}
    for family in cfg['families']:
        xs=[(i,p) for i,p in enumerate(pairs) if p['family']==family]
        result[family]={'pairs':len(xs),'both_greedy_correct':sum(index[p['clean']['id']]['strict_correct'] and index[p['corrupt']['id']]['strict_correct'] for _,p in xs)/len(xs),
                        'both_first_tokens_correct':sum(first[i][0] and first[i][1] for i,p in xs)/len(xs)}
    return result


def attribution(model,tokenizer,pairs,token_ids,cfg):
    model.requires_grad_(False);model.enable_input_require_grads()
    channels={f:torch.zeros(len(model.model.layers),model.config.intermediate_size) for f in cfg['families']}
    heads={f:torch.zeros(len(model.model.layers),model.config.num_attention_heads) for f in cfg['families']}
    counts={f:0 for f in cfg['families']}
    for start in range(0,len(pairs),cfg['attribution_batch_size']):
        xs=pairs[start:start+cfg['attribution_batch_size']];labels=[p['clean']['answer'] for p in xs]
        clean=inputs_for(tokenizer,[p['clean'] for p in xs],model.device)
        corrupt=inputs_for(tokenizer,[p['corrupt'] for p in xs],model.device)
        with torch.no_grad(),observations(model) as old:
            corrupt_logits=model(**corrupt,use_cache=False,logits_to_keep=1).logits
            old_odds=odds(corrupt_logits,labels,token_ids).cpu()
        with torch.enable_grad(),observations(model,grad=True) as new:
            output=model(**clean,use_cache=False,logits_to_keep=1)
            objective=odds(output.logits,labels,token_ids)
            denom=(objective.detach().cpu()-old_odds).abs().clamp_min(1)
            objective.sum().backward()
        for i,p in enumerate(xs):
            family=p['family'];counts[family]+=1
            for layer in range(len(model.model.layers)):
                for kind,target in [('z',channels),('head',heads)]:
                    tensor=new[(kind,layer)]
                    contribution=tensor.grad[i,-1].float().cpu()*(tensor[i,-1].detach().float().cpu()-old[(kind,layer)][i].float())/denom[i]
                    if kind=='head':contribution=contribution.reshape(model.config.num_attention_heads,model.config.head_dim).sum(-1)
                    target[family][layer]+=contribution
        del output,new,old,corrupt_logits;model.zero_grad(set_to_none=True)
    return {'channels':{f:(x/counts[f]).tolist() for f,x in channels.items()},
            'heads':{f:(x/counts[f]).tolist() for f,x in heads.items()},'counts':counts}


@torch.no_grad()
def causal_sweep(model,tokenizer,pairs,token_ids,cfg,candidates=None):
    sites={f'mlp:{i}':('mlp',i,None) for i in range(len(model.model.layers))} if candidates is None else candidates
    result={'ids':[p['id'] for p in pairs],'denoise':{s:[] for s in sites},'noise':{s:[] for s in sites},
            'first_token_correct':[],'clean_odds':[],'corrupt_odds':[],'self_patch_max_error':0.0,
            'output_copy_max_error':0.0,'peak_last_token_cache_bytes':0}
    for start in range(0,len(pairs),cfg['batch_size']):
        xs=pairs[start:start+cfg['batch_size']];labels=[p['clean']['answer'] for p in xs]
        clean=inputs_for(tokenizer,[p['clean'] for p in xs],model.device)
        corrupt=inputs_for(tokenizer,[p['corrupt'] for p in xs],model.device)
        with observations(model) as a:
            output=model(**clean,use_cache=False,logits_to_keep=1)
            clean_odds=odds(output.logits,labels,token_ids)
            clean_correct=[x==token_ids[y] for x,y in zip(output.logits[:,-1].argmax(-1).tolist(),labels)]
        with observations(model) as b:
            output=model(**corrupt,use_cache=False,logits_to_keep=1)
            corrupt_odds=odds(output.logits,labels,token_ids)
            corrupt_correct=[x==token_ids[1-y] for x,y in zip(output.logits[:,-1].argmax(-1).tolist(),labels)]
        denom=(clean_odds-corrupt_odds).abs().clamp_min(1)
        result['clean_odds']+=clean_odds.tolist();result['corrupt_odds']+=corrupt_odds.tolist()
        result['first_token_correct']+=list(zip(clean_correct,corrupt_correct))
        size=sum(v.numel()*v.element_size() for cache in [a,b] for v in cache.values())
        if size>cfg['activation_cache_bytes']:raise RuntimeError('Last-token cache exceeds bound')
        result['peak_last_token_cache_bytes']=max(size,result['peak_last_token_cache_bytes'])
        for name,site in sites.items():
            kind,layer,indices=site[:3];mode=site[3] if len(site)>3 else ''
            options={'zero':mode=='zero','shuffle':mode=='shuffle'}
            with intervene(model,kind,layer,a[(kind,layer)],indices,**options):
                value=odds(model(**corrupt,use_cache=False,logits_to_keep=1).logits,labels,token_ids)
                result['denoise'][name]+=((value-corrupt_odds)/denom).tolist()
            with intervene(model,kind,layer,b[(kind,layer)],indices,**options):
                value=odds(model(**clean,use_cache=False,logits_to_keep=1).logits,labels,token_ids)
                result['noise'][name]+=((clean_odds-value)/denom).tolist()
        with intervene(model,'mlp',0,a[('mlp',0)]):
            value=odds(model(**clean,use_cache=False,logits_to_keep=1).logits,labels,token_ids)
            result['self_patch_max_error']=max(result['self_patch_max_error'],float((value-clean_odds).abs().max()))
        final=len(model.model.layers)-1
        with intervene(model,'residual',final,a[('residual',final)]):
            value=odds(model(**corrupt,use_cache=False,logits_to_keep=1).logits,labels,token_ids)
            result['output_copy_max_error']=max(result['output_copy_max_error'],float((value-clean_odds).abs().max()))
        print('Causal sweep',start+len(xs),'/',len(pairs),flush=True)
    return result


def select(pairs,sweep,attr,supported,cfg):
    eligible=[f for f in cfg['related_families'] if min(supported[f]['both_greedy_correct'],supported[f]['both_first_tokens_correct'])>=cfg['minimum_primitive_accuracy']]
    ranking=[]
    for site in sweep['denoise']:
        related=subset(sweep['denoise'][site],pairs,eligible)
        controls=subset(sweep['denoise'][site],pairs,cfg['control_families'])
        ranking.append((site,difference_ci(related,controls,cfg)['mean']))
    ranking.sort(key=lambda x:-(x[1] if x[1] is not None else -1e20))
    chosen=[int(site.split(':')[1]) for site,value in ranking[:cfg['selected_sites']]]
    rng=random.Random(cfg['random_seed']);channels={};random_channels={}
    for layer in chosen:
        related=torch.tensor([attr['channels'][f][layer] for f in eligible]).mean(0) if eligible else torch.zeros(3072)
        controls=torch.tensor([attr['channels'][f][layer] for f in cfg['control_families']]).mean(0)
        order=(related-controls).argsort(descending=True).tolist()
        channels[str(layer)]=order[:cfg['channels_per_site']]
        pool=[x for x in order if x not in channels[str(layer)]]
        random_channels[str(layer)]=rng.sample(pool,cfg['channels_per_site'])
    head_score=torch.tensor([attr['heads'][f] for f in eligible]).mean(0) if eligible else torch.zeros(28,16)
    head_score-=torch.tensor([attr['heads'][f] for f in cfg['control_families']]).mean(0)
    head_order=head_score.flatten().argsort(descending=True).tolist()
    selected_heads=[(x//head_score.shape[1],x%head_score.shape[1]) for x in head_order[:4]]
    available=[x for x in head_order if (x//head_score.shape[1],x%head_score.shape[1]) not in selected_heads]
    random_heads=[(x//head_score.shape[1],x%head_score.shape[1]) for x in rng.sample(available,4)]
    control_support=all(min(supported[f]['both_greedy_correct'],supported[f]['both_first_tokens_correct'])>=cfg['minimum_primitive_accuracy'] for f in cfg['control_families'])
    return {'eligible_related_families':eligible,'controls_supported':control_support,
            'candidate_layers':chosen,'module_discovery_ranking':ranking,
            'selected_channels':channels,'random_channels':random_channels,'selected_heads':selected_heads,'random_heads':random_heads,
            'scope':'Last-prompt-position causal readout-path localization; candidates frozen before validation.'}


def validate(pairs,sweep,support,selection,cfg):
    eligible=selection['eligible_related_families'];supported=bool(eligible) and selection['controls_supported'] and all(
        min(support[f]['both_greedy_correct'],support[f]['both_first_tokens_correct'])>=cfg['minimum_primitive_accuracy'] for f in eligible+cfg['control_families'])
    result={'supported_primitives':supported,'sites':{},'eligible_related_families':eligible}
    for layer in selection['candidate_layers']:
        info={};conditions=[]
        for direction in ['denoise','noise']:
            for kind in ['mlp','selected']:
                values=sweep[direction][f'{kind}:{layer}'];rel=subset(values,pairs,eligible);ctrl=subset(values,pairs,cfg['control_families'])
                effect=ci(rel,cfg);specific=difference_ci(rel,ctrl,cfg)
                info[f'{direction}_{kind}_related']=effect;info[f'{direction}_{kind}_specificity']=specific
                conditions+=[effect['query_bootstrap_95'][0] is not None and effect['query_bootstrap_95'][0]>0,
                             specific['query_bootstrap_95'][0] is not None and specific['query_bootstrap_95'][0]>0]
                if kind=='selected':conditions.append(effect['mean'] is not None and effect['mean']>=cfg['minimum_normalized_channel_effect'])
            delta=[a-b for a,b in zip(sweep[direction][f'selected:{layer}'],sweep[direction][f'random:{layer}'])]
            effect=ci(subset(delta,pairs,eligible),cfg);info[direction+'_selected_minus_random']=effect
            conditions.append(effect['query_bootstrap_95'][0] is not None and effect['query_bootstrap_95'][0]>0)
            delta=[a-b for a,b in zip(sweep[direction][f'selected:{layer}'],sweep[direction][f'shuffled:{layer}'])]
            effect=ci(subset(delta,pairs,eligible),cfg);info[direction+'_selected_minus_shuffled']=effect
            conditions.append(effect['query_bootstrap_95'][0] is not None and effect['query_bootstrap_95'][0]>0)
            info[direction+'_zero_ablation_related']=ci(subset(sweep[direction][f'zero:{layer}'],pairs,eligible),cfg)
        info['validated']=bool(supported and all(conditions));result['sites'][str(layer)]=info
    result['proceed_to_growth']=supported and all(x['validated'] for x in result['sites'].values()) and len(result['sites'])==cfg['selected_sites']
    result['reason']='Both discovery-selected sites passed held-out causal specificity tests.' if result['proceed_to_growth'] else 'Localization prerequisite failed; do not train a purported related-circuit placement arm.'
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='configs/circuit_growth.json')
    parser.add_argument('--root',default='results/circuit_growth/screen-001');args=parser.parse_args()
    cfg=read(ROOT/args.config);root=(ROOT/args.root).resolve()
    if not (root/'frozen-localization.json').exists():prepare(root,cfg)
    check(root);started=time.perf_counter()
    _,tokenizer,model=load_runtime(ROOT/cfg['before_checkpoint']);model.eval();torch.manual_seed(cfg['seed'])
    torch.cuda.reset_peak_memory_stats()
    token_ids={i:tokenizer.encode(str(i),add_special_tokens=False) for i in [0,1]}
    if any(len(x)!=1 for x in token_ids.values()):raise RuntimeError('Binary probe labels must be single tokens')
    token_ids={i:x[0] for i,x in token_ids.items()}
    discovery=read(root/'data/discovery.json');validation=read(root/'data/validation.json')
    write(root/'current.json',{'phase':'discovery baseline'})
    rows=baseline(model,tokenizer,discovery,cfg);write(root/'discovery-baseline.json',rows)
    write(root/'current.json',{'phase':'discovery module interventions'})
    sweep=causal_sweep(model,tokenizer,discovery,token_ids,cfg);write(root/'discovery-sweep.json',sweep)
    support=family_support(discovery,rows,sweep['first_token_correct'],cfg);write(root/'discovery-support.json',support)
    write(root/'current.json',{'phase':'discovery neuron and head attribution'})
    attr=attribution(model,tokenizer,discovery,token_ids,cfg);write(root/'discovery-attribution.json',attr)
    selection=select(discovery,sweep,attr,support,cfg);write(root/'selection.json',selection)
    write(root/'current.json',{'phase':'held-out causal validation'})
    sites={}
    for i in selection['candidate_layers']:
        sites[f'mlp:{i}']=('mlp',i,None)
        sites[f'selected:{i}']=('z',i,selection['selected_channels'][str(i)])
        sites[f'random:{i}']=('z',i,selection['random_channels'][str(i)])
        sites[f'shuffled:{i}']=('z',i,selection['selected_channels'][str(i)],'shuffle')
        sites[f'zero:{i}']=('z',i,selection['selected_channels'][str(i)],'zero')
    for category,key in [('selected','selected_heads'),('random','random_heads')]:
        for layer,head in selection[key]:
            sites[f'{category}-head:{layer}:{head}']=('head',layer,list(range(head*model.config.head_dim,(head+1)*model.config.head_dim)))
    rows=baseline(model,tokenizer,validation,cfg);write(root/'validation-baseline.json',rows)
    sweep=causal_sweep(model,tokenizer,validation,token_ids,cfg,sites);write(root/'validation-sweep.json',sweep)
    support=family_support(validation,rows,sweep['first_token_correct'],cfg);write(root/'validation-support.json',support)
    decision=validate(validation,sweep,support,selection,cfg);write(root/'localization-decision.json',decision)
    check(root)
    write(root/'localization-resources.json',{'seconds':time.perf_counter()-started,
          'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'gpu':torch.cuda.get_device_name(),
          'torch_version':torch.__version__,'cuda_version':torch.version.cuda,'parameters':sum(p.numel() for p in model.parameters())})
    write(root/'current.json',{'phase':'localization completed','proceed_to_growth':decision['proceed_to_growth']})


if __name__=='__main__':main()
