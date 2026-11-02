#!/usr/bin/env python3
"""Frozen numeric tracing and matched update-protection experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from continual.group_preservation import read,sha
from continual.tasks import generate,general_examples
from continual.worker import write


def prepare(root):
    cfg=read(ROOT/'configs/group_preservation.json');reference=ROOT/cfg['reference_run']
    from scripts.circuit_growth_run import check as check_reference
    check_reference(reference)
    root.mkdir(parents=True,exist_ok=True);write(root/'config.json',cfg)
    student=read(reference/'configs/related.json');student['notes']='Matched output compression; only selected LoRA modules are frozen in protected arms.'
    write(root/'student-config.json',student)
    for name in ['train-data.json','schedule.json']:shutil.copy2(reference/name,root/name)
    seen=set()
    parents=[ROOT/'results/assimilation/experiment-001/archive',ROOT/'results/capacity_memory/screen-001/archive',
             ROOT/'results/causal_assimilation/diagnostic-001/data',ROOT/'results/nonlinear_assimilation/screen-002/data',reference/'data']
    for parent in parents:
        for p in parent.rglob('*.json'):
            for x in read(p):seen.add(hashlib.sha256(x['prompt'].encode()).hexdigest())
    for xs in read(root/'train-data.json').values():
        for x in xs:seen.add(hashlib.sha256(x['prompt'].encode()).hexdigest())
    def fresh(seed,counts):
        pools=generate({'seed':seed,'train_examples':1,'dev_examples':512,'test_examples':1})['dev'];items=[]
        for s,n in counts.items():
            xs=[x for x in pools[s] if x['fingerprint'] not in seen][:n]
            if len(xs)!=n:raise RuntimeError('Insufficient fresh probes')
            seen.update(x['fingerprint'] for x in xs);items+=xs
        return items
    for i,split in enumerate(['discovery','validation']):
        items=fresh(cfg['probe_seed']+i,cfg[split+'_counts'])
        for x in items:x['id']=split+'-fresh-'+x['id']
        write(root/'data'/f'{split}.json',items)
    for i,split in enumerate(['dev','test']):
        counts={s:cfg['dev_atomic_examples' if split=='dev' else 'test_atomic_examples'] for s in 'ABCDE'}
        counts.update({s:cfg['dev_composition_examples' if split=='dev' else 'test_composition_examples'] for s in ['BA','BC','BD','BE','BDA','BCE']})
        items=fresh(cfg['evaluation_seed']+i,counts)
        # New generator calls all use the dev namespace; give each cohort unique IDs.
        for x in items:x['id']=split+'-fresh-'+x['id']
        write(root/'data'/f'{split}.json',items+general_examples()[:1])
    sources={}
    paths=['continual/group_preservation.py','scripts/group_preservation_run.py','configs/group_preservation.json','tests/test_group_preservation.py','GROUP_PRESERVATION_STEPS.md',
           'continual/nonlinear_layers.py','continual/circuit_growth.py','continual/causal_diagnosis.py','continual/tasks.py','continual/worker.py','continual/assimilation.py']
    for name in paths:
        p=ROOT/name;out=root/'sources'/name;out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,out);sources[name]=sha(p)
    inputs=[root/'config.json',root/'student-config.json',root/'train-data.json',root/'schedule.json',*sorted((root/'data').glob('*.json'))]
    weights={str(p.relative_to(ROOT)):sha(p) for checkpoint in [cfg['before_checkpoint'],cfg['after_checkpoint']] for p in (ROOT/checkpoint).glob('*.safetensors')}
    teacher=reference/'teachers/related/layers.safetensors';weights[str(teacher.relative_to(ROOT))]=sha(teacher)
    write(root/'frozen.json',{'sources':sources,'inputs':{str(p.relative_to(root)):sha(p) for p in inputs},'weights':weights,
          'scope':'One D-to-E compression boundary. Discovery-selected protection, no validation reranking or hyperparameter search. No candidate auto-deployed.'})


def check(root,weights=False):
    frozen=read(root/'frozen.json')
    for key,base in [('sources',ROOT),('inputs',root)]+([('weights',ROOT)] if weights else []):
        for name,h in frozen[key].items():
            if sha(base/name)!=h:raise RuntimeError('Frozen '+key+' changed: '+name)


def call(experiment,module,action,out,source=None,data=None,config=None,**extras):
    root=experiment
    check(root);out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists() and (out.suffix in ['.json','.pt'] or (out/'training.json').exists()):return
    cmd=[sys.executable,'-m',module,action,'--config',str(config or root/'student-config.json')]
    if action=='trace':cmd+=['--root',str(root)]
    else:cmd+=['--source',str(source),'--output',str(out),'--data',str(data)]
    for k,v in extras.items():cmd+=['--'+k.replace('_','-'),str(v)]
    write(root/'current.json',{'phase':action,'output':str(out),'extras':{k:str(v) for k,v in extras.items()}})
    started=time.perf_counter();print(action,out.relative_to(root),flush=True)
    log=out.parent/(out.stem+'-'+action+'.log')
    with log.open('w') as f:subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    write(log.with_suffix('.timing.json'),{'seconds':time.perf_counter()-started,'command':cmd})


def decisions(root,name):
    cfg=read(root/'student-config.json');base=read(root/'before-dev.json');history=read(root/'historical-dev.json');r=read(root/'students'/name/'dev.json')
    current={s:base['groups'][s]['strict_correct'] for s in ['A','B','C','D','BA','BC','BD','BDA']}
    historical={**current,**{s:max(current[s],history['groups'][s]['strict_correct']) for s in ['A','B','C','BA','BC']}}
    output={}
    for label,refs in [('current_checkpoint',current),('historical_best',historical)]:
        reasons=[]
        if r['groups']['E']['strict_correct']<cfg['minimum_acquisition']:reasons.append('E below acquisition threshold')
        for s,v in refs.items():
            if v-r['groups'][s]['strict_correct']>cfg['retention_tolerance']+1e-9:reasons.append(s+' exceeds retention tolerance')
        if base['general_mc']['accuracy']-r['general_mc']['accuracy']>cfg['general_tolerance']+1e-9:reasons.append('General probe exceeds tolerance')
        output[label]={'accepted':not reasons,'reasons':reasons,'references':refs}
    output['selection']='Development only; test data never used for acceptance'
    write(root/'students'/name/'decision.json',output)


def run(root):
    cfg=read(root/'config.json');student=read(root/'student-config.json');reference=ROOT/cfg['reference_run'];source=ROOT/cfg['before_checkpoint'];data=root/'train-data.json'
    if not (root/'trace-resources.json').exists():call(root,'continual.group_preservation','trace',root/'trace-resources.json',config=root/'config.json')
    selection=read(root/'selection.json');write(root/'frozen-selection.json',{'sha256':sha(root/'selection.json'),'validation_decision_sha256':sha(root/'validation-decision.json')})
    call(root,'continual.circuit_growth','cache',root/'old-targets.pt',source=source,data=data,role='old')
    call(root,'continual.circuit_growth','cache',root/'new-targets.pt',source=source,data=data,role='new',teacher=reference/'teachers/related')
    for split in ['dev','test']:
        for name,checkpoint in [('before',source),('previous-compressed',ROOT/cfg['after_checkpoint']),('previous-direct',reference/'students/direct/checkpoint')]:
            call(root,'continual.worker','eval',root/f'{name}-{split}.json',source=checkpoint,data=root/'data'/f'{split}.json')
    call(root,'continual.worker','eval',root/'historical-dev.json',source=ROOT/student['historical_checkpoint'],data=root/'data/dev.json')
    for name in ['none','guided','random']:
        out=root/'students'/name
        call(root,'continual.group_preservation','train',out,source=source,data=data,root=root,constraint=name,arm='output',
             schedule=root/'schedule.json',old_cache=root/'old-targets.pt',new_cache=root/'new-targets.pt')
        for split in ['dev','test']:call(root,'continual.worker','eval',out/f'{split}.json',source=out/'checkpoint',data=root/'data'/f'{split}.json')
        decisions(root,name)
    for name in ['old-targets.pt','new-targets.pt']:(root/name).unlink()
    check(root,weights=True)
    if sha(root/'selection.json')!=read(root/'frozen-selection.json')['sha256']:raise RuntimeError('Selection changed during training')
    write(root/'progress.json',{'status':'completed','validated_trace':read(root/'validation-decision.json')['validated_grouping_update_contribution'],
          'scope':'Single compression protection screen, not continual-learning proof'})
    write(root/'current.json',{'phase':'completed'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='results/group_preservation/screen-001');args=p.parse_args();root=(ROOT/args.root).resolve()
    if not (root/'frozen.json').exists():prepare(root)
    try:run(root)
    except Exception as exc:write(root/'error.json',{'type':type(exc).__name__,'message':str(exc)});raise
