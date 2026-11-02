#!/usr/bin/env python3
"""Reproduce one update, preserve its weights, then freeze and trace it."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from continual.tasks import generate,general_examples
from continual.worker import write


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def freeze(root,name,paths,extras=None):
    sources={}
    for p in paths:
        key=str(p.relative_to(ROOT));target=root/'sources'/name/key
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target);sources[key]=sha(p)
    write(root/f'frozen-{name}.json',{'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'sources':sources,**(extras or {})})


def check(root):
    f=read(root/'frozen-reproduction.json')
    for key,value in f['sources'].items():
        if sha(ROOT/key)!=value:raise RuntimeError('Reproduction source changed: '+key)
    for key,value in f['inputs'].items():
        if sha(root/key)!=value:raise RuntimeError('Frozen input changed: '+key)


def call(root,module,action,source,data,output,config,**kwargs):
    check(root);output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,'-m',module,action,'--source',str(source),'--data',str(data),
         '--output',str(output),'--config',str(config)]
    for k,v in kwargs.items():cmd+=['--'+k.replace('_','-'),str(v)]
    write(root/'current.json',{'phase':action,'output':str(output)})
    print(action,output.relative_to(root),flush=True);start=time.perf_counter()
    log=output.parent/(output.stem+'-'+action+'.log')
    with log.open('w') as out:subprocess.run(cmd,cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,check=True)
    write(log.with_suffix('.timing.json'),{'wall_seconds':time.perf_counter()-start,'command':cmd})


def prepare(root,cfg):
    root.mkdir(parents=True,exist_ok=True);ref=ROOT/cfg['reference_run'];stage=ref/cfg['reference_stage']
    write(root/'config.json',cfg);write(root/'student-config.json',read(ref/'student-config.json'))
    write(root/'teacher-config.json',read(ref/'teacher-16-config.json'))
    write(root/'train-data.json',read(stage/'train-data.json'))
    write(root/'acquire-data.json',read(stage/'acquire-data.json'))
    discovery=read(ref/'archive/dev/C.json')[:cfg['discovery_examples']]
    write(root/'data/discovery.json',discovery)
    prior=[ROOT/'results/assimilation/experiment-001/archive',ref/'archive']
    seen={x['fingerprint'] for parent in prior for p in parent.rglob('*.json') for x in read(p)}
    fresh=generate({'seed':cfg['evaluation_seed'],'train_examples':1,'dev_examples':256,'test_examples':512})
    validation=[]
    for skill in 'ABCDE':
        n=cfg['validation_sort_examples'] if skill=='C' else cfg['validation_other_examples']
        xs=[x for x in fresh['test'][skill] if x['fingerprint'] not in seen][:n]
        if len(xs)!=n:raise RuntimeError('Insufficient fresh examples.')
        seen.update(x['fingerprint'] for x in xs)
        write(root/'data'/f'validation-{skill}.json',xs);validation+=xs
    write(root/'data/validation.json',validation+general_examples()[:1])
    generation=read(root/'data/validation-C.json')[:cfg['free_generation_sort_examples']]
    generation+=read(root/'data/validation-E.json')[:cfg['free_generation_join_examples']]
    write(root/'data/activation-generation.json',generation)
    weights={str(p):sha(p) for p in (ROOT/cfg['before_checkpoint']).glob('*.safetensors')}
    inputs={str(p.relative_to(root)):sha(p) for p in root.rglob('*.json')}
    paths=[ROOT/'continual/worker.py',ROOT/'continual/tasks.py',ROOT/'continual/assimilation.py',
           Path(__file__).resolve(),ROOT/'configs/causal_assimilation.json']
    freeze(root,'reproduction',paths,{'inputs':inputs,'before_weights':weights,
           'reference_decision':read(stage/'decision.json')})


def reproduce(root,cfg):
    check(root);source=ROOT/cfg['before_checkpoint']
    for p,h in read(root/'frozen-reproduction.json')['before_weights'].items():
        if sha(p)!=h:raise RuntimeError('Before weights changed.')
    if not (root/'acquire/training.json').exists():
        call(root,'continual.worker','train',source,root/'acquire-data.json',root/'acquire',root/'teacher-config.json')
    payload=read(root/'train-data.json')
    for role,xs,teacher in [('old',payload['memory'],source),('new',payload['current'],root/'acquire/checkpoint')]:
        write(root/f'{role}-targets.json',{'role':role,'examples':xs})
        if not (root/f'{role}-cache.pt').exists():
            call(root,'continual.assimilation','cache',teacher,root/f'{role}-targets.json',root/f'{role}-cache.pt',root/'student-config.json')
    if not (root/'after/training.json').exists():
        call(root,'continual.assimilation','train',source,root/'train-data.json',root/'after',root/'student-config.json',
             old_cache=root/'old-cache.pt',new_cache=root/'new-cache.pt')
    for role in ['old','new']:(root/f'{role}-cache.pt').unlink(missing_ok=True)
    write(root/'progress.json',{'status':'reproduced','before':str(source),'after':str(root/'after/checkpoint')})
    write(root/'current.json',{'phase':'reproduced'})


def trace(root,cfg):
    check(root)
    path=ROOT/'continual/causal_diagnosis.py'
    if not (root/'frozen-tracing.json').exists():
        freeze(root,'tracing',[path],{'after_weights':{str(p):sha(p) for p in (root/'after/checkpoint').glob('*.safetensors')},
                                   'selection':'all discovery C examples only; validation never selects sites'})
    for name,h in read(root/'frozen-tracing.json')['sources'].items():
        if sha(ROOT/name)!=h:raise RuntimeError('Frozen tracing source changed.')
    for label,source in [('before',ROOT/cfg['before_checkpoint']),('after',root/'after/checkpoint')]:
        for split in ['discovery','validation']:
            target=root/f'{label}-{split}.json'
            if not target.exists():call(root,'continual.worker','eval',source,root/'data'/f'{split}.json',target,root/'student-config.json')
    cmd=[sys.executable,'-m','continual.causal_diagnosis','--root',str(root)]
    write(root/'current.json',{'phase':'tracing','output':str(root/'trace.json')})
    with (root/'tracing.log').open('w') as out:subprocess.run(cmd,cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,check=True)
    write(root/'progress.json',{'status':'completed'});write(root/'current.json',{'phase':'completed'})


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['reproduce','trace'],required=True)
    p.add_argument('--config',default='configs/causal_assimilation.json');p.add_argument('--output',default='results/causal_assimilation/diagnostic-001')
    a=p.parse_args();root=(ROOT/a.output).resolve();cfg=read(ROOT/a.config)
    if not (root/'frozen-reproduction.json').exists():prepare(root,cfg)
    if read(root/'config.json')!=cfg:raise RuntimeError('Configuration changed.')
    try:(reproduce if a.phase=='reproduce' else trace)(root,cfg)
    except Exception as exc:write(root/'error.json',{'type':type(exc).__name__,'message':str(exc)});raise


if __name__=='__main__':main()
