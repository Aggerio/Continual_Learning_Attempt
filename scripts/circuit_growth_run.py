#!/usr/bin/env python3
"""Matched native-channel growth and compact output-only compression screen."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from continual.circuit_localization import check as check_local
from continual.nonlinear_layers import make_schedule
from continual.tasks import generate,general_examples
from continual.worker import write


def read(p):return json.loads(Path(p).read_text())


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for part in iter(lambda:f.read(8*1024*1024),b''):h.update(part)
    return h.hexdigest()


def check(root):
    frozen=read(root/'frozen-growth.json')
    for key,base in [('sources',ROOT),('inputs',root),('localization',root.parent)]:
        for name,h in frozen[key].items():
            if sha(base/name)!=h:raise RuntimeError('Frozen '+key+' changed: '+name)
    check_local(root.parent)


def prepare(root):
    from transformers import AutoTokenizer
    local=root.parent;check_local(local)
    if not read(local/'localization-decision.json')['proceed_to_growth']:raise RuntimeError('Localization did not validate two sites')
    cfg=read(ROOT/'configs/nonlinear_assimilation.json')
    cfg.update({'training_tokens':98304,'channels_per_site':64,'guided_layers':[],'random_layers':[],
                'eval_batch_size':16,'teacher_cache_bytes':16777216,'evaluation_seed':20261107,
                'dev_atomic_examples':32,'dev_composition_examples':32,'test_examples':64,'composition_examples':32,
                'notes':'Placement comparison after independent localization; output-only compression held constant. Direct control receives no expanded-teacher targets. Old composition retention is gated.'})
    cfg.pop('bottleneck',None);cfg.pop('guided_source',None)
    root.mkdir(parents=True,exist_ok=True)
    selection=read(local/'selection.json');rng=random.Random(1701)
    arms={'related':selection['candidate_layers'],'random':rng.sample(range(28),2),'fixed':[14,21]}
    cfg['growth_arms']=arms;write(root/'config.json',cfg)
    for name,sites in arms.items():write(root/'configs'/f'{name}.json',{**cfg,'insertion_layers':sites})
    write(root/'configs/direct.json',{**cfg,'insertion_layers':[]})
    original=read(ROOT/cfg['reference_data']);payload={'new':original['current'],'old':original['memory']}
    write(root/'train-data.json',payload)
    tok=AutoTokenizer.from_pretrained(ROOT/cfg['before_checkpoint'],local_files_only=True)
    write(root/'schedule.json',make_schedule(tok,{'current':payload['new'],'memory':payload['old']},cfg))
    seen=set()
    parents=[ROOT/'results/assimilation/experiment-001/archive',ROOT/'results/capacity_memory/screen-001/archive',
             ROOT/'results/causal_assimilation/diagnostic-001/data',ROOT/'results/nonlinear_assimilation/screen-002/data']
    for parent in parents:
        for p in parent.rglob('*.json'):
            for x in read(p):seen.add(hashlib.sha256(x['prompt'].encode()).hexdigest())
    seen.update(hashlib.sha256(x['prompt'].encode()).hexdigest() for xs in payload.values() for x in xs)
    fresh=generate({'seed':cfg['evaluation_seed'],'train_examples':1,'dev_examples':128,'test_examples':256})
    for split in ['dev','test']:
        examples=[];skills=list('ABCDE')+(['BA','BC','BD','BDA'] if split=='dev' else ['BA','BC','BD','BE','BDA','BCE'])
        for skill in skills:
            n=cfg['dev_atomic_examples'] if split=='dev' and len(skill)==1 else cfg['dev_composition_examples'] if split=='dev' else cfg['test_examples'] if len(skill)==1 else cfg['composition_examples']
            xs=[x for x in fresh[split][skill] if x['fingerprint'] not in seen][:n]
            if len(xs)!=n:raise RuntimeError('Insufficient fresh cohort')
            seen.update(x['fingerprint'] for x in xs);examples+=xs
        write(root/'data'/f'{split}.json',examples+general_examples()[:1])
    write(root/'data/ablation.json',[x for x in read(root/'data/dev.json') if x['skill']=='E'])
    paths=[Path(__file__).resolve(),ROOT/'continual/circuit_growth.py',ROOT/'continual/nonlinear_layers.py',ROOT/'continual/worker.py',
           ROOT/'continual/tasks.py',ROOT/'continual/assimilation.py',ROOT/'tests/test_circuit_growth.py']
    sources={}
    for p in paths:
        target=root/'sources'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
        sources[str(p.relative_to(ROOT))]=sha(p)
    inputs=[root/'config.json',root/'train-data.json',root/'schedule.json',*sorted((root/'configs').glob('*.json')),*sorted((root/'data').glob('*.json'))]
    write(root/'frozen-growth.json',{'sources':sources,'inputs':{str(p.relative_to(root)):sha(p) for p in inputs},
          'localization':{name:sha(local/name) for name in ['selection.json','localization-decision.json']},
          'placement_overlap':{name:sorted(set(arms[name])&set(arms['related'])) for name in ['random','fixed']},
          'scope':'One D-to-E boundary. Fixed criteria, one seed, no validation-based reranking. No candidate auto-deployed.'})


def call(root,module,action,out,data=None,config=None,source=None,**extras):
    check(root);cfg=read(root/'config.json');out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists() and (out.suffix in ['.json','.pt'] or (out/'training.json').exists()):return
    cmd=[sys.executable,'-m',module,action,'--config',str(config or root/'configs/direct.json'),
         '--source',str(source or ROOT/cfg['before_checkpoint']),'--output',str(out)]
    if data:cmd+=['--data',str(data)]
    for key,value in extras.items():cmd+=['--'+key.replace('_','-'),str(value)]
    write(root/'current.json',{'phase':action,'output':str(out),'extras':{k:str(v) for k,v in extras.items()}})
    print(action,out.relative_to(root),flush=True);started=time.perf_counter()
    log=out.parent/(out.stem+'-'+action+'.log')
    with log.open('w') as f:subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    write(log.with_suffix('.timing.json'),{'seconds':time.perf_counter()-started,'command':cmd})


def acceptance(root,out):
    cfg=read(root/'config.json');before=read(root/'before-dev.json');history=read(root/'historical-dev.json');candidate=read(out/'dev.json')
    best={s:max(before['groups'][s]['strict_correct'],history['groups'][s]['strict_correct']) for s in ['A','B','C','BA','BC']}
    best.update({s:before['groups'][s]['strict_correct'] for s in ['D','BD','BDA']})
    reasons=[]
    if candidate['groups']['E']['strict_correct']<cfg['minimum_acquisition']:reasons.append('E below acquisition threshold')
    for skill,value in best.items():
        if value-candidate['groups'][skill]['strict_correct']>cfg['retention_tolerance']+1e-9:reasons.append(skill+' exceeds retention tolerance')
    if before['general_mc']['accuracy']-candidate['general_mc']['accuracy']>cfg['general_tolerance']+1e-9:reasons.append('General probe exceeds tolerance')
    write(out/'decision.json',{'accepted':not reasons,'reasons':reasons,'historical_reference_same_dev':best,'selection':'dev only'})


def run(root):
    cfg=read(root/'config.json');data=root/'train-data.json';schedule=root/'schedule.json'
    call(root,'continual.circuit_growth','cache',root/'old-targets.pt',data=data,role='old')
    for split in ['dev','test']:
        call(root,'continual.worker','eval',root/f'before-{split}.json',data=root/'data'/f'{split}.json')
    call(root,'continual.worker','eval',root/'historical-dev.json',data=root/'data/dev.json',source=ROOT/cfg['historical_checkpoint'])
    # Direct learner matches each compact student's trainable capacity and schedule,
    # with CE + old-teacher KL, but no expanded-teacher new-task target.
    direct=root/'students/direct'
    call(root,'continual.circuit_growth','train',direct,data=data,arm='output',schedule=schedule,old_cache=root/'old-targets.pt')
    for split in ['dev','test']:
        call(root,'continual.worker','eval',direct/f'{split}.json',data=root/'data'/f'{split}.json',source=direct/'checkpoint')
    acceptance(root,direct)
    for name,sites in cfg['growth_arms'].items():
        armcfg=root/'configs'/f'{name}.json';teacher=root/'teachers'/name
        call(root,'continual.circuit_growth','train',teacher,data=data,config=armcfg,arm='teacher',schedule=schedule,old_cache=root/'old-targets.pt')
        for split in ['dev','test']:
            call(root,'continual.circuit_growth','eval',teacher/f'{split}.json',data=root/'data'/f'{split}.json',config=armcfg,teacher=teacher)
        for label,bypass in [('first',str(sites[0])),('second',str(sites[1])),('both',','.join(map(str,sites)))]:
            call(root,'continual.circuit_growth','eval',teacher/f'bypass-{label}.json',data=root/'data/ablation.json',config=armcfg,teacher=teacher,bypass=bypass)
        before={x['id']:x['response'] for x in read(root/'before-dev.json')['results']}
        if any(before[x['id']]!=x['response'] for x in read(teacher/'bypass-both.json')['results']):raise RuntimeError('Bypass control failed')
        dev=read(teacher/'dev.json');base=read(root/'before-dev.json')
        acquired=dev['groups']['E']['strict_correct']>=cfg['minimum_acquisition'] and dev['groups']['E']['strict_correct']-base['groups']['E']['strict_correct']>=.1
        write(teacher/'gate.json',{'acquired':acquired,'join':dev['groups']['E']['strict_correct']})
        if acquired:
            target=root/f'{name}-targets.pt'
            call(root,'continual.circuit_growth','cache',target,data=data,config=armcfg,role='new',teacher=teacher)
            student=root/'students'/name
            call(root,'continual.circuit_growth','train',student,data=data,config=armcfg,arm='output',schedule=schedule,old_cache=root/'old-targets.pt',new_cache=target)
            for split in ['dev','test']:
                call(root,'continual.worker','eval',student/f'{split}.json',data=root/'data'/f'{split}.json',source=student/'checkpoint')
            acceptance(root,student);target.unlink()
    (root/'old-targets.pt').unlink();check(root)
    write(root/'progress.json',{'status':'completed','scope':'One validated-placement acquisition/compression boundary; not continual-learning proof'})
    write(root/'current.json',{'phase':'completed'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='results/circuit_growth/screen-001/growth');args=p.parse_args();root=(ROOT/args.root).resolve()
    if not (root/'frozen-growth.json').exists():prepare(root)
    try:run(root)
    except Exception as exc:write(root/'error.json',{'type':type(exc).__name__,'message':str(exc)});raise
