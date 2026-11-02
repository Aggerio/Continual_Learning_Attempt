#!/usr/bin/env python3
"""Frozen single-boundary screen of nonlinear expansion and compact compression."""
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
from continual.nonlinear_layers import make_schedule
from continual.tasks import generate, general_examples
from continual.worker import write


def read(p):return json.loads(Path(p).read_text())


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def check(root):
    frozen=read(root/'frozen.json')
    for path,h in frozen['sources'].items():
        if sha(ROOT/path)!=h:raise RuntimeError('Frozen source changed: '+path)
    for path,h in frozen['inputs'].items():
        if sha(root/path)!=h:raise RuntimeError('Frozen input changed: '+path)
    for path,h in frozen['backbone'].items():
        if sha(ROOT/path)!=h:raise RuntimeError('Frozen backbone changed: '+path)


def prepare(root,cfg):
    from transformers import AutoTokenizer
    root.mkdir(parents=True,exist_ok=True)
    write(root/'config.json',cfg)
    original=read(ROOT/cfg['reference_data'])
    payload={'new':original['current'],'old':original['memory']}
    assert len({x['id'] for xs in payload.values() for x in xs})==sum(map(len,payload.values()))
    write(root/'train-data.json',payload)
    tokenizer=AutoTokenizer.from_pretrained(ROOT/cfg['before_checkpoint'],local_files_only=True)
    write(root/'schedule.json',make_schedule(tokenizer,{'current':payload['new'],'memory':payload['old']},cfg))
    seen=set()
    for parent in [ROOT/'results/assimilation/experiment-001/archive',ROOT/'results/capacity_memory/screen-001/archive',
                   ROOT/'results/causal_assimilation/diagnostic-001/data']:
        for path in parent.rglob('*.json'):
            for x in read(path):seen.add(hashlib.sha256(x['prompt'].encode()).hexdigest())
    seen.update(hashlib.sha256(x['prompt'].encode()).hexdigest() for xs in payload.values() for x in xs)
    fresh=generate({'seed':cfg['evaluation_seed'],'train_examples':1,'dev_examples':128,'test_examples':256})
    for split in ['dev','test']:
        xs=[]
        skills=list('ABCDE') if split=='dev' else list('ABCDE')+['BA','BC','BD','BE','BDA','BCE']
        for skill in skills:
            n=cfg['dev_examples'] if split=='dev' else cfg['test_examples'] if len(skill)==1 else cfg['composition_examples']
            chosen=[x for x in fresh[split][skill] if x['fingerprint'] not in seen][:n]
            if len(chosen)!=n:raise RuntimeError('Insufficient fresh cohort')
            seen.update(x['fingerprint'] for x in chosen);xs+=chosen
        write(root/'data'/f'{split}.json',xs+general_examples()[:1])
    dev=read(root/'data/dev.json')
    ablation=[x for skill,n in [('C',32),('D',32),('E',64)] for x in [x for x in dev if x['skill']==skill][:n]]
    write(root/'data/ablation.json',ablation)
    source_files=[ROOT/'continual/nonlinear_layers.py',ROOT/'continual/worker.py',ROOT/'continual/assimilation.py',
                  ROOT/'continual/tasks.py',Path(__file__).resolve(),ROOT/'configs/nonlinear_assimilation.json',
                  ROOT/'tests/test_nonlinear_layers.py']
    sources={}
    for p in source_files:
        target=root/'sources'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
        sources[str(p.relative_to(ROOT))]=sha(p)
    inputs={str(p.relative_to(root)):sha(p) for p in [root/'config.json',root/'train-data.json',root/'schedule.json',*sorted((root/'data').glob('*.json'))]}
    backbone={str(p.relative_to(ROOT)):sha(p) for key in ['before_checkpoint','historical_checkpoint']
              for p in (ROOT/cfg[key]).glob('*.safetensors')}
    write(root/'frozen.json',{'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'sources':sources,'inputs':inputs,'backbone':backbone,
          'selection_evidence_sha256':sha(ROOT/cfg['guided_source']),
          'notes':'Same schedule in teacher and all compression arms; test never selects sites or coefficients.'})


def call(root,module,action,output,data=None,source=None,**kwargs):
    check(root);cfg=read(root/'config.json');output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists() and (output.suffix=='.json' or action=='cache' or (output/'training.json').exists()):return
    cmd=[sys.executable,'-m',module,action,'--config',str(root/'config.json'),
         '--source',str(source or ROOT/cfg['before_checkpoint']),'--output',str(output)]
    if data:cmd+=['--data',str(data)]
    for key,value in kwargs.items():cmd+=['--'+key.replace('_','-'),str(value)]
    write(root/'current.json',{'phase':action,'module':module,'output':str(output),
          'extra':{key:str(value) for key,value in kwargs.items()}})
    print(action,output.relative_to(root),flush=True);started=time.perf_counter()
    log=output.parent/(output.stem+'-'+action+'.log')
    with log.open('w') as out:subprocess.run(cmd,cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,check=True)
    write(log.with_suffix('.timing.json'),{'wall_seconds':time.perf_counter()-started,'command':cmd})


def run(root,cfg):
    data=root/'train-data.json';schedule=root/'schedule.json'
    call(root,'continual.nonlinear_layers','cache',root/'old-targets.pt',data=data,role='old')
    call(root,'continual.nonlinear_layers','train',root/'teacher',data=data,arm='teacher',schedule=schedule,old_cache=root/'old-targets.pt')
    for split in ['dev','test']:
        call(root,'continual.worker','eval',root/f'before-{split}.json',data=root/'data'/f'{split}.json')
    call(root,'continual.worker','eval',root/'historical-dev.json',data=root/'data/dev.json',source=ROOT/cfg['historical_checkpoint'])
    # Matched no-expanded-teacher control: same starting checkpoint, examples,
    # token schedule, old replay/KL and compact LoRA budget; no new-task teacher.
    call(root,'continual.nonlinear_layers','train',root/'students/direct',data=data,arm='direct',schedule=schedule,
         old_cache=root/'old-targets.pt')
    for split in ['dev','test']:
        call(root,'continual.worker','eval',root/'students/direct'/f'{split}.json',
             data=root/'data'/f'{split}.json',source=root/'students/direct/checkpoint')
    write(root/'students/direct/decision.json',{'accepted':None,
          'note':'Matched no-expanded-teacher supervised control; reported descriptively, not used to gate other arms.'})
    call(root,'continual.nonlinear_layers','eval',root/'teacher-dev.json',data=root/'data/dev.json',teacher=root/'teacher')
    teacher=read(root/'teacher-dev.json');before=read(root/'before-dev.json')
    acquired=(teacher['groups']['E']['strict_correct']>=cfg['minimum_acquisition'] and
              teacher['groups']['E']['strict_correct']-before['groups']['E']['strict_correct']>=0.1)
    write(root/'teacher-gate.json',{'acquired':acquired,'join':teacher['groups']['E'],
          'reason':'Proceed to compression' if acquired else 'Expanded teacher did not acquire join; compression not attempted.'})
    for name,bypass in [('bypass-14','14'),('bypass-21','21'),('bypass-both','14,21')]:
        call(root,'continual.nonlinear_layers','eval',root/f'{name}.json',data=root/'data/ablation.json',teacher=root/'teacher',bypass=bypass)
    both={x['id']:x['response'] for x in read(root/'bypass-both.json')['results']}
    if not all(x['response']==both[x['id']] for x in before['results'] if x['id'] in both):
        raise RuntimeError('Bypassing all added layers did not reproduce original model')
    call(root,'continual.nonlinear_layers','eval',root/'teacher-test.json',data=root/'data/test.json',teacher=root/'teacher')
    if acquired:
        call(root,'continual.nonlinear_layers','cache',root/'new-targets.pt',data=data,role='new',teacher=root/'teacher')
        for arm in ['output','guided','random']:
            out=root/'students'/arm
            call(root,'continual.nonlinear_layers','train',out,data=data,arm=arm,schedule=schedule,
                 old_cache=root/'old-targets.pt',new_cache=root/'new-targets.pt')
            for split in ['dev','test']:
                call(root,'continual.worker','eval',out/f'{split}.json',data=root/'data'/f'{split}.json',source=out/'checkpoint')
            report=read(out/'dev.json');reasons=[]
            if report['groups']['E']['strict_correct']<cfg['minimum_acquisition']:reasons.append('Join below acquisition threshold')
            # Re-evaluate accepted history on this same cohort rather than comparing different query samples.
            historical=read(root/'historical-dev.json')
            best={s:max(before['groups'][s]['strict_correct'],historical['groups'][s]['strict_correct'])
                  for s in 'ABC'}
            best['D']=before['groups']['D']['strict_correct']
            for skill,value in best.items():
                if value-report['groups'][skill]['strict_correct']>cfg['retention_tolerance']+1e-9:reasons.append(skill+' exceeds historical retention tolerance')
            if before['general_mc']['accuracy']-report['general_mc']['accuracy']>cfg['general_tolerance']+1e-9:
                reasons.append('Fixed general probe exceeds tolerance')
            write(out/'decision.json',{'accepted':not reasons,'reasons':reasons,'used':'dev only; test does not select an arm',
                  'historical_best_development_reference':best})
    for role in ['old','new']:(root/f'{role}-targets.pt').unlink(missing_ok=True)
    check(root)
    write(root/'progress.json',{'status':'completed','teacher_acquired':acquired,
          'scope':'Single D-to-E boundary; does not establish continual learning or sustained future learnability.'})
    write(root/'current.json',{'phase':'completed'})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='configs/nonlinear_assimilation.json')
    parser.add_argument('--output',default='results/nonlinear_assimilation/screen-001')
    args=parser.parse_args();cfg=read(ROOT/args.config);root=(ROOT/args.output).resolve()
    if not (root/'frozen.json').exists():prepare(root,cfg)
    if cfg!=read(root/'config.json'):raise RuntimeError('Configuration changed')
    try:run(root,cfg)
    except Exception as exc:write(root/'error.json',{'type':type(exc).__name__,'message':str(exc)});raise


if __name__=='__main__':main()
