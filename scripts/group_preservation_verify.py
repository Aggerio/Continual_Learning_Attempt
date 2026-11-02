#!/usr/bin/env python3
"""Read-only real-model check of the cached decoder used in causal tests."""
import argparse
import gc
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from continual.causal_diagnosis import greedy
from continual.group_preservation import read,sha
from continual.worker import load_runtime,prompt_text,write


def main(root):
    import torch
    verification_started=time.perf_counter()
    if read(root/'progress.json')['status']!='completed':raise RuntimeError('Wait until experiment workers finish')
    cfg=read(root/'config.json');val=read(root/'data/validation.json')
    items=[x for x in val if x['skill']=='D'][:4]+[x for x in val if x['skill']=='E'][:4]
    checks={}
    for role in ['before','after']:
        source=ROOT/cfg[role+'_checkpoint'];_,tok,model=load_runtime(source);model.eval();manual=[];hf=[]
        for start in range(0,len(items),4):
            xs=items[start:start+4];manual+=greedy(model,tok,xs)
            inputs=tok([prompt_text(tok,x) for x in xs],padding=True,add_special_tokens=False,return_tensors='pt').to('cuda')
            with torch.inference_mode():
                ids=model.generate(**inputs,do_sample=False,max_new_tokens=192,use_cache=True,pad_token_id=tok.pad_token_id)
            hf+=tok.batch_decode(ids[:,inputs.input_ids.shape[1]:],skip_special_tokens=True)
        same=all(x['response']==text for x,text in zip(manual,hf));checks[role]={'matched':same,'examples':len(items)}
        if not same:raise RuntimeError('Manual causal decoder differs from model.generate: '+role)
        del model;gc.collect();torch.cuda.empty_cache()
    write(root/'decoder-verification.json',{'status':'passed','checks':checks,'ids':[x['id'] for x in items],
          'source_sha256':sha(Path(__file__)),'causal_decoder_source_sha256':sha(ROOT/'continual/causal_diagnosis.py'),
          'scope':'Read-only supplement after fixed experiment; no training or selection feedback.'})
    print('Real-model cached generation matched model.generate for both checkpoints.')
    # This read-only initialization probe and one original-trainer repeat were
    # planned while the fresh unconstrained run was being evaluated, before
    # outcomes of the protected arms. Neither changes training or selection.
    program='''
import sys,hashlib,torch,json
from pathlib import Path
entry,source,config,out=sys.argv[1:]
if entry=='protected':
 import continual.group_preservation
 import peft
else:
 import continual.circuit_growth
from continual.worker import load_runtime,write
cfg=json.loads(Path(config).read_text());torch.manual_seed(cfg['seed'])
_,tok,base=load_runtime(source)
from peft import LoraConfig,get_peft_model
model=get_peft_model(base,LoraConfig(r=cfg['lora_rank'],lora_alpha=2*cfg['lora_rank'],target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],lora_dropout=0.0,task_type='CAUSAL_LM'))
if entry=='protected':continual.group_preservation.protect_adapters(model,[])
h=hashlib.sha256()
for name,p in sorted(model.named_parameters()):
 if 'lora_' in name:
  h.update(name.encode());h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
write(out,{'entrypoint':entry,'initial_adapter_sha256':h.hexdigest(),'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'deterministic_algorithms_enabled':torch.are_deterministic_algorithms_enabled(),'torch_version':torch.__version__,'cuda_version':torch.version.cuda})
'''
    directory=root/'reproduction';directory.mkdir(exist_ok=True)
    for entry in ['original','protected']:
        out=directory/f'initialization-{entry}.json'
        with (directory/f'initialization-{entry}.log').open('w') as log:
            subprocess.run([sys.executable,'-c',program,entry,str(ROOT/cfg['before_checkpoint']),str(root/'student-config.json'),str(out)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    original=read(directory/'initialization-original.json');protected=read(directory/'initialization-protected.json')
    if original['initial_adapter_sha256']!=protected['initial_adapter_sha256']:raise RuntimeError('Original and protected entrypoints initialize different adapters')
    plan=read(root/'reproduction-plan.json')
    for record in plan['caches'].values():
        if sha(record['path'])!=record['sha256']:raise RuntimeError('Reproduction target cache changed')
    out=directory/'canonical';command=[sys.executable,'-m','continual.circuit_growth','train','--config',str(root/'student-config.json'),
             '--source',str(ROOT/cfg['before_checkpoint']),'--output',str(out),'--data',str(root/'train-data.json'),
             '--arm','output','--schedule',str(root/'schedule.json'),'--old-cache',plan['caches']['old']['path'],'--new-cache',plan['caches']['new']['path']]
    started=time.perf_counter()
    with (directory/'canonical-train.log').open('w') as log:subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    write(directory/'canonical-train.timing.json',{'command':command,'seconds':time.perf_counter()-started})
    hashes=lambda path:[sha(p) for p in sorted(Path(path).glob('*.safetensors'))]
    same=hashes(out/'checkpoint')==hashes(root/'students/none/checkpoint')
    report={'initialization_matches':True,'initial_adapter_sha256':original['initial_adapter_sha256'],
            'original_trainer_checkpoint_matches_fresh_unconstrained':same,'deterministic_algorithms_enabled':original['deterministic_algorithms_enabled'],
            'source_sha256':sha(Path(__file__)),'scope':'One fixed-input original-entrypoint repeat to quantify reproducibility; no checkpoint selection or method tuning.'}
    if not same:
        for split in ['dev','test']:
            command=[sys.executable,'-m','continual.worker','eval','--config',str(root/'student-config.json'),'--source',str(out/'checkpoint'),
                     '--data',str(root/'data'/f'{split}.json'),'--output',str(out/f'{split}.json')]
            started=time.perf_counter()
            with (directory/f'canonical-{split}-eval.log').open('w') as log:subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            write(directory/f'canonical-{split}-eval.timing.json',{'command':command,'seconds':time.perf_counter()-started})
    for record in plan['caches'].values():Path(record['path']).unlink()
    report['verification_target_caches_removed']=True;report['verification_wall_seconds']=time.perf_counter()-verification_started
    write(root/'reproduction-verification.json',report)
    print('Original/protected initializations match; canonical checkpoint match:',same)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='results/group_preservation/screen-001');args=p.parse_args();main((ROOT/args.root).resolve())
