#!/usr/bin/env python3
"""Frozen capacity comparison and bounded external-memory screen."""
import argparse
import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from continual.tasks import compact, generate, general_examples
from continual.worker import choose_replay, write
from scripts.assimilation_run import acceptance, anchors


def read(p):
    return json.loads(Path(p).read_text())


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def call(root, module, source, data, output, config, action=None, **extras):
    frozen = read(root/'frozen.json')
    for name, expected in frozen['sources'].items():
        if digest(ROOT/name) != expected:
            raise RuntimeError('Frozen source changed: '+name)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-m', module]
    if action:
        command.append(action)
    command += ['--source', str(source), '--data', str(data), '--output', str(output), '--config', str(config)]
    for key, value in extras.items():
        command += ['--'+key.replace('_', '-'), str(value)]
    write(root/'current.json', {'phase': action or 'memory-eval', 'module': module, 'source': str(source),
                               'output': str(output), 'started': datetime.datetime.now().isoformat()})
    print(action or 'memory-eval', output.relative_to(root), flush=True)
    start = time.perf_counter()
    log = output.parent/(output.stem+'-'+(action or 'memory-eval')+'.log')
    with log.open('w') as stream:
        subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
    write(log.with_suffix('.timing.json'), {'wall_seconds': time.perf_counter()-start, 'command': command})


def prepare(root, cfg):
    from transformers import AutoTokenizer
    old = ROOT/cfg['start_run']
    base = read(old/'frozen.json')['base']
    root.mkdir(parents=True, exist_ok=True)
    write(root/'config.json', cfg)
    training = read(old/'config.json')
    training.update(lora_rank=cfg['student_rank'], intermediate_eval_examples=cfg['development_examples'])
    write(root/'student-config.json', training)
    for rank in cfg['teacher_ranks']:
        write(root/f'teacher-{rank}-config.json', {**training, 'lora_rank': rank,
                                               'stage_training_tokens': training['acquisition_tokens']})
    memory_config = {**training, 'max_sequence_tokens': cfg['retrieval_context_tokens'],
                     'retrieval_examples': cfg['retrieval_examples'],
                     'retrieval_prompt_tokens': cfg['retrieval_prompt_tokens']}
    write(root/'memory-config.json', memory_config)
    seen = {x['fingerprint'] for p in (old/'archive').rglob('*.json') for x in read(p)}
    fresh = generate({'seed': cfg['evaluation_seed'], 'train_examples': 1,
                      'dev_examples': cfg['development_examples']*4, 'test_examples': cfg['test_examples']*4})
    for split, count in [('dev', cfg['development_examples']), ('test', cfg['test_examples'])]:
        for skill, examples in fresh[split].items():
            xs = [x for x in examples if x['fingerprint'] not in seen][:count]
            if len(xs) != count:
                raise RuntimeError('Insufficient fresh evaluation examples.')
            seen.update(x['fingerprint'] for x in xs)
            write(root/'archive'/split/f'{skill}.json', xs)
    shutil.copytree(old/'archive/train', root/'archive/train')
    write(root/'capacity-initial-memory.json', read(old/'streams/composition/D/train-data.json')['memory'])
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    memory, _ = choose_replay(tokenizer, anchors(), training)
    write(root/'external-initial-memory.json', memory)
    sources = {}
    paths = [*sorted((ROOT/'continual').glob('*.py')), Path(__file__).resolve(),
             ROOT/'scripts/assimilation_run.py', ROOT/'configs/capacity_memory.json']
    for p in paths:
        name = str(p.relative_to(ROOT)); target = root/'sources'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target); sources[name] = digest(p)
    inputs = {str(p.relative_to(root)): digest(p) for p in root.rglob('*.json') if 'sources' not in p.parts}
    initial = ROOT/cfg['start_checkpoint']
    write(root/'frozen.json', {'frozen_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'base': base, 'initial_checkpoint': str(initial), 'sources': sources, 'inputs': inputs,
        'initial_weights': {str(p): digest(p) for p in initial.glob('*.safetensors')},
        'packages': {p: importlib.metadata.version(p) for p in ['torch','transformers','peft']},
        'scope': cfg['notes']})


def evaluation(root, split, name, count=None, stage=4):
    skills = list('ABCDE')+(['BDA'] if stage >= 3 else [])+(['BCE'] if stage >= 4 else [])
    examples = []
    for skill in skills:
        xs = read(root/'archive'/split/f'{skill}.json')
        examples += xs[:count] if count else xs
    examples += general_examples()[:1]
    path = root/'evaluation'/f'{name}.json';write(path, examples)
    return path


def memory_eval(root, source, data, output, memory, mode):
    if not Path(output).exists():
        call(root, 'continual.external_memory', source, data, output, root/'memory-config.json', memory=memory, mode=mode)


def run(root, cfg):
    from transformers import AutoTokenizer
    frozen = read(root/'frozen.json')
    for name, expected in frozen['inputs'].items():
        if digest(root/name) != expected:
            raise RuntimeError('Frozen input changed: '+name)
    for name, expected in frozen['initial_weights'].items():
        if digest(name) != expected:
            raise RuntimeError('Initial checkpoint changed.')
    initial, base = frozen['initial_checkpoint'], frozen['base']
    training = read(root/'student-config.json')
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    baseline_data = evaluation(root, 'dev', 'initial-dev')
    baseline = root/'initial-dev.json'
    if not baseline.exists():
        call(root, 'continual.worker', initial, baseline_data, baseline, root/'student-config.json', 'eval')
    base_mc = read(ROOT/cfg['start_run']/'baseline-dev.json')['general_mc']['accuracy']
    summary = {'status': 'running', 'capacity': {}, 'external_stream': [], 'base_general_mc': base_mc}
    write(root/'progress.json', summary)
    final_data = evaluation(root, 'test', 'final-test')
    for rank in cfg['teacher_ranks']:
        arm = root/'capacity'/f'rank-{rank}'
        source = initial
        memory = read(root/'capacity-initial-memory.json')
        best = {k:read(baseline)['groups'][k]['strict_correct'] for k in 'ABC'}
        current_scores = read(baseline)
        stages = []
        for skill in cfg['incoming_order']:
            stage = arm/skill
            current = read(root/'archive/train'/f'{skill}.json')
            stats = {'examples': len(memory), 'tokens':len(tokenizer.encode(compact(memory),add_special_tokens=False)),
                     'bytes':len(compact(memory).encode())}
            payload = {'current':current,'memory':memory,'replay':memory,'replay_stats':stats,'compositions':[]}
            write(stage/'train-data.json', payload)
            if not (stage/'decision.json').exists():
                write(stage/'acquire-data.json', {'current':current})
                if not (stage/'acquire/training.json').exists():
                    call(root,'continual.worker',source,stage/'acquire-data.json',stage/'acquire',
                         root/f'teacher-{rank}-config.json','train')
                teacher_data=stage/'teacher-dev-data.json';write(teacher_data,read(root/'archive/dev'/f'{skill}.json'))
                if not (stage/'teacher-dev.json').exists():
                    call(root,'continual.worker',stage/'acquire/checkpoint',teacher_data,stage/'teacher-dev.json',root/'student-config.json','eval')
                for role,examples,teacher in [('old',memory,source),('new',current,stage/'acquire/checkpoint')]:
                    write(stage/f'{role}-targets.json',{'role':role,'examples':examples})
                    if not (stage/f'{role}-cache.pt').exists():
                        call(root,'continual.assimilation',teacher,stage/f'{role}-targets.json',stage/f'{role}-cache.pt',root/'student-config.json','cache')
                if not (stage/'student/training.json').exists():
                    call(root,'continual.assimilation',source,stage/'train-data.json',stage/'student',root/'student-config.json','train',
                         old_cache=stage/'old-cache.pt',new_cache=stage/'new-cache.pt')
                dev = evaluation(root,'dev',f'rank-{rank}-{skill}-dev',stage=3 if skill=='D' else 4)
                if not (stage/'proposal-dev.json').exists():
                    call(root,'continual.worker',stage/'student/checkpoint',dev,stage/'proposal-dev.json',root/'student-config.json','eval')
                proposal=read(stage/'proposal-dev.json')
                accepted,reasons=acceptance(proposal['groups'],proposal['general_mc']['accuracy'],skill,best,base_mc,training)
                write(stage/'decision.json',{'accepted':accepted,'reasons':reasons,'old_source':str(source),
                    'proposal_source':str(stage/'student/checkpoint'),'best_prior_scores':best,'new_skill_accuracy':proposal['groups'][skill]['strict_correct']})
            decision=read(stage/'decision.json');proposal=read(stage/'proposal-dev.json')
            if decision['accepted']:
                old=Path(source);source=str(stage/'student/checkpoint');current_scores=proposal
                for k in [*best,skill]:best[k]=max(best.get(k,0),proposal['groups'][k]['strict_correct'])
                if str(old)!=initial and str(old)!=source:shutil.rmtree(old,ignore_errors=True)
            else:shutil.rmtree(stage/'student/checkpoint',ignore_errors=True)
            memory,memstats=choose_replay(tokenizer,memory+current,training)
            write(stage/'memory.json',memory);write(stage/'memory-stats.json',memstats)
            for role in ['old','new']:(stage/f'{role}-cache.pt').unlink(missing_ok=True)
            shutil.rmtree(stage/'acquire/checkpoint',ignore_errors=True)
            probe=evaluation(root,'dev',f'rank-{rank}-{skill}-probe',cfg['stream_probe_examples'],3 if skill=='D' else 4)
            for mode in ['off','on']:memory_eval(root,source,probe,stage/f'hybrid-{mode}.json',stage/'memory.json',mode)
            stages.append({'stage':skill,'decision':decision,'proposed':{k:v['strict_correct'] for k,v in proposal['groups'].items()},
                           'accepted_model':{k:v['strict_correct'] for k,v in current_scores['groups'].items()}})
            summary['capacity'][str(rank)]={'stages':stages,'source':source,'memory':str(stage/'memory.json')}
            write(root/'progress.json',summary)
            print('DECISION',rank,skill,decision['accepted'],decision['reasons'],flush=True)
        for mode in ['off','on']:memory_eval(root,source,final_data,arm/f'final-{mode}.json',stage/'memory.json',mode)
    # Memory-only stream: unchanged weights, a genuinely bounded evolving store.
    memory=read(root/'external-initial-memory.json')
    all_probe=evaluation(root,'dev','external-off-probe',cfg['stream_probe_examples'])
    memory_eval(root,base,all_probe,root/'external/off-probe.json',root/'external-initial-memory.json','off')
    for i,skill in enumerate('ABCDE'):
        stage=root/'external'/skill
        incoming=read(root/'archive/train'/f'{skill}.json')
        write(stage/'incoming.json',incoming);write(stage/'previous-memory.json',memory)
        memory,stats=choose_replay(tokenizer,memory+incoming,training)
        write(stage/'memory.json',memory);write(stage/'memory-stats.json',stats)
        probe=evaluation(root,'dev',f'external-{skill}-probe',cfg['stream_probe_examples'],i)
        memory_eval(root,base,probe,stage/'on-probe.json',stage/'memory.json','on')
        summary['external_stream'].append({'stage':skill,'memory':stats,'scores':read(stage/'on-probe.json')['groups']})
        write(root/'progress.json',summary)
    for mode in ['off','on']:memory_eval(root,base,final_data,root/'external'/f'final-{mode}.json',stage/'memory.json',mode)
    negative=evaluation(root,'test','negative-control',cfg['negative_control_examples'])
    for name,source,path in [('base',base,stage/'memory.json')]+[(f'rank-{rank}',v['source'],Path(v['memory'])) for rank,v in summary['capacity'].items()]:
        corrupt=json.loads(json.dumps(read(path)))
        # Intentionally false but syntactically valid JSON demonstrations; never modify real memory.
        for item in corrupt:
            if item['skill']!='anchor':item['answer']=[]
        target=root/'controls'/f'{name}-wrong-memory.json';write(target,corrupt)
        memory_eval(root,source,negative,root/'controls'/f'{name}-wrong-eval.json',target,'on')
    summary['status']='completed';write(root/'progress.json',summary)
    write(root/'current.json',{'phase':'completed'})


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/capacity_memory.json')
    p.add_argument('--output',default='results/capacity_memory/screen-001');a=p.parse_args()
    cfg=read(ROOT/a.config);root=(ROOT/a.output).resolve()
    if not (root/'frozen.json').exists():prepare(root,cfg)
    elif read(root/'config.json')!=cfg:raise RuntimeError('Frozen configuration differs.')
    try:run(root,cfg)
    except Exception as exc:
        write(root/'error.json',{'type':type(exc).__name__,'message':str(exc)});raise


if __name__=='__main__':main()
