#!/usr/bin/env python3
"""Independent scoring, provenance, size audit and report for added layers."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from continual.tasks import compact, execute, score
from continual.worker import prompt_text, write


def read(p):return json.loads(Path(p).read_text())


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def accuracy(rows,skill):
    xs=[x for x in rows if x['skill']==skill]
    return sum(x['strict_correct'] for x in xs)/len(xs)


def paired(a,b,skill):
    aa={x['id']:x['strict_correct'] for x in a if x['skill']==skill}
    bb={x['id']:x['strict_correct'] for x in b if x['skill']==skill}
    assert aa.keys()==bb.keys()
    values=[int(bb[k])-int(aa[k]) for k in aa];rng=random.Random(1701)
    means=sorted(sum(rng.choices(values,k=len(values)))/len(values) for _ in range(2000))
    return {'n':len(values),'gains':sum(x==1 for x in values),'losses':sum(x==-1 for x in values),
            'mean':sum(values)/len(values),'query_bootstrap_95':[means[49],means[1949]]}


def audit(root):
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    torch.set_num_threads(4)
    cfg=read(root/'config.json');progress=read(root/'progress.json');assert progress['status']=='completed'
    frozen=read(root/'frozen.json');checks={}
    for key,base in [('sources',ROOT),('inputs',root),('backbone',ROOT)]:
        for name,h in frozen[key].items():assert sha(base/name)==h,name
        checks['frozen_'+key]=True
    assert sha(ROOT/cfg['guided_source'])==frozen['selection_evidence_sha256']
    prior=read(ROOT/cfg['guided_source'])
    assert sorted(int(s.split(':')[0]) for s in prior['weight_variants']['top3'])==cfg['guided_layers']
    assert sorted(int(s.split(':')[0]) for s in prior['weight_variants']['random3'])==cfg['random_layers']
    data=read(root/'train-data.json');dev=read(root/'data/dev.json');test=read(root/'data/test.json')
    train_hashes={hashlib.sha256(x['prompt'].encode()).hexdigest() for xs in data.values() for x in xs}
    old=set()
    for parent in [ROOT/'results/assimilation/experiment-001/archive',ROOT/'results/capacity_memory/screen-001/archive',
                   ROOT/'results/causal_assimilation/diagnostic-001/data']:
        for p in parent.rglob('*.json'):
            for x in read(p):old.add(hashlib.sha256(x['prompt'].encode()).hexdigest())
    cohorts=[]
    for xs in [dev,test]:
        hashes=set()
        for x in xs:
            if 'fingerprint' not in x:continue
            fp=hashlib.sha256(x['prompt'].encode()).hexdigest();assert fp==x['fingerprint']
            assert fp not in train_hashes and fp not in old and fp not in hashes
            program=json.loads(x['prompt']);assert compact(execute(program['rows'],program['ops']))==compact(x['answer'])
            hashes.add(fp)
        cohorts.append(hashes)
    assert not cohorts[0]&cohorts[1]
    checks['fresh_atomic_and_composition_cohorts']=True
    assert len(dev)==321 and len(test)==513
    tokenizer=AutoTokenizer.from_pretrained(ROOT/cfg['before_checkpoint'],local_files_only=True)
    prompt_lengths=[len(tokenizer.encode(prompt_text(tokenizer,x),add_special_tokens=False)) for x in dev+test]
    answer_lengths=[len(tokenizer.encode(compact(x['answer'])+tokenizer.eos_token,add_special_tokens=False)) for x in dev+test]
    assert max(prompt_lengths)<=cfg['max_sequence_tokens']
    assert max(answer_lengths)<=cfg['max_new_tokens']
    checks['evaluation_token_limits']={'max_prompt_tokens':max(prompt_lengths),'max_gold_answer_tokens':max(answer_lengths),
                                       'prompt_cap':cfg['max_sequence_tokens'],'generation_cap':cfg['max_new_tokens']}
    memory_tokens=len(tokenizer.encode(compact(data['old']),add_special_tokens=False))
    assert len(data['old'])<=cfg['replay_examples'] and memory_tokens<=cfg['replay_tokens']
    checks['bounded_replay']={'examples':len(data['old']),'tokens':memory_tokens}
    train=read(root/'teacher/training.json')
    assert train['backbone_before_sha256']==train['backbone_after_sha256']
    assert train['base_parameters']==596049920 and train['trainable_parameters']==1050624
    assert train['expanded_parameters']==597100544
    extra=load_file(str(root/'teacher/layers.safetensors'))
    assert sum(t.numel() for t in extra.values())==1050624
    assert all(torch.isfinite(t).all() for t in extra.values())
    assert all(extra[f'{i}.down.weight'].abs().sum()>0 for i in cfg['insertion_layers'])
    checks['only_added_layers_trained_backbone_unchanged']=True
    schedule=read(root/'schedule.json');assert schedule['tokens']<=cfg['training_tokens']
    assert train['training_tokens']==schedule['tokens']
    assert train['schedule_sha256']==sha(root/'schedule.json')
    budget_bytes=sum(read(root/f'{role}-targets.json')['cache_bytes'] for role in ['old','new'] if (root/f'{role}-targets.json').exists())
    assert budget_bytes<=cfg['teacher_cache_bytes']
    checks['peak_combined_target_cache_bytes']=budget_bytes
    assert not list(root.glob('*-targets.pt'))
    checks['temporary_target_caches_removed']=True
    teacher_reports=[root/f'{name}.json' for name in ['teacher-dev','teacher-test','bypass-14','bypass-21','bypass-both']]
    ordinary=[root/f'{name}.json' for name in ['before-dev','before-test','historical-dev']]
    if progress['teacher_acquired']:
        ordinary += [root/'students'/arm/f'{split}.json' for arm in ['output','guided','random'] for split in ['dev','test']]
    pids=[]
    for p in ordinary+teacher_reports:
        report=read(p)
        cohort_path=root/'data'/('ablation.json' if p.name.startswith('bypass-') else 'dev.json' if p.stem.endswith('dev') or p.stem=='dev' else 'test.json')
        cohort=read(cohort_path)
        assert report['data_sha256']==sha(cohort_path)
        assert len(report['results'])==len(cohort)
        for expected_item,actual_item in zip(cohort,report['results']):
            assert expected_item['id']==actual_item['id'] and expected_item['skill']==actual_item['skill']
            assert compact(expected_item['answer'])==compact(actual_item['answer'])
        assert report['worker_source_sha256']==frozen['sources']['continual/worker.py']
        assert report['task_source_sha256']==frozen['sources']['continual/tasks.py']
        pids.append(report['fresh_process_pid'])
        for x in report['results']:assert all(x[k]==v for k,v in score(x['response'],x['answer']).items()),(p,x['id'])
        for skill,group in report['groups'].items():
            xs=[x for x in report['results'] if x['skill']==skill];assert group['n']==len(xs)
            for key in ['strict_correct','semantic_correct','valid_json']:
                assert group[key]==sum(x[key] for x in xs)/len(xs)
        if 'general_mc' in report:
            for x in report['general_mc']['results']:assert x['correct']==(x['selected']==x['answer'])
            assert report['general_mc']['accuracy']==sum(x['correct'] for x in report['general_mc']['results'])/32
        assert report['deployed_parameters']==(597100544 if p in teacher_reports else 596049920)
        if p in teacher_reports:assert report['teacher_layers_sha256']==sha(root/'teacher/layers.safetensors')
    assert len(set(pids))==len(pids)
    checks['rescored_reports']=len(ordinary+teacher_reports)
    checks['fresh_evaluation_processes_and_exact_cohort_hashes']=True
    before={x['id']:x for x in read(root/'before-dev.json')['results']}
    both=read(root/'bypass-both.json');assert both['active_extra_parameters']==0
    assert len(both['results'])==128
    for x in both['results']:assert x['response']==before[x['id']]['response']
    checks['bypassing_both_exactly_restores_original_responses']=True
    gate=read(root/'teacher-gate.json');report=read(root/'teacher-dev.json')
    expected=(report['groups']['E']['strict_correct']>=cfg['minimum_acquisition'] and
              report['groups']['E']['strict_correct']-read(root/'before-dev.json')['groups']['E']['strict_correct']>=0.1)
    assert gate['acquired']==expected==progress['teacher_acquired']
    if expected:
        base=read(root/'before-dev.json');history=read(root/'historical-dev.json')
        best={s:max(base['groups'][s]['strict_correct'],history['groups'][s]['strict_correct']) for s in 'ABC'}
        best['D']=base['groups']['D']['strict_correct']
        for arm in ['output','guided','random']:
            out=root/'students'/arm;stats=read(out/'training.json');r=read(out/'dev.json')
            assert stats['schedule_sha256']==sha(root/'schedule.json')
            assert stats['training_tokens']==schedule['tokens'] and stats['sampled_roles']==train['sampled_roles']
            assert stats['trainable_parameters']==10092544 and stats['deployed_parameters']==596049920
            layers=cfg['guided_layers'] if arm=='guided' else cfg['random_layers'] if arm=='random' else []
            assert stats['hint_layers']==layers
            if layers:assert stats['hint_last_8']>0 and stats['hidden_weight']==cfg['hidden_weight']
            reasons=[]
            if r['groups']['E']['strict_correct']<cfg['minimum_acquisition']:reasons.append('Join below acquisition threshold')
            for skill,value in best.items():
                if value-r['groups'][skill]['strict_correct']>cfg['retention_tolerance']+1e-9:reasons.append(skill+' exceeds historical retention tolerance')
            if base['general_mc']['accuracy']-r['general_mc']['accuracy']>cfg['general_tolerance']+1e-9:reasons.append('Fixed general probe exceeds tolerance')
            decision=read(out/'decision.json');assert decision['reasons']==reasons and decision['accepted']==(not reasons)
            with safe_open(next((out/'checkpoint').glob('*.safetensors')),framework='pt',device='cpu') as f:
                assert sum(f.get_tensor(k).numel() for k in f.keys())==596049920
                assert all('temporary_layers' not in k and 'lora_' not in k for k in f.keys())
            assert AutoTokenizer.from_pretrained(out/'checkpoint',local_files_only=True).get_vocab()==tokenizer.get_vocab()
        checks['compact_exports_and_matched_compression_schedules']=True
        checks['development_only_acceptance_recomputed']=True
    else:
        assert not (root/'students').exists()
        checks['failed_acquisition_correctly_stops_compression']=True
    verification=read(root/'verification.json')
    assert verification['unit_tests']['count']==19 and verification['unit_tests']['status']=='passed'
    assert sha(root/'tests.log')==verification['unit_tests']['sha256']
    checks['status']='passed';write(root/'AUDIT.json',checks)
    return checks


def report(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    cfg=read(root/'config.json');progress=read(root/'progress.json')
    reports={'before':read(root/'before-test.json'),'expanded teacher':read(root/'teacher-test.json')}
    if progress['teacher_acquired']:
        reports.update({arm:read(root/'students'/arm/'test.json') for arm in ['output','guided','random']})
    skills=list('ABCDE')+['BA','BC','BD','BE','BDA','BCE']
    summary={'scope':'One synthetic D-to-E boundary; no sustained future-learning test',
             'teacher_acquired':progress['teacher_acquired'],
             'test':{name:{s:accuracy(r['results'],s) for s in skills} for name,r in reports.items()},
             'generation_cap_hits':{name:sum(x['hit_generation_cap'] for x in r['results']) for name,r in reports.items()},
             'paired_against_before':{name:{s:paired(reports['before']['results'],r['results'],s) for s in skills}
                                       for name,r in reports.items() if name!='before'}}
    if progress['teacher_acquired']:
        summary['guided_against_output']={s:paired(reports['output']['results'],reports['guided']['results'],s) for s in skills}
        summary['guided_against_random']={s:paired(reports['random']['results'],reports['guided']['results'],s) for s in skills}
        summary['development_decisions']={arm:read(root/'students'/arm/'decision.json') for arm in ['output','guided','random']}
    ablation={name:read(root/f'{name}.json') for name in ['bypass-14','bypass-21','bypass-both']}
    devteacher=read(root/'teacher-dev.json');ablids={x['id'] for x in read(root/'data/ablation.json')}
    ablation={'both active':{'results':[x for x in devteacher['results'] if x['id'] in ablids]},**ablation}
    summary['ablation_dev']={name:{s:accuracy(r['results'],s) for s in 'CDE'} for name,r in ablation.items()}
    write(root/'summary.json',summary)
    lines=['# Temporary nonlinear-layer screen','',
           'Completed on the RTX 3060 Laptop GPU with direct Hugging Face/PyTorch CUDA access. '
           'The original backbone was frozen during expanded-teacher acquisition; two actual nonlinear residual branches were trained.','',
           '## Acquisition and ablations','',
           f'The expanded teacher {"passed" if progress["teacher_acquired"] else "failed"} the frozen development acquisition gate '
           f'(join {100*devteacher["groups"]["E"]["strict_correct"]:.2f}%, required ≥75% and ≥10 pp gain). '
           'The branch weights contain 1,050,624 parameters; the expanded model contains 597,100,544. '
           'The backbone tensor digest was unchanged by training.','',
           '| Development ablation | Sorting /32 | Grouping /32 | Join /64 |','|---|---:|---:|---:|']
    for name,d in summary['ablation_dev'].items():lines.append('| '+name+' | '+' | '.join(f'{100*d[s]:.2f}%' for s in 'CDE')+' |')
    lines+=['','Bypassing both branches exactly reproduced all 128 original-model responses. '
            'Individual bypass effects are causal interventions on this model, not unique knowledge addresses.','',
            '## Fresh final test','',
            '| Model | Select /64 | Filter /64 | Sort /64 | Group /64 | Join /64 | Fixed general MC /32 |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for name,r in reports.items():lines.append('| '+name+' | '+' | '.join(f'{100*accuracy(r["results"],s):.2f}%' for s in 'ABCDE')+f' | {100*r["general_mc"]["accuracy"]:.2f}% |')
    lines+=['','Atomic and composition test prompts are fingerprint-disjoint from current training, historical archives, prior diagnosis and development prompts. '
            'The general MC probe is fixed and reused; it is not a fresh general-capability benchmark.','',
            '## Compact compression and acceptance','']
    if progress['teacher_acquired']:
        lines+=['Every compact student was exported and reloaded in a fresh ordinary evaluation process with retrieval off. '
                'Each contains 596,049,920 parameters, with no temporary layers, teacher, training adapter or previous context.','',
                '| Arm | Development accepted | Rejection reasons |','|---|---|---|']
        for arm in ['output','guided','random']:
            d=read(root/'students'/arm/'decision.json');lines.append(f'| {arm} | {d["accepted"]} | {"; ".join(d["reasons"]) or "None"} |')
        lines+=['','Acceptance uses only development data, with historical accepted checkpoints re-evaluated on that same cohort. '
                'All three arms share the same frozen sampling schedule, token budget, replay and output targets. '
                'Guided matches decoder-boundary states at layers 2/5/17; random matches 9/21/23 with the same coefficient. '
                'The earlier diagnostic selected broad modules with weak validation; these hidden boundaries are hypotheses, not established safe locations.','',
                '| Fresh test difference | Sort | Group | Join |','|---|---|---|---|']
        for key,label in [('guided_against_output','Guided minus output'),('guided_against_random','Guided minus random')]:
            cells=[]
            for s in 'CDE':
                d=summary[key][s];lo,hi=d['query_bootstrap_95'];cells.append(f'{100*d["mean"]:+.2f} pp [{100*lo:+.2f}, {100*hi:+.2f}]')
            lines.append('| '+label+' | '+' | '.join(cells)+' |')
        lines+=['','Brackets are paired query-bootstrap 95% intervals. They do not cover training seeds or correct for multiple task comparisons. '
                'Final test does not choose an arm, site or coefficient. No candidate is deployed automatically.','']
    else:
        lines+=['Compression was not attempted because the expanded teacher failed the predeclared acquisition gate. '
                'This result tests acquisition by this particular two-branch architecture and budget; it does not test compression feasibility.','']
    lines+=['## Composition transfer','',
            '| Model | BA /32 | BC /32 | BD /32 | BE /32 | BDA /32 | BCE /32 |','|---|---:|---:|---:|---:|---:|---:|']
    for name,r in reports.items():lines.append('| '+name+' | '+' | '.join(f'{100*accuracy(r["results"],s):.2f}%' for s in skills[5:])+' |')
    lines+=['','This boundary receives no composition training. The old backbone has previously seen BA/BC; incoming-join combinations BE/BCE test new transfer. '
            'The development acceptance gate covers atomic skills, not composition retention. In particular, the random-location student '
            'passed that gate while filter-group test accuracy fell from 12/32 to 4/32; output/guided scored 10/32. '
            'Passing the atomic gate therefore does not establish broad retention.','',
            '## Cost and validation','',
            '| Training phase | Trainable parameters | Input tokens | Training seconds | Peak allocated VRAM GiB |',
            '|---|---:|---:|---:|---:|']
    phases={'teacher':root/'teacher/training.json'}
    if progress['teacher_acquired']:phases.update({arm:root/'students'/arm/'training.json' for arm in ['output','guided','random']})
    for name,p in phases.items():
        s=read(p);lines.append(f'| {name} | {s["trainable_parameters"]:,} | {s["training_tokens"]:,} | {s["training_seconds"]:.1f} | {s["peak_cuda_allocated_bytes"]/2**30:.2f} |')
    cache_bytes=sum(read(root/f'{role}-targets.json')['cache_bytes'] for role in ['old','new'] if (root/f'{role}-targets.json').exists())
    lines+=['',f'Combined target cache: {cache_bytes:,} bytes (ceiling {cfg["teacher_cache_bytes"]:,}); tensor caches deleted after the run. '
            'All three compression arms receive the same union cache; output-only ignores its hidden states. '
            'A minimal output-only cache could omit those states, but that storage optimization is not measured here. '
            'Representation losses add training compute at the same token budget. '
            'Retained model checkpoints are research audit artifacts, not bounded online memory.','',
            'The expanded teacher receives replay during acquisition; the earlier LoRA teacher received new examples alone. '
            'Across-run teacher comparisons therefore do not isolate architecture. Within-run compression arms use the same teachers and samples.','',
            '`AUDIT.json` checks source/input/backbone hashes, target execution, held-out fingerprints, every response score, '
            'unchanged-backbone digests, ablations, parameter counts, compact tensor names, identical training schedules, cache bounds and development-only decisions. '
            'Unit tests and CUDA smoke evidence are in `verification.json`.','',
            'The first launch in `screen-001` failed in status-path JSON logging before teacher training. '
            'It was preserved; `screen-002` uses the same scientific inputs with logging serialization fixed.','',
            '## Limits and next decision','',
            'This is one synthetic procedural boundary, one seed and one insertion/bottleneck choice. '
            'It does not establish factual-knowledge assimilation, universal retention or sustained ability to learn subsequent tasks. '
            'Every compact student also receives verified new-task answers directly. A successful student therefore does not establish '
            'that temporary-teacher targets were necessary or better than ordinary supervised learning; a matched no-expanded-teacher '
            'control is required for that claim. '
            'If the teacher succeeds but compact students lose skills, compression is the remaining bottleneck. '
            'If guided does not reliably beat output-only and random controls, do not assume causal localization improves assimilation. '
            'A promising candidate must next survive repeated learning boundaries, later-task acquisition, held-out compositions and bounded-resource accounting.','',
            'Protocol: [NONLINEAR_ASSIMILATION_STEPS.md](../../../NONLINEAR_ASSIMILATION_STEPS.md).','',
            '![Temporary-layer results and ablations](results.png)','']
    teacher_join=accuracy(reports['expanded teacher']['results'],'E')
    before_join=accuracy(reports['before']['results'],'E')
    finding=(f'Temporary branches raised fresh join accuracy from {100*before_join:.2f}% to {100*teacher_join:.2f}% '
             'while the backbone weights stayed unchanged. '
             f'Development join successes were {round(summary["ablation_dev"]["bypass-14"]["E"]*64)}/64 with branch 14 bypassed, '
             f'and {round(summary["ablation_dev"]["bypass-21"]["E"]*64)}/64 with branch 21 bypassed. ')
    if progress['teacher_acquired']:
        accepted=sum(d['accepted'] for d in summary['development_decisions'].values())
        finding+=f'{accepted}/3 compact students passed all frozen development acquisition/retention gates. '
        finding+='Final compact join accuracies were '+', '.join(f'{arm}: {100*accuracy(reports[arm]["results"],"E"):.2f}%'
                     for arm in ['output','guided','random'])+'. '
        finding+='The compact models have no added branches or teacher at evaluation. '
        if not accepted:
            finding+='Passing isolated join acquisition is insufficient when old-skill retention gates fail. '
    finding+=(f'The expanded teacher solved only {round(32*accuracy(reports["expanded teacher"]["results"],"BE"))}/32 filter-join '
              f'and {round(32*accuracy(reports["expanded teacher"]["results"],"BCE"))}/32 filter-sort-join queries; '
              'isolated acquisition did not establish compositional transfer.')
    lines[4:4]=['## Main finding','',finding,'']
    (root/'REPORT.md').write_text('\n'.join(lines))
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),constrained_layout=True)
    x=np.arange(5);width=0.8/len(reports)
    for i,(name,r) in enumerate(reports.items()):
        axes[0].bar(x+(i-(len(reports)-1)/2)*width,[100*accuracy(r['results'],s) for s in 'ABCDE'],width,label=name)
    axes[0].set_xticks(x,['Select','Filter','Sort','Group','Join']);axes[0].set_ylim(0,105);axes[0].set_ylabel('Strict accuracy (%)')
    axes[0].set_title('Fresh final test: 64 queries per atomic task')
    axes[0].legend(fontsize=8,loc='upper center',bbox_to_anchor=(0.5,-0.12),ncol=3,frameon=False)
    names=list(ablation);axes[1].bar(range(len(names)),[100*summary['ablation_dev'][name]['E'] for name in names],color=['#2878a5','#e38b3e','#a787bd','#888888'])
    axes[1].set_xticks(range(len(names)),['Both active','Bypass 14','Bypass 21','Bypass both']);axes[1].set_ylim(0,105)
    axes[1].set_ylabel('Strict join accuracy (%)');axes[1].set_title('Development ablations: 64 join queries')
    fig.savefig(root/'results.png',dpi=160);fig.savefig(root/'results.svg');plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='results/nonlinear_assimilation/screen-002')
    args=p.parse_args();root=(ROOT/args.root).resolve()
    print(json.dumps(audit(root),indent=2),flush=True);report(root)
    write(root/'report-provenance.json',{'generator':str(Path(__file__).relative_to(ROOT)),
          'generator_sha256':sha(Path(__file__)),
          'artifact_sha256':{name:sha(root/name) for name in ['AUDIT.json','REPORT.md','summary.json','results.png','results.svg']}})
