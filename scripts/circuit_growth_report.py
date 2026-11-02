#!/usr/bin/env python3
"""Rescore saved outputs and audit the frozen native-channel placement screen."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from continual.tasks import compact,execute,score,general_mc_examples
from continual.worker import prompt_text,write
from continual.circuit_localization import check as check_local,family_support,select,validate


def read(p):return json.loads(Path(p).read_text())


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def mean(xs):return sum(xs)/len(xs)


def ci(xs):
    rng=random.Random(1701);samples=sorted(mean(rng.choices(xs,k=len(xs))) for _ in range(2000))
    return {'mean':mean(xs),'query_bootstrap_95':[samples[49],samples[1949]],'n':len(xs)}


def paired(a,b,skill):
    aa={x['id']:x['strict_correct'] for x in a['results'] if x['skill']==skill}
    bb={x['id']:x['strict_correct'] for x in b['results'] if x['skill']==skill}
    assert aa.keys()==bb.keys()
    delta=[int(bb[k])-int(aa[k]) for k in aa]
    return {**ci(delta),'gains':delta.count(1),'losses':delta.count(-1)}


def audit_local(root):
    check_local(root);cfg=read(root/'config.json');checks={};fingerprints=set()
    for split in ['discovery','validation']:
        pairs=read(root/'data'/f'{split}.json');rows=read(root/f'{split}-baseline.json')
        assert len(pairs)==4*cfg[split+'_pairs_per_family']
        assert len(rows)==2*len(pairs)
        for pair in pairs:
            fp=hashlib.sha256(compact([pair['clean']['system'],*sorted([pair[s]['prompt'] for s in ['clean','corrupt']])]).encode()).hexdigest()
            assert fp==pair['fingerprint'] and fp not in fingerprints;fingerprints.add(fp)
            assert pair['clean']['answer']==1-pair['corrupt']['answer']
            for side in ['clean','corrupt']:
                item=pair[side];p=item['prompt'];family=pair['family']
                if family=='lookup':
                    data=json.loads(re.search(r'\[.*\]',p).group())
                    key=re.search(r'(?:equals |\w+=|look up \w+ )"([xy])"',p).group(1)
                    key_field=next(k for k,v in data[0].items() if isinstance(v,str))
                    value_field=next(k for k,v in data[0].items() if isinstance(v,int))
                    answer=next(r[value_field] for r in data if r[key_field]==key)
                elif family=='equality':
                    a,b=re.findall(r'"([a-z]+)"',p);answer=int(a==b)
                elif family=='copy':
                    obj=json.loads(re.search(r'\{.*\}',p).group());answer=next(v for v in obj.values() if isinstance(v,int))
                else:
                    answer=int(re.findall(r'\b[01]\b',p)[0])
                assert answer==item['answer']
        expected=[p[s] for p in pairs for s in ['clean','corrupt']]
        for gold,row in zip(expected,rows):
            assert gold['id']==row['id'] and gold['answer']==row['answer']
            assert all(row[k]==v for k,v in score(row['response'],gold['answer']).items())
        sweep=read(root/f'{split}-sweep.json')
        assert sweep['ids']==[p['id'] for p in pairs]
        assert sweep['self_patch_max_error']==0 and sweep['output_copy_max_error']==0
        assert sweep['peak_last_token_cache_bytes']<=cfg['activation_cache_bytes']
        assert all(len(xs)==len(pairs) for direction in ['denoise','noise'] for xs in sweep[direction].values())
        support=family_support(pairs,rows,sweep['first_token_correct'],cfg)
        assert support==read(root/f'{split}-support.json')
        checks[split+'_strict_counts']={f:sum(x['strict_correct'] for x in rows if x['skill']==f) for f in cfg['families']}
    selection=read(root/'selection.json');discovery=read(root/'data/discovery.json')
    regenerated=select(discovery,read(root/'discovery-sweep.json'),read(root/'discovery-attribution.json'),read(root/'discovery-support.json'),cfg)
    assert selection==json.loads(json.dumps(regenerated))
    decision=validate(read(root/'data/validation.json'),read(root/'validation-sweep.json'),read(root/'validation-support.json'),selection,cfg)
    assert decision==read(root/'localization-decision.json') and decision['proceed_to_growth']
    for layer in selection['candidate_layers']:
        a=selection['selected_channels'][str(layer)];b=selection['random_channels'][str(layer)]
        assert len(set(a))==len(set(b))==64 and not set(a)&set(b)
    assert not set(map(tuple,selection['selected_heads']))&set(map(tuple,selection['random_heads']))
    checks.update({'status':'passed','frozen_sources_inputs_backbone':True,'independent_probe_truth_and_output_scoring':True,
                   'discovery_only_selection_reproduced':True,'held_out_gates_recomputed':True,'validated_sites':selection['candidate_layers'],
                   'eligible_related_primitives':selection['eligible_related_families'],'pair_count':len(fingerprints),
                   'scope':'Only last-prompt-position interventions and binary-answer lookup probes. No full skill circuit or SAE feature map.'})
    write(root/'LOCALIZATION_AUDIT.json',checks);return checks


def audit_growth(root):
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    from scripts.circuit_growth_run import check
    torch.set_num_threads(4);check(root);assert read(root/'progress.json')['status']=='completed'
    cfg=read(root/'config.json');frozen=read(root/'frozen-growth.json');checks={}
    payload=read(root/'train-data.json');dev=read(root/'data/dev.json');test=read(root/'data/test.json')
    assert len(dev)==289 and len(test)==513
    assert Counter(x['skill'] for x in dev)==Counter({**{s:32 for s in ['A','B','C','D','E','BA','BC','BD','BDA']},'general':1})
    old=set()
    for parent in [ROOT/'results/assimilation/experiment-001/archive',ROOT/'results/capacity_memory/screen-001/archive',
                   ROOT/'results/causal_assimilation/diagnostic-001/data',ROOT/'results/nonlinear_assimilation/screen-002/data']:
        for p in parent.rglob('*.json'):
            for x in read(p):old.add(hashlib.sha256(x['prompt'].encode()).hexdigest())
    old.update(hashlib.sha256(x['prompt'].encode()).hexdigest() for xs in payload.values() for x in xs)
    seen=set()
    for x in dev+test:
        if x['skill']=='general':continue
        fp=hashlib.sha256(x['prompt'].encode()).hexdigest()
        assert fp==x['fingerprint'] and fp not in old and fp not in seen;seen.add(fp)
        p=json.loads(x['prompt']);assert compact(execute(p['rows'],p['ops']))==compact(x['answer'])
    tok=AutoTokenizer.from_pretrained(ROOT/cfg['before_checkpoint'],local_files_only=True)
    plen=max(len(tok.encode(prompt_text(tok,x),add_special_tokens=False)) for x in dev+test)
    alen=max(len(tok.encode(compact(x['answer'])+tok.eos_token,add_special_tokens=False)) for x in dev+test)
    assert plen<=cfg['max_sequence_tokens'] and alen<=cfg['max_new_tokens']
    replay_tokens=len(tok.encode(compact(payload['old']),add_special_tokens=False))
    assert len(payload['old'])<=cfg['replay_examples'] and replay_tokens<=cfg['replay_tokens']
    checks['cohort_freshness_and_executable_truth']={'fresh_programs':len(seen),'dev':len(dev),'test':len(test),'general_probe_reused':True}
    checks['token_limits']={'max_prompt':plen,'max_gold_answer':alen}
    checks['bounded_replay']={'examples':len(payload['old']),'tokens':replay_tokens}
    schedule=read(root/'schedule.json');assert schedule['tokens']<=cfg['training_tokens']
    roles=dict(Counter(x['role'] for batch in schedule['batches'] for x in batch))
    reports=[root/n for n in ['before-dev.json','before-test.json','historical-dev.json']]
    oldbytes=read(root/'old-targets.json')['cache_bytes'];peakbytes=oldbytes;training=[]
    for name,sites in cfg['growth_arms'].items():
        out=root/'teachers'/name;stats=read(out/'training.json');training.append(stats)
        assert stats['base_parameters']==596049920 and stats['trainable_parameters']==393216 and stats['expanded_parameters']==596443136
        assert stats['backbone_before_sha256']==stats['backbone_after_sha256']
        assert stats['teacher_cache_bytes']==oldbytes
        state=load_file(str(out/'layers.safetensors'))
        assert set(state)=={f'{s}.{p}.weight' for s in sites for p in ['gate','up','down']}
        assert sum(t.numel() for t in state.values())==393216 and all(torch.isfinite(t).all() for t in state.values())
        assert all(state[f'{s}.down.weight'].abs().sum()>0 for s in sites)
        reports += [out/f'{s}.json' for s in ['dev','test','bypass-first','bypass-second','bypass-both']]
        before={x['id']:x['response'] for x in read(root/'before-dev.json')['results']}
        both=read(out/'bypass-both.json');assert both['active_extra_parameters']==0
        assert len(both['results'])==32 and all(x['response']==before[x['id']] for x in both['results'])
        r=read(out/'dev.json');base=read(root/'before-dev.json')
        acquired=r['groups']['E']['strict_correct']>=cfg['minimum_acquisition'] and r['groups']['E']['strict_correct']-base['groups']['E']['strict_correct']>=.1
        assert read(out/'gate.json')['acquired']==acquired
        assert (root/'students'/name/'training.json').exists()==acquired
        if acquired:peakbytes=max(peakbytes,oldbytes+read(root/f'{name}-targets.json')['cache_bytes'])
    for out in sorted((root/'students').iterdir()):
        if not out.is_dir():continue
        stats=read(out/'training.json');training.append(stats)
        assert stats['base_parameters']==stats['deployed_parameters']==596049920 and stats['trainable_parameters']==10092544
        assert stats['hint_layers']==[] and stats['hidden_weight']==0
        expectedbytes=oldbytes if out.name=='direct' else oldbytes+read(root/f'{out.name}-targets.json')['cache_bytes']
        assert stats['teacher_cache_bytes']==expectedbytes
        command=read(out.parent/(out.name+'-train.timing.json'))['command']
        assert ('--new-cache' in command)==(out.name!='direct')
        reports += [out/f'{s}.json' for s in ['dev','test']]
        base=read(root/'before-dev.json');history=read(root/'historical-dev.json');r=read(out/'dev.json')
        best={s:max(base['groups'][s]['strict_correct'],history['groups'][s]['strict_correct']) for s in ['A','B','C','BA','BC']}
        best.update({s:base['groups'][s]['strict_correct'] for s in ['D','BD','BDA']})
        reasons=[]
        if r['groups']['E']['strict_correct']<cfg['minimum_acquisition']:reasons.append('E below acquisition threshold')
        for s,v in best.items():
            if v-r['groups'][s]['strict_correct']>cfg['retention_tolerance']+1e-9:reasons.append(s+' exceeds retention tolerance')
        if base['general_mc']['accuracy']-r['general_mc']['accuracy']>cfg['general_tolerance']+1e-9:reasons.append('General probe exceeds tolerance')
        decision=read(out/'decision.json');assert decision['accepted']==(not reasons) and decision['reasons']==reasons and decision['historical_reference_same_dev']==best
        with safe_open(next((out/'checkpoint').glob('*.safetensors')),framework='pt',device='cpu') as f:
            assert sum(f.get_tensor(k).numel() for k in f.keys())==596049920
            assert all('temporary_layers' not in k and 'lora_' not in k for k in f.keys())
        assert AutoTokenizer.from_pretrained(out/'checkpoint',local_files_only=True).get_vocab()==tok.get_vocab()
    assert peakbytes<=cfg['teacher_cache_bytes'] and not list(root.glob('*.pt'))
    for stats in training:
        assert stats['schedule_sha256']==sha(root/'schedule.json') and stats['training_tokens']==schedule['tokens']
        assert stats['sampled_roles']==roles and stats['replay_tokens']==replay_tokens
        assert stats['gpu']=='NVIDIA GeForce RTX 3060 Laptop GPU'
    pids=[]
    for p in reports:
        report=read(p);is_teacher=p.parent.parent.name=='teachers'
        cohort_path=root/'data'/('ablation.json' if p.stem.startswith('bypass') else 'dev.json' if 'dev' in p.stem else 'test.json')
        cohort=read(cohort_path);assert report['data_sha256']==sha(cohort_path) and len(report['results'])==len(cohort)
        for expected,row in zip(cohort,report['results']):
            assert (expected['id'],expected['skill'],expected['answer'])==(row['id'],row['skill'],row['answer'])
            assert all(row[k]==v for k,v in score(row['response'],row['answer']).items())
        for s,g in report['groups'].items():
            xs=[x for x in report['results'] if x['skill']==s];assert g['n']==len(xs)
            assert all(g[k]==mean([x[k] for x in xs]) for k in ['strict_correct','semantic_correct','valid_json'])
        if 'general_mc' in report:
            mc=report['general_mc'];assert len(mc['results'])==32
            for expected,row in zip(general_mc_examples(),mc['results']):
                assert all(row[k]==expected[k] for k in ['id','question','answer'])
                assert row['selected']==max(row['conditional_label_probabilities'],key=row['conditional_label_probabilities'].get)
            assert all(x['correct']==(x['selected']==x['answer']) for x in mc['results'])
            assert mc['accuracy']==mean([x['correct'] for x in mc['results']])
        assert report['worker_source_sha256']==frozen['sources']['continual/worker.py'] and report['task_source_sha256']==frozen['sources']['continual/tasks.py']
        assert report['deployed_parameters']==(596443136 if is_teacher else 596049920)
        if is_teacher:
            assert report['teacher_layers_sha256']==sha(p.parent/'layers.safetensors')
            expected_active=196608*(2-len(report['bypassed_layers']));assert report['active_extra_parameters']==expected_active
        pids.append(report['fresh_process_pid'])
    assert len(pids)==len(set(pids))
    verification=read(root.parent/'verification.json');assert verification['unit_tests']['count']==23 and verification['unit_tests']['status']=='passed'
    assert sha(root.parent/'tests.log')==verification['unit_tests']['sha256']
    teacher_digests={x['backbone_before_sha256'] for x in training if x['arm']=='teacher'}
    assert len(teacher_digests)==1
    checks.update({'status':'passed','frozen_sources_inputs_and_selection':True,'rescored_reports':len(reports),
                   'matched_training_tokens':schedule['tokens'],'matched_sampled_roles':roles,
                   'teacher_backbones_unchanged':True,'both_bank_bypass_restores_original_responses':True,
                   'compact_exports_original_size_no_added_modules':True,'development_only_atomic_and_composition_gates_recomputed':True,
                   'target_cache_peak_combined_bytes':peakbytes,'cache_ceiling_bytes':cfg['teacher_cache_bytes'],'temporary_target_caches_removed':True,
                   'placement_overlap':frozen['placement_overlap'],'fresh_process_count':len(pids)})
    write(root/'AUDIT.json',checks);return checks


def report(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    local=root.parent;cfg=read(root/'config.json');selection=read(local/'selection.json');sweep=read(local/'validation-sweep.json')
    pairs=read(local/'data/validation.json');eligible=selection['eligible_related_families'];li=[i for i,p in enumerate(pairs) if p['family'] in eligible]
    heads={k:{d:ci([sweep[d][k][i] for i in li]) for d in ['denoise','noise']} for k in sweep['denoise'] if '-head:' in k}
    arms={'before':read(root/'before-test.json')}
    for name in cfg['growth_arms']:arms['teacher-'+name]=read(root/'teachers'/name/'test.json')
    for out in sorted((root/'students').iterdir()):
        if out.is_dir():arms['compact-'+out.name]=read(out/'test.json')
    summary={'localization':read(local/'localization-decision.json'),'attention_head_diagnostics':heads,'placements':cfg['growth_arms'],
             'evaluation_resources':{k:{f:v[f] for f in ['deployed_parameters','evaluation_seconds','peak_allocated_bytes','fresh_process_pid']} for k,v in arms.items()},
             'generation_cap_hits':{k:dict(Counter(x['skill'] for x in v['results'] if x['hit_generation_cap'])) for k,v in arms.items()},
             'test':{k:{**{s:g['strict_correct'] for s,g in v['groups'].items()},'general_mc':v['general_mc']['accuracy']} for k,v in arms.items()},
             'development_acceptance':{p.name:read(p/'decision.json') for p in (root/'students').iterdir() if p.is_dir()},
             'teacher_acquisition':{name:read(root/'teachers'/name/'gate.json') for name in cfg['growth_arms']},
             'bank_bypass_development_E':{name:{label:read(root/'teachers'/name/f'bypass-{label}.json')['groups']['E']['strict_correct'] for label in ['first','second','both']} for name in cfg['growth_arms']},
             'training':{str(p.parent.relative_to(root)):read(p) for p in root.glob('*/**/training.json')},
             'paired_query_differences':{}}
    for a,b in [('teacher-random','teacher-related'),('teacher-fixed','teacher-related'),('compact-direct','compact-related'),('compact-random','compact-related'),('compact-fixed','compact-related')]:
        if a in arms and b in arms:
            summary['paired_query_differences'][b+' minus '+a]={s:paired(arms[a],arms[b],s) for s in ['A','B','C','D','E','BA','BC','BD','BE','BDA','BCE']}
    summary['grouping_error_diagnostics']={}
    prompts={x['id']:x['prompt'] for x in read(root/'data/test.json')}
    for n,r in arms.items():
        kinds=Counter();examples=[]
        for x in r['results']:
            if x['skill']!='D' or x['strict_correct']:continue
            try:
                parsed=json.loads(x['response'])
                strip=lambda rows:[{k:v for k,v in row.items() if k!='total'} for row in rows]
                same_keys=(isinstance(parsed,list) and all(isinstance(y,dict) for y in parsed) and
                           all(set(y)==set(z) for y,z in zip(parsed,x['answer'])) and len(parsed)==len(x['answer']) and
                           strip(parsed)==strip(x['answer']))
                kind='wrong_totals_with_correct_group_keys' if same_keys else 'other_valid_json_error'
            except (ValueError,TypeError):kind='invalid_json'
            kinds[kind]+=1
            if len(examples)<3:examples.append({'id':x['id'],'prompt':prompts[x['id']],'gold':x['answer'],'response':x['response'],'category':kind})
        summary['grouping_error_diagnostics'][n]={'counts':dict(kinds),'first_three_errors':examples}
    stats=[read(p) for p in root.glob('*/**/training.json')]
    summary['resources']={'training_tokens_total':sum(x['training_tokens'] for x in stats),'training_seconds_total':sum(x['training_seconds'] for x in stats),
                          'peak_training_cuda_bytes':max(x['peak_cuda_allocated_bytes'] for x in stats),
                          'worker_wall_seconds':sum(read(p)['seconds'] for p in root.rglob('*.timing.json')),
                          'training_runs':len(stats),'localization':read(local/'localization-resources.json')}
    tr=summary['test'];related=tr['teacher-related']['E'];controls=[tr['teacher-'+x]['E'] for x in ['random','fixed']]
    placement=('did not outperform both placement controls' if related<=max(controls) else 'scored above both placement controls on this one query cohort')
    interpretation=(f'Both selected MLP interfaces passed causal lookup validation, but guided growth {placement}. '
                    f'Teacher join test accuracy: related {related*100:.1f}%, random {controls[0]*100:.1f}%, fixed {controls[1]*100:.1f}%. ')
    if 'compact-related' in tr:
        cr=tr['compact-related'];direct=tr['compact-direct'];accepted=summary['development_acceptance']['related']['accepted']
        interpretation+=(f'The guided compact learner returned to the original parameter count, with join {cr["E"]*100:.1f}% versus direct {direct["E"]*100:.1f}%. '
                         f'Its grouping test accuracy was {cr["D"]*100:.1f}% versus before {tr["before"]["D"]*100:.1f}% and direct {direct["D"]*100:.1f}%; '
                         f'its development retention decision was {"accepted" if accepted else "rejected"}. ')
    interpretation+='Causal contribution of an existing primitive did not by itself establish a superior place to assimilate a new skill. No repeated continual-learning claim follows.'
    summary['interpretation']=interpretation
    write(root/'summary.json',summary)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,4.4),constrained_layout=True)
    categories=['selected','random','shuffled'];colors=['#1976b5','#81909e','#c99532']
    for ax,d in zip(axes,['denoise','noise']):
        for j,category in enumerate(categories):
            vals=[];lower=[];upper=[]
            for layer in selection['candidate_layers']:
                c=ci([sweep[d][f'{category}:{layer}'][i] for i in li]);vals.append(c['mean']*100)
                lower.append((c['mean']-c['query_bootstrap_95'][0])*100);upper.append((c['query_bootstrap_95'][1]-c['mean'])*100)
            ax.bar(np.arange(2)+(j-1)*.24,vals,.23,label=category,color=colors[j],yerr=[lower,upper],capsize=3)
        ax.set_xticks([0,1],[f'MLP {i}' for i in selection['candidate_layers']]);ax.axhline(0,color='#333333',lw=.6)
        ax.set_ylabel('Normalized log-odds gap effect (%)');ax.set_title('Clean → corrupt patch' if d=='denoise' else 'Corrupt → clean patch')
    axes[1].legend(loc='upper right');fig.suptitle('Held-out lookup probes: selected 64 channels vs controls (query bootstrap 95%)')
    for suffix in ['png','svg']:fig.savefig(root/f'localization.{suffix}',dpi=160)
    plt.close(fig)
    names=list(arms);skills=['A','B','C','D','E','BA','BC','BD','BE','BDA','BCE']
    matrix=np.array([[summary['test'][n][s]*100 for s in skills] for n in names])
    fig,ax=plt.subplots(figsize=(12,max(5,len(names)*.55)),constrained_layout=True)
    im=ax.imshow(matrix,vmin=0,vmax=100,cmap='Blues',aspect='auto')
    ax.set_xticks(range(len(skills)),skills);ax.set_yticks(range(len(names)),names)
    for i in range(len(names)):
        for j in range(len(skills)):ax.text(j,i,f'{matrix[i,j]:.0f}',ha='center',va='center',color='white' if matrix[i,j]>55 else '#111111')
    ax.set_title('Fresh program-test accuracy (%): atomic tasks and compositions')
    fig.colorbar(im,ax=ax,label='Strict exact JSON accuracy (%)',shrink=.7)
    for suffix in ['png','svg']:fig.savefig(root/f'comparison.{suffix}',dpi=160)
    plt.close(fig)
    lines=['# Causal lookup localization and native-channel growth','',interpretation,'',
           'One predeclared D-to-E boundary on Qwen3-0.6B, one training seed. Final program-task test prompts were held apart from discovery, causal validation, training and development selection. Fixed general diagnostics are reused. No model was automatically deployed.','',
           '## What was localized','',
           f'Discovery selected zero-based MLPs {selection["candidate_layers"]}. Lookup passed the primitive capability prerequisite; equality did not. Both sites passed all held-out tests in both patch directions: positive related effect and specificity versus copying/constant controls, plus superiority of selected 64 channels over same-layer random channels and shuffled donors. Full-module effects were also positive and specific.','',
           'The intervention is at the final prompt position. Its endpoint gradient attribution and native-channel patches provide evidence about a lookup-related readout path. They do not identify a complete join circuit, uniquely semantic neurons, a safe editable knowledge address, or all relevant attention routing. No sparse autoencoder/transcoder was trained. Attention heads were screened and patched separately; their query-level results are in summary.json. Zero ablation is diagnostic and was not used to rerank candidates.','',
           '![Held-out causal effects](localization.png)','',
           '## Matched placement and direct controls','',
           f'Placement arms: {compact(cfg["growth_arms"])}. Every expanded teacher adds 64 native gated MLP channels at each of two sites: 393,216 new trainable parameters, 596,443,136 total. All old weights remain frozen, with matching before/after digests. The new down projections start at zero. The bank is mathematically equivalent to widening the gated MLP, with separately registered weights. New inputs are randomly initialized; selected old channel indices are evidence for the interface, not an initialization or neuron-neighborhood rule.','',
           'Each successful teacher is compressed using the same output-only method into a separate rank-16 LoRA student, then merged, saved and evaluated in a fresh process at exactly 596,049,920 parameters without added banks or retrieval. Each compact learner trains 10,092,544 adapter parameters. The direct control receives identical verified examples, schedule and old-teacher preservation targets, but no expanded-teacher new-task targets. Each teacher and each compact learner consumes the same token budget separately; temporary growth therefore costs an additional training phase and teacher target caching.','',
           '## Untouched final tests','',
           'A = select/rename; B = filter; C = sort; D = group/sum; E = join. Letters compose left to right. Atomic tasks have 64 examples each; each composition has 32. Strict correctness requires exact executable JSON output.','',
           '| Model | A | B | C | D | E | BA | BC | BD | BE | BDA | BCE | General MC |',
           '|---|'+'---:|'*12]
    for n in names:lines.append('| '+n+' | '+' | '.join(f'{summary["test"][n][s]*100:.1f}' for s in skills+['general_mc'])+' |')
    lines += ['', '![Final test comparison](comparison.png)','', '## Development-only acceptance','',
              'Acquisition requires E ≥75%; old-task and old-composition degradation must be ≤5 percentage points, with the fixed general MC probe also within 5 points. References for A/B/C/BA/BC are the better of the before-D checkpoint and historical-C checkpoint on the same development cohort; D/BD/BDA use the before-D checkpoint. The old model already has weak composition performance; retention is measured against that existing behavior, not a claim of strong composition capability.','']
    for n,d in summary['development_acceptance'].items():lines.append(f'- {n}: '+('accepted' if d['accepted'] else 'rejected: '+', '.join(d['reasons'])))
    if 'compact-related' in summary['grouping_error_diagnostics']:
        errors=summary['grouping_error_diagnostics']['compact-related'];count=errors['counts'].get('wrong_totals_with_correct_group_keys',0)
        lines += ['',f'In the guided compact learner, {count} grouping failures preserved the correct group keys and output shape but produced wrong totals. Saved counterexamples and exact diagnostic counts for every arm are in summary.json. These are behavioral error categories, not an identified internal cause.']
    lines += ['', 'All candidates remain research artifacts, including rejected models. Test results do not change the predeclared development decision. The historical sorting gate can reject a learner that inherits an existing deficit: compare before-D and learner scores before calling a rejection new forgetting.','',
              '## Added-bank causal controls','',
              '| Placement | Full teacher dev E | Bypass first site | Bypass second site | Bypass both |',
              '|---|---:|---:|---:|---:|']
    for n in cfg['growth_arms']:
        bypass=summary['bank_bypass_development_E'][n]
        lines.append('| '+n+' | '+f'{summary["teacher_acquisition"][n]["join"]*100:.1f}'+' | '+' | '.join(f'{bypass[k]*100:.1f}' for k in ['first','second','both'])+' |')
    lines += ['', 'These bypasses change only the added banks. Both-bank bypass reproduces original responses exactly; individual bypasses test contribution of each learned bank without identifying unique skill storage.','',
              '## Evidence, cost and limits','',
              f'Each phase used {read(root/"schedule.json")["tokens"]:,} actual input tokens against a 98,304 ceiling. Replay is bounded to 115 examples / 15,762 tokenizer tokens. Localization used 224 counterfactual pairs; development has 289 prompts and final test has 513, including one reused general generation prompt. The 32-item fixed general MC diagnostic is narrow and reused, not a general benchmark. Target caches have a combined 16 MiB ceiling and are deleted after training. See AUDIT.json for observed bytes and token limits, and summary.json for per-phase resources and paired comparisons.','',
              'Every teacher has separate first-bank, second-bank and both-bank bypass tests. Both-bank bypass must reproduce all 32 original development E responses exactly. Each compact checkpoint is independently counted and reloaded. Saved outputs are rescored; sources, examples, schedule, selection and original backbone are hash checked.','',
              'The original model reached the shared 192-token generation cap on three filter-join and two filter-sort-join test prompts; trained candidates had no test cap hits. The longest gold test/development answer is 55 tokens. The common generation cap was not changed after seeing results.','',
              'Paired bootstrap intervals in summary.json quantify variation over this small query cohort only. They do not measure variation over training seeds, task orders or localization choices; comparisons are exploratory and not corrected for multiplicity. New-task teacher targets are enabled only when teacher argmax matches every gold answer token under teacher forcing, as in the existing compression recipe; verified-answer cross entropy is still supplied to every compact learner.','',
              'A placement advantage must appear against both random and fixed placement. A temporary-teacher benefit must appear against the direct compact learner after accounting for extra training compute. Successful acquisition or compact export alone is insufficient: repeated task streams must preserve old atomics and compositions and sustain later learning under bounded resources. This experiment cannot establish continual learning or safe deletion of knowledge.','',
              'Artifacts: [audit](AUDIT.json), [summary](summary.json), [frozen growth manifest](frozen-growth.json), [localization audit](../LOCALIZATION_AUDIT.json), [protocol](../../../../CIRCUIT_GUIDED_GROWTH.md).']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    write(root/'report-provenance.json',{'report_source_sha256':sha(Path(__file__)),
          'artifacts':{str(p.relative_to(root)):sha(p) for p in [root/'REPORT.md',root/'AUDIT.json',root/'summary.json',root/'comparison.png',root/'localization.png']}})
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='results/circuit_growth/screen-001/growth');p.add_argument('--local-only',action='store_true');args=p.parse_args();root=(ROOT/args.root).resolve()
    print(json.dumps(audit_local(root.parent),indent=2))
    if not args.local_only:
        print(json.dumps(audit_growth(root),indent=2));report(root)
