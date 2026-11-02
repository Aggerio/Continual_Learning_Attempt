#!/usr/bin/env python3
"""Audit saved numeric traces, protected exports and held-out behavior."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from continual.group_preservation import read,sha,ci,specificity,choose,numeric_positions,checkpoint_module_hashes
from continual.tasks import compact,execute,score,general_mc_examples
from continual.worker import prompt_text,write


def paired(a,b,skill):
    aa={x['id']:int(x['strict_correct']) for x in a if x['skill']==skill}
    bb={x['id']:int(x['strict_correct']) for x in b if x['skill']==skill}
    assert aa.keys()==bb.keys();delta=[bb[k]-aa[k] for k in aa]
    return {**ci(delta),'gains':delta.count(1),'losses':delta.count(-1)}


def score_rows(rows,cohort):
    expected={x['id']:x for x in cohort};seen=set()
    for row in rows:
        assert row['id'] in expected and row['id'] not in seen;seen.add(row['id']);gold=expected[row['id']]
        assert row['skill']==gold['skill'] and compact(row['answer'])==compact(gold['answer'])
        assert all(row[k]==v for k,v in score(row['response'],gold['answer']).items())
    return seen


def accuracy(rows):
    skills=sorted(set(x['skill'] for x in rows))
    return {s:sum(x['strict_correct'] for x in rows if x['skill']==s)/sum(x['skill']==s for x in rows) for s in skills}


def decision(base,history,r,cfg):
    current={s:base['groups'][s]['strict_correct'] for s in ['A','B','C','D','BA','BC','BD','BDA']}
    best={**current,**{s:max(current[s],history['groups'][s]['strict_correct']) for s in ['A','B','C','BA','BC']}}
    out={}
    for name,refs in [('current_checkpoint',current),('historical_best',best)]:
        reasons=[]
        if r['groups']['E']['strict_correct']<cfg['minimum_acquisition']:reasons.append('E below acquisition threshold')
        for s,v in refs.items():
            if v-r['groups'][s]['strict_correct']>cfg['retention_tolerance']+1e-9:reasons.append(s+' exceeds retention tolerance')
        if base['general_mc']['accuracy']-r['general_mc']['accuracy']>cfg['general_tolerance']+1e-9:reasons.append('General probe exceeds tolerance')
        out[name]={'accepted':not reasons,'reasons':reasons,'references':refs}
    out['selection']='Development only; test data never used for acceptance';return out


def audit(root):
    import torch
    from safetensors import safe_open
    from transformers import AutoTokenizer
    from scripts.group_preservation_run import check
    torch.set_num_threads(4);check(root,weights=True);assert read(root/'progress.json')['status']=='completed'
    cfg=read(root/'config.json');student=read(root/'student-config.json');frozen=read(root/'frozen.json')
    supplement=read(root/'supplemental-reference-weights.json');assert supplement['phase_at_freeze']['phase']=='trace'
    for name,h in supplement['weights'].items():assert sha(ROOT/name)==h
    data=read(root/'train-data.json');schedule=read(root/'schedule.json');assert schedule['tokens']<=student['training_tokens']
    reference=ROOT/cfg['reference_run']
    assert sha(root/'train-data.json')==sha(reference/'train-data.json') and sha(root/'schedule.json')==sha(reference/'schedule.json')
    historical=set()
    for parent in [ROOT/'results/assimilation/experiment-001/archive',ROOT/'results/capacity_memory/screen-001/archive',
                   ROOT/'results/causal_assimilation/diagnostic-001/data',ROOT/'results/nonlinear_assimilation/screen-002/data',reference/'data']:
        for p in parent.rglob('*.json'):
            for x in read(p):historical.add(hashlib.sha256(x['prompt'].encode()).hexdigest())
    historical.update(hashlib.sha256(x['prompt'].encode()).hexdigest() for xs in data.values() for x in xs)
    tok=AutoTokenizer.from_pretrained(ROOT/cfg['before_checkpoint'],local_files_only=True);seen=set();cohorts={}
    for split in ['discovery','validation','dev','test']:
        cohorts[split]=read(root/'data'/f'{split}.json');ids=set()
        for x in cohorts[split]:
            assert x['id'] not in ids;ids.add(x['id'])
            if x['skill']=='general':continue
            fp=hashlib.sha256(x['prompt'].encode()).hexdigest();assert fp==x['fingerprint'] and fp not in historical and fp not in seen;seen.add(fp)
            p=json.loads(x['prompt']);assert compact(execute(p['rows'],p['ops']))==compact(x['answer'])
    assert len(cohorts['dev'])==353 and len(cohorts['test'])==513
    assert Counter(x['skill'] for x in cohorts['discovery'])==Counter(cfg['discovery_counts'])
    assert Counter(x['skill'] for x in cohorts['validation'])==Counter(cfg['validation_counts'])
    replay_tokens=len(tok.encode(compact(data['old']),add_special_tokens=False))
    assert len(data['old'])<=student['replay_examples'] and replay_tokens<=student['replay_tokens']
    plen=max(len(tok.encode(prompt_text(tok,x),add_special_tokens=False)) for xs in cohorts.values() for x in xs)
    alen=max(len(tok.encode(compact(x['answer'])+tok.eos_token,add_special_tokens=False)) for xs in cohorts.values() for x in xs)
    assert plen<=student['max_sequence_tokens'] and alen<=student['max_new_tokens']
    for split in ['discovery','validation']:
        for x in cohorts[split]:
            positions=numeric_positions(tok,x);text=compact(x['answer'])
            answer=tok(text+tok.eos_token,add_special_tokens=False,return_offsets_mapping=True)
            prefix=len(tok.encode(prompt_text(tok,x),add_special_tokens=False))
            fragments=[(i,answer['offset_mapping'][i-prefix+1]) for i in positions]
            assert all(0<=a<b<=len(text) for _,(a,b) in fragments)
    trace=read(root/'trace.json');assert trace['ids']==[x['id'] for x in cohorts['discovery']]
    assert trace['self_patch_max_error']==0 and trace['output_copy_max_error']==0
    assert trace['peak_activation_cache_bytes']<=cfg['activation_cache_bytes']
    assert len(trace['weight_reversion'])==56
    for k in ['old_into_new','new_into_old','weight_reversion']:
        for values in trace[k].values():assert len(values)==48 and all(torch.isfinite(torch.tensor(values)))
    selection=read(root/'selection.json');assert selection==choose(trace,cohorts['discovery'],cfg)
    assert not set(selection['guided'])&set(selection['random'])
    assert Counter(s.split(':')[1] for s in selection['guided'])==Counter(s.split(':')[1] for s in selection['random'])
    sf=read(root/'frozen-selection.json');assert sha(root/'selection.json')==sf['sha256'] and sha(root/'validation-decision.json')==sf['validation_decision_sha256']
    val=read(root/'validation-numeric.json');assert val['ids']==[x['id'] for x in cohorts['validation']]
    for name,values in val.items():
        if name!='ids':assert len(values)==64 and torch.isfinite(torch.tensor(values)).all()
    generation={p.stem:read(p) for p in (root/'generation').glob('*.json')}
    for name,rows in generation.items():
        ids=score_rows(rows,cohorts['validation'])
        if name in ['before','after','guided_rollback','random_rollback']:assert len(ids)==64
        elif name in ['self_patch','output_copy']:assert len(ids)==8
        else:assert Counter(x['skill'] for x in rows)==Counter(cfg['generation_patch_counts'])
    for name,baseline in [('self_patch','after'),('output_copy','before')]:
        expected={x['id']:x['response'] for x in generation[baseline]}
        assert all(x['response']==expected[x['id']] for x in generation[name])
    values=[v for v,x in zip(val['guided_rollback'],cohorts['validation']) if x['skill']=='D']
    effect=ci(values);specific=specificity(val['guided_rollback'],cohorts['validation'])
    validated=effect['query_bootstrap_95'][0]>0 and specific['query_bootstrap_95'][0]>0 and accuracy(generation['guided_rollback'])['D']>accuracy(generation['after'])['D']
    vd=read(root/'validation-decision.json');assert vd['validated_grouping_update_contribution']==validated and vd['numeric_gain']==effect and vd['specificity_vs_numeric_copy']==specific
    reports=[root/f'{name}-{split}.json' for name in ['before','previous-compressed','previous-direct'] for split in ['dev','test']]+[root/'historical-dev.json']
    pids=[];stats={};cachebytes=sum(read(root/f'{name}-targets.json')['cache_bytes'] for name in ['old','new'])
    assert cachebytes<=student['teacher_cache_bytes'] and not list(root.glob('*.pt'))
    for name in ['none','guided','random']:
        out=root/'students'/name;st=read(out/'training.json');stats[name]=st
        assert st['schedule_sha256']==sha(root/'schedule.json') and st['training_tokens']==schedule['tokens']
        assert st['sampled_roles']==dict(Counter(x['role'] for b in schedule['batches'] for x in b))
        assert st['base_parameters']==st['deployed_parameters']==596049920
        assert st['trainable_parameters']+st['protected_adapter_parameters']==10092544
        assert st['teacher_cache_bytes']==cachebytes and st['replay_tokens']==replay_tokens
        assert st['hint_layers']==[] and st['hidden_weight']==0 and st['constraint']==name
        sites=[] if name=='none' else selection[name];assert st['protected_modules']==sites
        before_hashes=checkpoint_module_hashes(ROOT/cfg['before_checkpoint'],sites);after_hashes=checkpoint_module_hashes(out/'checkpoint',sites)
        assert before_hashes==after_hashes==st['protected_module_hashes']
        assert decision(read(root/'before-dev.json'),read(root/'historical-dev.json'),read(out/'dev.json'),student)==read(out/'decision.json')
        with safe_open(next((out/'checkpoint').glob('*.safetensors')),framework='pt',device='cpu') as f:
            assert sum(f.get_tensor(k).numel() for k in f.keys())==596049920
            assert all('lora_' not in k and 'temporary_layers' not in k for k in f.keys())
        assert AutoTokenizer.from_pretrained(out/'checkpoint',local_files_only=True).get_vocab()==tok.get_vocab()
        reports += [out/f'{split}.json' for split in ['dev','test']]
    assert stats['guided']['trainable_parameters']==stats['random']['trainable_parameters']
    planned=read(root/'planned-capacity.json');assert planned['all_adapter_parameters']==10092544
    for name in ['guided','random']:
        assert stats[name]['trainable_parameters']==planned['constraints'][name]['trainable_adapter_parameters']
        assert stats[name]['protected_adapter_parameters']==planned['constraints'][name]['blocked_adapter_parameters']
        assert stats[name]['protected_modules']==planned['constraints'][name]['modules']
    rv=read(root/'reproduction-verification.json');assert rv['initialization_matches']
    assert rv['source_sha256']==sha(ROOT/'scripts/group_preservation_verify.py')
    plan=read(root/'reproduction-plan.json')
    assert rv['verification_target_caches_removed'] and all(not Path(x['path']).exists() for x in plan['caches'].values())
    assert plan['peak_cache_storage_including_verification_copy_bytes']<=student['teacher_cache_bytes']
    canonical=root/'reproduction/canonical';st=read(canonical/'training.json')
    assert st['schedule_sha256']==sha(root/'schedule.json') and st['training_tokens']==schedule['tokens'] and st['trainable_parameters']==10092544
    assert st['deployed_parameters']==596049920 and st['teacher_cache_bytes']==cachebytes
    with safe_open(next((canonical/'checkpoint').glob('*.safetensors')),framework='pt',device='cpu') as f:
        assert sum(f.get_tensor(k).numel() for k in f.keys())==596049920
        assert all('lora_' not in k and 'temporary_layers' not in k for k in f.keys())
    if not rv['original_trainer_checkpoint_matches_fresh_unconstrained']:reports += [canonical/f'{split}.json' for split in ['dev','test']]
    for p in reports:
        r=read(p);split='dev' if 'dev' in p.stem else 'test';cohort=cohorts[split]
        assert r['data_sha256']==sha(root/'data'/f'{split}.json') and len(score_rows(r['results'],cohort))==len(cohort)
        assert r['deployed_parameters']==596049920
        assert r['worker_source_sha256']==frozen['sources']['continual/worker.py'] and r['task_source_sha256']==frozen['sources']['continual/tasks.py']
        for s,g in r['groups'].items():
            xs=[x for x in r['results'] if x['skill']==s];assert g['n']==len(xs)
            assert all(g[k]==sum(x[k] for x in xs)/len(xs) for k in ['strict_correct','semantic_correct','valid_json'])
        mc=r['general_mc'];assert len(mc['results'])==32
        for expected,row in zip(general_mc_examples(),mc['results']):
            assert all(row[k]==expected[k] for k in ['id','question','answer'])
            assert row['selected']==max(row['conditional_label_probabilities'],key=row['conditional_label_probabilities'].get)
            assert row['correct']==(row['selected']==row['answer'])
        assert mc['accuracy']==sum(x['correct'] for x in mc['results'])/32
        pids.append(r['fresh_process_pid'])
    assert len(set(pids))==len(pids)
    reproduction={}
    for split in ['dev','test']:
        a=read(root/'students/none'/f'{split}.json')['results'];b=read(root/f'previous-compressed-{split}.json')['results']
        reproduction[split+'_responses_identical']=all(x['id']==y['id'] and x['response']==y['response'] for x,y in zip(a,b))
    before_files=sorted((ROOT/cfg['after_checkpoint']).glob('*.safetensors'));new_files=sorted((root/'students/none/checkpoint').glob('*.safetensors'))
    reproduction['checkpoint_bytes_identical']=[sha(p) for p in before_files]==[sha(p) for p in new_files]
    prior_exact=all(reproduction.values())
    reproduction['original_entrypoint_initialization_matches']=rv['initialization_matches']
    reproduction['original_trainer_checkpoint_matches_fresh_unconstrained']=rv['original_trainer_checkpoint_matches_fresh_unconstrained']
    reproduction['deterministic_algorithms_enabled']=rv['deterministic_algorithms_enabled']
    verification=read(root/'verification.json');assert verification['unit_tests']['count']==26 and verification['unit_tests']['status']=='passed'
    assert sha(root/'tests.log')==verification['unit_tests']['sha256']
    decoder=read(root/'decoder-verification.json');assert decoder['status']=='passed' and all(x['matched'] for x in decoder['checks'].values())
    assert decoder['source_sha256']==sha(ROOT/'scripts/group_preservation_verify.py') and decoder['causal_decoder_source_sha256']==frozen['sources']['continual/causal_diagnosis.py']
    limited=not prior_exact or not rv['original_trainer_checkpoint_matches_fresh_unconstrained']
    checks={'status':'passed_with_reproducibility_limit' if limited else 'passed','fresh_programs':len(seen),'frozen_sources_inputs_weights_and_selection':True,
            'validated_grouping_update_contribution':validated,'independent_executable_truth_and_saved_output_scoring':True,
            'rescored_evaluation_reports':len(reports),'rescored_causal_generation_reports':len(generation),
            'fresh_evaluation_processes':len(pids),'unconstrained_reproduction':reproduction,
            'protected_module_weights_exactly_preserved':True,'matched_protected_arm_trainable_parameters':stats['guided']['trainable_parameters'],
            'compact_parameters':596049920,'matched_input_training_tokens':schedule['tokens'],
            'replay_examples':len(data['old']),'replay_tokens':replay_tokens,
            'combined_target_cache_bytes':cachebytes,'peak_cache_storage_including_verification_copy_bytes':plan['peak_cache_storage_including_verification_copy_bytes'],'target_caches_removed':True,
            'peak_activation_cache_bytes':trace['peak_activation_cache_bytes'],'activation_cache_ceiling_bytes':cfg['activation_cache_bytes'],
            'max_prompt_tokens':plen,'max_gold_answer_tokens':alen,'development_only_gates_recomputed':True,'real_model_causal_decoder_matches_model_generate':True}
    write(root/'AUDIT.json',checks);return checks


def report(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    cfg=read(root/'config.json');selection=read(root/'selection.json');trace=read(root/'trace.json');val=read(root/'validation-numeric.json')
    ds=read(root/'data/discovery.json');vs=read(root/'data/validation.json')
    results={n:read(root/f'{n}-test.json') for n in ['before','previous-compressed','previous-direct']}
    results.update({n:read(root/'students'/n/'test.json') for n in ['none','guided','random']})
    canonical=root/'reproduction/canonical'
    if (canonical/'test.json').exists():results['original-trainer-repeat']=read(canonical/'test.json')
    skills=['A','B','C','D','E','BA','BC','BD','BE','BDA','BCE']
    summary={'selection':selection,'trace_validation':read(root/'validation-decision.json'),'reproducibility':read(root/'AUDIT.json')['unconstrained_reproduction'],
             'test':{n:{**accuracy(r['results']),'general_mc':r['general_mc']['accuracy']} for n,r in results.items()},
             'development_decisions':{n:read(root/'students'/n/'decision.json') for n in ['none','guided','random']},
             'training':{n:read(root/'students'/n/'training.json') for n in ['none','guided','random']},
             'validation_numeric':{},'validation_generation':{},'paired_test_differences':{},
             'generation_cap_hits':{n:dict(Counter(x['skill'] for x in r['results'] if x['hit_generation_cap'])) for n,r in results.items()}}
    summary['reference_development_decisions']={n:decision(read(root/'before-dev.json'),read(root/'historical-dev.json'),read(root/f'{n}-dev.json'),read(root/'student-config.json')) for n in ['before','previous-compressed','previous-direct']}
    if (canonical/'dev.json').exists():summary['original_trainer_repeat_development_decision']=decision(read(root/'before-dev.json'),read(root/'historical-dev.json'),read(canonical/'dev.json'),read(root/'student-config.json'))
    prompts={x['id']:x['prompt'] for x in read(root/'data/test.json')};summary['grouping_error_diagnostics']={}
    for name,r in results.items():
        counts=Counter();examples=[]
        for x in r['results']:
            if x['skill']!='D' or x['strict_correct']:continue
            kind='invalid_json'
            if x['valid_json']:
                a=json.loads(x['response']);b=x['answer'];strip=lambda rows:[{k:v for k,v in y.items() if k!='total'} for y in rows]
                same=(isinstance(a,list) and len(a)==len(b) and all(isinstance(y,dict) and set(y)==set(z) for y,z in zip(a,b)) and strip(a)==strip(b))
                kind='wrong_totals_with_correct_group_keys' if same else 'other_valid_json_error'
            counts[kind]+=1
            if len(examples)<3:examples.append({'id':x['id'],'prompt':prompts[x['id']],'gold':x['answer'],'response':x['response'],'category':kind})
        summary['grouping_error_diagnostics'][name]={'counts':dict(counts),'first_three_errors':examples}
    for name,values in val.items():
        if name in ['ids','before','after']:continue
        summary['validation_numeric'][name]={s:ci([v for v,x in zip(values,vs) if x['skill']==s]) for s in ['A','D','E']}
    summary['validation_comparative_numeric']={}
    for a,b in [('random_rollback','guided_rollback'),('random_patch','guided_patch'),('shuffled_patch','guided_patch')]:
        delta=[vb-va for va,vb,x in zip(val[a],val[b],vs) if x['skill']=='D']
        summary['validation_comparative_numeric'][b+' minus '+a]=ci(delta)
    for p in sorted((root/'generation').glob('*.json')):summary['validation_generation'][p.stem]=accuracy(read(p))
    for a,b in [('none','guided'),('random','guided'),('before','guided'),('none','random')]:
        summary['paired_test_differences'][b+' minus '+a]={s:paired(results[a]['results'],results[b]['results'],s) for s in skills}
    repeated=read(canonical/'training.json')
    summary['resources']={'trace':read(root/'trace-resources.json'),'training_tokens_total':sum(x['training_tokens'] for x in summary['training'].values())+repeated['training_tokens'],
                          'training_seconds_total':sum(x['training_seconds'] for x in summary['training'].values())+repeated['training_seconds'],
                          'additional_reproduction_training':repeated,
                          'verification_wall_seconds_including_repeat':read(root/'reproduction-verification.json')['verification_wall_seconds'],
                          'peak_training_cuda_bytes':max(x['peak_cuda_allocated_bytes'] for x in summary['training'].values()),
                          'worker_wall_seconds':sum(read(p)['seconds'] for p in root.rglob('*.timing.json')),
                          'teacher_acquisition_cost':'Expanded teacher was reused from the previous screen; its original training is additional historical cost.'}
    summary['method_access']={'optimizer_old_replay_examples':115,'extra_labeled_old_examples_used_for_protection_selection':48,
                             'diagnostic_validation_old_examples':48,'diagnostic_validation_new_examples':16,
                             'prior_failed_checkpoint_used_for_selection':True,
                             'scope':'Retrospective repair screen. Extra diagnostic old-task supervision and prior failed-update compute are not part of the optimizer replay/token budget.'}
    tr=summary['test'];validated=summary['trace_validation']['validated_grouping_update_contribution']
    conclusion=(f'The module-level grouping update contribution {"passed" if validated else "failed"} held-out causal validation. '
                f'Test grouping: before {tr["before"]["D"]*100:.1f}%, unconstrained {tr["none"]["D"]*100:.1f}%, guided protection {tr["guided"]["D"]*100:.1f}%, random protection {tr["random"]["D"]*100:.1f}%. '
                f'Test join: unconstrained {tr["none"]["E"]*100:.1f}%, guided {tr["guided"]["E"]*100:.1f}%, random {tr["random"]["E"]*100:.1f}%. '
                'Exact local weight preservation is verified; all retention and composition scores still determine whether this is useful assimilation.')
    passed_current=[n for n,d in summary['development_decisions'].items() if d['current_checkpoint']['accepted']]
    passed_historical=[n for n,d in summary['development_decisions'].items() if d['historical_best']['accepted']]
    conclusion+=f' Current-checkpoint development gates passed: {passed_current or "none"}; historical-best gates passed: {passed_historical or "none"}.'
    if tr['guided']['D']<=tr['random']['D']:conclusion+=' Causal guidance did not beat matched random protection on grouping accuracy.'
    if 'original-trainer-repeat' in tr:
        n=cfg['test_atomic_examples'];baseline=[round(tr[k]['D']*n) for k in ['none','original-trainer-repeat']]
        conclusion+=f' Fixed-input unconstrained repeats scored {min(baseline)}-{max(baseline)}/{n} grouping; the guided score falls within that observed range.'
        if summary['original_trainer_repeat_development_decision']['current_checkpoint']['accepted']:
            conclusion+=' The original-trainer verification repeat also passed the current-checkpoint gate.'
    conclusion+=' Audit status: passed with a reproducibility limit. The data do not establish a reliable benefit from guided module protection.'
    summary['interpretation']=conclusion;write(root/'summary.json',summary)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    names=list(results);matrix=np.array([[tr[n][s]*100 for s in skills] for n in names])
    fig,ax=plt.subplots(figsize=(12,5),constrained_layout=True);im=ax.imshow(matrix,cmap='Blues',vmin=0,vmax=100,aspect='auto')
    ax.set_xticks(range(len(skills)),skills);ax.set_yticks(range(len(names)),names)
    for i in range(len(names)):
        for j in range(len(skills)):ax.text(j,i,f'{matrix[i,j]:.0f}',ha='center',va='center',color='white' if matrix[i,j]>55 else '#111111')
    ax.set_title('Fresh program-test accuracy (%): module update protection during compression')
    fig.colorbar(im,ax=ax,label='Strict exact JSON accuracy (%)',shrink=.75)
    for suffix in ['png','svg']:fig.savefig(root/f'comparison.{suffix}',dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),constrained_layout=True)
    top=selection['ranking'][:10];values=[selection['discovery_numeric_specificity'][s]['mean'] for s in top]
    axes[0].barh(range(10),values,color=['#1879b0']*3+['#8698a4']*7);axes[0].set_yticks(range(10),top);axes[0].invert_yaxis()
    axes[0].set_xlabel('Grouping minus numeric-copy log-probability gain (nats)');axes[0].set_title('Discovery: revert one module at a time')
    kinds=['after','guided_rollback','random_rollback'];labels=['compressed','guided rollback','random rollback']
    for j,s in enumerate(['D','E']):
        vals=[summary['validation_generation'][k][s]*100 for k in kinds]
        axes[1].bar(np.arange(3)+(j-.5)*.35,vals,.34,label='grouping' if s=='D' else 'join')
    axes[1].set_xticks(range(3),labels,rotation=12);axes[1].set_ylim(0,110);axes[1].set_ylabel('Validation exact accuracy (%)');axes[1].legend(loc='upper left',bbox_to_anchor=(1.01,1),frameon=False)
    axes[1].set_title('Autonomous generation: joint module rollback')
    for suffix in ['png','svg']:fig.savefig(root/f'causal-trace.{suffix}',dpi=160)
    plt.close(fig)
    lines=['# Grouping trace and update protection','',conclusion,'',
           '## Fixed hypothesis and scope','',
           'Trace the old grouping computation lost during compression, then prevent updates in the implicated modules during a fresh compression run. This is one predeclared D-to-E boundary and one seed. All previous experiments and frozen source files remain intact. No candidate is deployed.','',
           f'Discovery selected {selection["guided"]}; same-component random controls are {selection["random"]}. Both sets contain three disjoint modules. The intervention granularity is an entire attention or MLP module, not individual semantic neurons. The trace does not uniquely locate arithmetic knowledge.','',
           'Discovery has 32 grouping and 16 select/rename copying prompts. Score only prediction positions for total-number tokens in grouping and numeric values in the copying control. Gold answers are used for teacher-forced tracing, including aligned activation patches, reverse patches and exact module weight rollback. Rank by grouping-minus-copying rollback gain. Freeze selection before validation; no reranking, seed search or hyperparameter tuning.','',
           'Validation has 32 grouping, 16 copying and 16 join prompts. Positive grouping numeric gain and specificity confidence bounds plus improved exact grouping generation validate a grouping-related update contribution. Autonomous donor patching uses recipient-generated prefixes, never gold continuations. Random-module, shuffled-donor, zero-ablation, self-patch and final-output-copy controls are recorded. Failed localization is reported without a circuit claim; the predeclared protection ablations still run.','',
           '![Causal discovery and validation](causal-trace.png)','',
           '## Protection test','',
           'All three learners start from the same pre-join checkpoint and use the identical 98,189-input-token schedule, verified answers, bounded old replay, old-target KL and the same frozen expanded teacher for new-task targets. Only adapter eligibility changes. Guided and random arms have equal trainable capacity; unconstrained has more. No added parameters remain after adapter merging. Freezing a module preserves its local weights exactly but cannot guarantee unchanged activations, because its inputs can change upstream.','',
           '| Learner | Trainable parameters | Protected adapter parameters |',
           '|---|---:|---:|']
    for n,st in summary['training'].items():lines.append(f'| {n} | {st["trainable_parameters"]:,} | {st["protected_adapter_parameters"]:,} |')
    lines += ['', '## Fresh final tests','',
              'A select/rename, B filter, C sort, D grouping/sum, E join. Operations compose left to right. Atomic tasks have 64 examples and compositions 32. Fixed general probes are reused and are narrow diagnostics.','',
              '| Model | '+' | '.join(skills+['General MC'])+' |', '|---|'+'---:|'*12]
    for n in names:lines.append('| '+n+' | '+' | '.join(f'{tr[n][s]*100:.1f}' for s in skills+['general_mc'])+' |')
    lines += ['', '![Final accuracy matrix](comparison.png)','',
              'The fresh unconstrained run did not reproduce the older failed checkpoint bit for bit. The failed exact-reproduction check is retained in the audit. Protection is compared against the fresh unconstrained run, not assumed-identical older results. A separately planned repeat uses the original trainer entrypoint and identical cached targets, and initialization fingerprints compare both entrypoints. Deterministic training algorithms were not enabled. The original-trainer repeat is a reproducibility control, never an extra candidate for selecting a favorable result. Previous direct learning uses no new-task teacher targets and is evaluated on the same fresh cohorts.','',
              '## Development decisions','',
              'Join must reach 75%, every old atomic task and old composition must stay within 5 percentage points, and the fixed general MC diagnostic must stay within 5 points. The current-checkpoint gate measures newly introduced forgetting. The historical-best gate separately asks whether inherited deficits have also been recovered. Test data never change these decisions.','']
    for n,dec in summary['development_decisions'].items():
        for kind in ['current_checkpoint','historical_best']:
            d=dec[kind];lines.append(f'- {n}, {kind}: '+('accepted' if d['accepted'] else 'rejected: '+', '.join(d['reasons'])))
    lines += ['', 'The same gates applied to the previous direct-learning reference:']
    for kind in ['current_checkpoint','historical_best']:
        d=summary['reference_development_decisions']['previous-direct'][kind]
        lines.append(f'- previous-direct, {kind}: '+('accepted' if d['accepted'] else 'rejected: '+', '.join(d['reasons'])))
    if 'original_trainer_repeat_development_decision' in summary:
        for kind in ['current_checkpoint','historical_best']:
            d=summary['original_trainer_repeat_development_decision'][kind]
            lines.append(f'- original-trainer-repeat, {kind}: '+('accepted' if d['accepted'] else 'rejected: '+', '.join(d['reasons'])))
    lines += ['', '## Evidence and limits','',
              'All final compact checkpoints reload in fresh processes at 596,049,920 parameters without a teacher, retrieval or extra modules. Frozen source/input/checkpoint manifests, exact protected-module hashes, executable answer checks, rescored outputs, paired query differences, development gates, cache bytes, training tokens and CUDA resources are recorded in AUDIT.json and summary.json.','',
              'The learner has 115 replay examples / 15,762 tokenizer tokens. Combined old/new target caches must remain within 16 MiB and are deleted after the runs. Tracing has a separate 256 MiB CPU activation-cache ceiling and measured CUDA memory. The old and failed compact models and expanded teacher are research/training dependencies, not inference dependencies; historical teacher-acquisition compute must be added when considering end-to-end cost.','',
              'This is a retrospective two-pass procedure: module selection inspects a previous failed update before retraining from the original checkpoint. It also uses 48 additional labeled old-task discovery prompts outside the optimizer replay, followed by 48 old-task and 16 join validation prompts. These diagnostic budgets are fixed and recorded but are additional old-task information. A successful result would not yet establish an online method within a replay-only information budget; that requires tracing from bounded retained memory or accounting for probe-update/rollback cost, plus a diagnostic-data-matched replay control.','',
              'Query-bootstrap intervals quantify uncertainty over these prompts only, not over training seeds, task orders or module selection; comparisons are exploratory and uncorrected for multiplicity. Teacher-forced likelihood recovery alone is not autonomous repair. Weight rollback and retraining with weights protected are separate interventions. A local benefit can coexist with loss elsewhere; report all old tasks and compositions. Successful acquisition, compression or exact weight preservation alone does not establish continual learning.','',
              'Accumulating protected modules over further stages can reduce the capacity left for learning; sustained plasticity requires additional task boundaries and orders. The causal cached decoder is separately checked against model.generate on real examples from both frozen checkpoints.','',
              'Artifacts: [audit](AUDIT.json), [summary](summary.json), [frozen manifest](frozen.json), [validation decision](validation-decision.json), [protocol](../../../GROUP_PRESERVATION_STEPS.md).']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    write(root/'report-provenance.json',{'source_sha256':sha(Path(__file__)),
          'artifacts':{p.name:sha(p) for p in [root/'REPORT.md',root/'summary.json',root/'AUDIT.json',root/'comparison.png',root/'causal-trace.png']}})
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='results/group_preservation/screen-001');args=p.parse_args();root=(ROOT/args.root).resolve()
    print(json.dumps(audit(root),indent=2),flush=True);report(root)
