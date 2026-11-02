#!/usr/bin/env python3
"""Independently rescore, audit frozen inputs/weights, and report the diagnosis."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from continual.tasks import compact, execute, score
from continual.worker import write
from continual.causal_diagnosis import bootstrap


def read(p):return json.loads(Path(p).read_text())


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def accuracy(rows,skill):
    xs=[x for x in rows if x['skill']==skill]
    return sum(x['strict_correct'] for x in xs)/len(xs)


def paired(a,b,skill):
    aa={x['id']:x for x in a if x['skill']==skill};bb={x['id']:x for x in b if x['skill']==skill}
    assert aa.keys()==bb.keys()
    gains=sum(not aa[k]['strict_correct'] and bb[k]['strict_correct'] for k in aa)
    losses=sum(aa[k]['strict_correct'] and not bb[k]['strict_correct'] for k in aa)
    return {'gains':gains,'losses':losses,'n':len(aa),
            **bootstrap([int(bb[k]['strict_correct'])-int(aa[k]['strict_correct']) for k in aa])}


def audit(root):
    from safetensors import safe_open
    import torch
    torch.set_num_threads(4)
    assert read(root/'progress.json')['status']=='completed'
    cfg=read(root/'config.json');checks={};count=0
    for family in ['reproduction','tracing']:
        frozen=read(root/f'frozen-{family}.json')
        for path,h in frozen['sources'].items():assert sha(ROOT/path)==h,path
        for path,h in frozen.get('inputs',{}).items():assert sha(root/path)==h,path
        for key in ['before_weights','after_weights']:
            for path,h in frozen.get(key,{}).items():assert sha(path)==h,path
        checks[family+'_frozen_hashes']=True
    data={name:read(root/'data'/f'{name}.json') for name in ['discovery','validation','activation-generation']}
    training=read(root/'train-data.json');train=training['current']+training['memory']+training['compositions']
    train_fp={hashlib.sha256(x['prompt'].encode()).hexdigest() for x in train}
    for name,xs in data.items():
        for x in xs:
            if 'fingerprint' not in x:continue
            assert x['fingerprint']==hashlib.sha256(x['prompt'].encode()).hexdigest()
            assert x['fingerprint'] not in train_fp
            program=json.loads(x['prompt']);assert compact(execute(program['rows'],program['ops']))==compact(x['answer'])
    assert not {x['fingerprint'] for x in data['discovery']} & {x.get('fingerprint') for x in data['validation']}
    historical={x['fingerprint'] for parent in [ROOT/'results/assimilation/experiment-001/archive',
                ROOT/cfg['reference_run']/'archive'] for p in parent.rglob('*.json') for x in read(p)}
    assert not historical & {x.get('fingerprint') for x in data['validation']}
    checks['no_train_discovery_validation_overlap']=True
    checks['validation_disjoint_from_historical_archives']=True
    assert len(data['discovery'])==64 and len(data['validation'])==385
    for report in list(root.glob('*-discovery.json'))+list(root.glob('*-validation.json'))+list(root.glob('variants/*/validation.json')):
        r=read(report)
        for x in r['results']:
            rescored=score(x['response'],x['answer'])
            assert all(x[k]==v for k,v in rescored.items())
        for skill,group in r['groups'].items():
            xs=[x for x in r['results'] if x['skill']==skill]
            assert group['n']==len(xs)
            for key in ['strict_correct','semantic_correct','valid_json']:
                assert abs(group[key]-sum(x[key] for x in xs)/len(xs))<1e-12
        assert r['deployed_parameters']==596049920
        if 'general_mc' in r:
            for x in r['general_mc']['results']:assert x['correct']==(x['selected']==x['answer'])
        count+=1
    for path in (root/'generation').glob('*.json'):
        if path.name=='controls.json':continue
        for x in read(path):assert all(x[k]==v for k,v in score(x['response'],x['answer']).items())
        count+=1
    assert count==16,count
    checks['rescored_reports']=count
    selection=read(root/'selection.json');trace=read(root/'trace.json')
    ranking=sorted(trace['weight_reversion'],key=lambda s:-sum(trace['weight_reversion'][s])/len(trace['ids']))
    assert ranking==selection['weight_ranking']
    assert all(not s.endswith(':residual') for s in ranking)
    for n in [1,3]:
        chosen=selection['weight_variants']['top'+str(n)];control=selection['weight_variants']['random'+str(n)]
        assert chosen==ranking[:n] and set(chosen).isdisjoint(control)
        assert [s.split(':')[1] for s in chosen]==[s.split(':')[1] for s in control]
    assert trace['ids']==[x['id'] for x in data['discovery']]
    checks['discovery_selection_and_matched_controls']=True
    paths=[ROOT/cfg['before_checkpoint'],root/'after/checkpoint']
    from transformers import AutoTokenizer
    old_tok=AutoTokenizer.from_pretrained(paths[0],local_files_only=True)
    new_tok=AutoTokenizer.from_pretrained(paths[1],local_files_only=True)
    assert old_tok.get_vocab()==new_tok.get_vocab()
    assert old_tok.chat_template==new_tok.chat_template
    assert old_tok.eos_token_id==new_tok.eos_token_id and old_tok.pad_token_id==new_tok.pad_token_id
    checks['before_after_tokenizer_alignment']=True
    oldfile=next(paths[0].glob('*.safetensors'));newfile=next(paths[1].glob('*.safetensors'))
    # Compare every serialized tensor, not merely parameter counts.
    with safe_open(oldfile,framework='pt',device='cpu') as old,safe_open(newfile,framework='pt',device='cpu') as new:
        keys=set(new.keys());assert keys==set(old.keys())
        params=sum(new.get_tensor(k).numel() for k in keys)
        assert params==596049920
        for name,sites in selection['weight_variants'].items():
            prefixes=[f'model.layers.{s.split(":")[0]}.{s.split(":")[1]}.' for s in sites]
            file=next((root/'variants'/name/'checkpoint').glob('*.safetensors'))
            with safe_open(file,framework='pt',device='cpu') as variant:
                assert set(variant.keys())==keys
                for k in keys:
                    expected=old if any(k.startswith(p) for p in prefixes) else new
                    assert torch.equal(variant.get_tensor(k),expected.get_tensor(k)),(name,k)
        # The final-state copy control assumes the same readout/norm/embedding.
        for k in keys:
            if not k.startswith('model.layers.'):
                assert torch.equal(old.get_tensor(k),new.get_tensor(k)),k
    checks['all_variant_tensors_match_declared_reversions']=True
    checks['deployed_parameters']=params
    assert trace['peak_activation_cache_bytes']<=cfg['activation_cache_bytes']
    assert trace['self_patch_max_error']<1e-5
    final=str(max(int(s.split(':')[0]) for s in trace['old_into_new']))+':residual'
    err=max(abs(p-(a-b)) for p,a,b in zip(trace['old_into_new'][final],trace['before'],trace['after']))
    assert err<1e-4,err
    assert all(read(root/'generation/controls.json').values())
    checks['self_patch_and_output_copy_controls']=True
    checks['final_residual_logp_copy_max_error']=err
    checks['no_teacher_cache_retained']=not list(root.glob('*-cache.pt'))
    assert checks['no_teacher_cache_retained']
    for name in ['acquire','after']:
        r=read(root/name/'training.json');assert r['training_tokens']<=98304
    checks['status']='passed'
    write(root/'AUDIT.json',checks)
    return checks


def report(root):
    cfg=read(root/'config.json');selection=read(root/'selection.json');trace=read(root/'trace.json')
    reports={name:read(root/f'{name}-validation.json') for name in ['before','after']}
    reports.update({name:read(root/'variants'/name/'validation.json') for name in selection['weight_variants']})
    rows={name:r['results'] for name,r in reports.items()}
    discovery=paired(read(root/'before-discovery.json')['results'],read(root/'after-discovery.json')['results'],'C')
    comparisons={name:{s:paired(rows['after'],xs,s) for s in 'ABCDE'} for name,xs in rows.items() if name!='after'}
    matched={name:{s:paired(rows['random'+name[-1]],rows[name],s) for s in 'ABCDE'} for name in ['top1','top3']}
    generation={p.stem:read(p) for p in (root/'generation').glob('*.json') if p.stem!='controls'}
    gs={name:{s:accuracy(xs,s) for s in 'CE'} for name,xs in generation.items()}
    summary={'validation':{name:{s:accuracy(xs,s) for s in 'ABCDE'} for name,xs in rows.items()},
             'discovery_before_to_after':discovery,'paired_against_after':comparisons,
             'selected_against_matched_random':matched,
             'generation':gs,
             'generation_paired_against_after':{name:{s:paired(generation['after'],xs,s) for s in 'CE'}
                                               for name,xs in generation.items() if name!='after'},
             'selection':selection,'resources':read(root/'resources.json')}
    write(root/'summary.json',summary)
    # Standalone scientific plot, with the final residual explicitly marked as a copy control.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    count=max(int(s.split(':')[0]) for s in trace['old_into_new'])+1
    fig,axes=plt.subplots(3,1,figsize=(13,7),constrained_layout=True)
    for ax,family,title,kinds in zip(axes,['old_into_new','new_into_old','weight_reversion'],
            ['Old activations into new: change in answer log probability',
             'New activations into old: change in answer log probability',
             'Old module weights into new: change in answer log probability'],
            [['self_attn','mlp','residual'],['self_attn','mlp','residual'],['self_attn','mlp']]):
        matrix=np.array([[sum(trace[family][f'{i}:{kind}'])/64 for i in range(count)] for kind in kinds])
        limit=max(abs(matrix.min()),abs(matrix.max()),0.01)
        im=ax.imshow(matrix,aspect='auto',cmap='RdBu',vmin=-limit,vmax=limit)
        ax.set_yticks(range(len(kinds)),kinds);ax.set_xticks(range(count),range(count));ax.set_title(title)
        ax.set_xlabel('Layer index (zero based)');fig.colorbar(im,ax=ax,label='Mean delta log probability')
    fig.suptitle('Discovery only: 64 sorting queries; final residual patch copies accumulated state')
    fig.savefig(root/'causal-sweep.png',dpi=160);fig.savefig(root/'causal-sweep.svg');plt.close(fig)
    lines=['# Causal diagnosis of one join update','',
           'Completed with Hugging Face/PyTorch on CUDA, using direct checkpoint access for internal hooks rather than the Ollama API. '
           'This is a component diagnosis, not a continual-learning or added-layer assimilation result. '
           'The tasks are synthetic JSON table operations; this does not demonstrate locating or deleting factual world knowledge.','',
           '## Reproduction and fresh validation','',
           f'Discovery sorting changed with {discovery["losses"]} losses and {discovery["gains"]} gains out of 64. '
           'The learning recipe and inputs were fixed; there was no seed search. '
           'Discovery is the old development set; the five atomic-task columns below use fresh fingerprint-disjoint queries. '
           'General MC uses the fixed legacy 32-question probe, not a new held-out general-capability benchmark.','',
           '| Saved model | Select /64 | Filter /64 | Sort /128 | Group /64 | Join /64 | General MC /32 |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for name,r in reports.items():
        cells=[f'{100*accuracy(r["results"],s):.2f}%' for s in 'ABCDE']
        lines.append('| '+name+' | '+' | '.join(cells)+f' | {100*r["general_mc"]["accuracy"]:.2f}% |')
    lines+=['','All variants are ordinary reloaded compact models with 596,049,920 parameters and retrieval off. '
            'Variant names refer to sorting discovery scores, not validation winners. No variant is automatically accepted.','',
            '## Discovery-selected interventions','',
            f'Activation hypothesis: `{selection["activation_site"]}`. Random activation control: `{selection["random_activation_site"]}`. '
            'Module indices are zero based. The ranking uses all 64 sorting queries and raw mean answer-token log probability changes.','',
            '| Variant | Old modules restored | Fresh sorting difference vs after | Query bootstrap 95% | Fresh join difference |',
            '|---|---|---:|---|---:|']
    for name,sites in selection['weight_variants'].items():
        c=comparisons[name]['C'];e=comparisons[name]['E'];lo,hi=c['query_bootstrap_95']
        lines.append(f'| {name} | {", ".join(sites)} | {100*c["mean"]:+.2f} pp | [{100*lo:+.2f}, {100*hi:+.2f}] pp | {100*e["mean"]:+.2f} pp |')
    lines+=['','These intervals resample queries, not training seeds; they do not correct discovery ranking for multiple comparisons. '
            'Independent validation and matched random controls are the primary checks.','',
            '## Autonomous activation validation','',
            '| Condition | Sorting /32 | Join /16 |','|---|---:|---:|']
    for name,g in gs.items():lines.append(f'| {name} | {100*g["C"]:.2f}% | {100*g["E"]:.2f}% |')
    lines+=['','No gold answers or prefixes enter autonomous generation. The donor sees the same generated prefix as the recipient. '
            'The self-patch must reproduce the recipient; final residual replacement must reproduce the donor. '
            'The latter is an output-copy control and does not locate knowledge. Shuffled donor activations can be off distribution. '
            'Activation-patched inference requires the donor model at runtime; it is a diagnostic, not assimilated memory. '
            'Only the separately reloaded weight variants run without that donor.','',
            '## Interpretation and added layers','',
            'Activation recovery shows a causal dependency under a specific intervention. Weight reversion tests changed module parameters. '
            'Neither identifies an exclusive address for a fact, proves erasure, or guarantees a safe edit. '
            'This study uses broad modules rather than Anthropic-style sparse feature circuits. Cross-model activations may be misaligned.','',
            'For actual temporary nonlinear blocks, use these measurements to choose hypotheses for representation distillation and preservation losses. '
            'Compare output-only compression, representation-guided compression and matched random-site compression with the same budget. '
            'Ablate the added blocks to establish useful contribution, then remove the blocks and all teachers before evaluating the saved compact student. '
            'Repeat learning boundaries and test acquisition, retention, transfer and sustained learnability. '
            'This run trained LoRA acquisition/consolidation; it did not train or assimilate added nonlinear blocks.','',
            'See [the frozen protocol](../../../CAUSAL_ASSIMILATION_STEPS.md) for the follow-up. '
            'Intermediate-representation distillation has precedent in [FitNets](https://arxiv.org/abs/1412.6550); '
            'causal interpretation follows [activation-patching guidance](https://arxiv.org/abs/2404.15255) and '
            '[Anthropic’s methods](https://transformer-circuits.pub/2025/attribution-graphs/methods.html).','',
            '## Resources and audit','',
            '```json',json.dumps(read(root/'resources.json'),indent=2),'```','',
            'The audit rechecks source/input/checkpoint hashes, all generated-answer scores, held-out fingerprints, target execution, '
            'site selection, every tensor of all four reversion checkpoints, parameter counts, cache bounds and copy controls. '
            'Full audit: `AUDIT.json`; per-query effects: `trace.json`; selection: `selection.json`; paired uncertainties: `summary.json`. '
            'Acquisition and consolidation token counts are in their respective `training.json` files. '
            'Retained before/after/teacher/variant checkpoints are research audit artifacts, not a bounded online memory mechanism.','',
            '![Causal discovery sweep](causal-sweep.png)','']
    m=matched['top3']['C'];lo,hi=m['query_bootstrap_95']
    conclusion=(f'The selected three-module reversion changed sorting from {100*accuracy(rows["after"],"C"):.2f}% '
                f'to {100*accuracy(rows["top3"],"C"):.2f}%, versus {100*accuracy(rows["random3"],"C"):.2f}% '
                f'for matched random reversion. The selected-minus-random difference was {100*m["mean"]:+.2f} pp '
                f'(paired query bootstrap 95% [{100*lo:+.2f}, {100*hi:+.2f}] pp). '
                f'The original model scored {100*accuracy(rows["before"],"C"):.2f}% on sorting. '
                f'Grouping after selected reversion was {100*accuracy(rows["top3"],"D"):.2f}%, '
                f'compared with {100*accuracy(rows["before"],"D"):.2f}% before learning join. '
                'This is partial recovery, not restoration of prior skills or a validated safe editing location. '
                'The activation intervention improved sorting by one of 32 cases while losing one of 16 joins; '
                'the single selected weight reversion did not reproduce that improvement.')
    lines[4:4]=['## Main finding','',conclusion,'']
    (root/'REPORT.md').write_text('\n'.join(lines))


if __name__=='__main__':
    root=ROOT/'results/causal_assimilation/diagnostic-001'
    print(json.dumps(audit(root),indent=2),flush=True)
    report(root)
