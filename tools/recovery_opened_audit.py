"""Frozen re-evaluation of already opened matches, not a fresh holdout or training."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='1'
import argparse,json,sys,time,subprocess
from datetime import datetime,timezone
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from pff_reader import sha,load_geometry,tracking_reference
from pff_development import spans
from pff_mixed_score import decoder
from decoder_development import metric
from quality_audit import official
DOC=ROOT/'docs/experiments/recovery-opened-audit';OUT=ROOT/'reports/recovery_opened_audit'
SCENARIOS=['natural_detected','label_blind_seed7']


def now():return datetime.now(timezone.utc).isoformat()
def plan():return json.loads((DOC/'plan.json').read_text())
def check():
    f=json.loads((DOC/'freeze.json').read_text())
    amendment=DOC/'reporting_amendment.json'
    replacements={}
    if amendment.exists():
        a=json.loads(amendment.read_text())
        if a['original_freeze_sha256']!=sha(DOC/'freeze.json'):raise ValueError('Wrong audit amendment')
        if set(a['changes'])!={'tools/recovery_opened_audit.py'}:raise ValueError('Only reporting-tool amendment allowed')
        for name,item in a['changes'].items():
            if f['files_sha256'][name]!=item['before']:raise ValueError('Wrong original reporting source')
            replacements[name]=item['after']
    for name,h in f['files_sha256'].items():
        if sha(ROOT/name)!=replacements.get(name,h):raise ValueError('Frozen evidence/source changed: '+name)
    return f

def raw_paths(game):
    return {k:ROOT/'external/pff'/f/(game+s) for k,f,s in [('tracking','Tracking Data','.jsonl.bz2'),('metadata','Metadata','.json'),('rosters','Rosters','.json')]}

def freeze():
    if (DOC/'freeze.json').exists() or OUT.exists():raise ValueError('Already frozen')
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True):raise ValueError('Commit audit source before freezing')
    files=set((ROOT/'mtah').rglob('*.py'))|set((ROOT/'mtah').rglob('*.json'))
    for folder in ['assets/v4','assets/pff-mixed-candidate','assets/recovery-candidate']:files.update(p for p in (ROOT/folder).rglob('*') if p.is_file())
    files.update(ROOT/f for f in ['tools/recovery_opened_audit.py','tools/recovery_candidate_runtime.py','tools/short_gap_methods.py','tools/pff_reader.py','tools/pff_development.py','tools/pff_mixed_score.py','tools/decoder_development.py','tools/quality_audit.py','docs/experiments/recovery-opened-audit/plan.json','docs/experiments/more-data/pff_protocol.json','ref/official/evaluation/possession_metrics.py','submission_manifest.json'])
    fixtures=plan()['fixtures'];protocol=json.loads((ROOT/'docs/experiments/more-data/pff_protocol.json').read_text());train={g['game_id'] for g in protocol['training']};dev={g['game_id'] for g in protocol['development']}
    assert len(fixtures)==18 and len({g['game_id'] for g in fixtures})==18
    assert not {g['game_id'] for g in fixtures}&(train|dev)
    for g in fixtures:
        paths=raw_paths(g['game_id']);old=ROOT/g['historical_scores'];source=json.loads(old.read_text())['source']
        for k,p in paths.items():
            if sha(p)!=source['geometry']['source_sha256'][k]:raise ValueError('Changed original input: '+str(p))
        files.update(paths.values());files.add(old)
        pred=old.with_name(g['game_id']+'_predictions.npz')
        if sha(pred)!=source['predictions_sha256']:raise ValueError('Historical predictions changed')
        files.add(pred)
    record=dict(frozen_utc=now(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                scope='Previously opened 12+6 fixtures; fixed current/parent comparison, not a new independent test',
                files_sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted(files)})
    (DOC/'freeze.json').write_text(json.dumps(record,indent=2)+'\n');print('Frozen all 18 fixtures, source, weights, reference code and prior outputs',flush=True)


def diagnostic(gt,pred,data):
    runs=official.valid_runs(gt);filled=official.fill_abstentions(pred,runs)
    truth=official.possession_spells(gt,runs);predicted=official.possession_spells(filled,runs)
    overlaps=official.spell_overlaps(truth,predicted);matched=official.matched_gt_flags(overlaps,len(truth),.5)
    lengths=np.array([(b-a)/25 for a,b,k in truth]);bins={}
    for key,lo,hi in [('under_1s',0,1),('1_to_3s',1,3),('3_to_10s',3,10),('10s_plus',10,np.inf)]:
        mask=(lengths>=lo)&(lengths<hi);bins[key]=dict(reference=int(mask.sum()),matched=int(matched[mask].sum()))
    seen=np.isfinite(data['ball'][:,:2]).all(axis=1);obs={}
    for label,m in [('visible',seen),('missing',~seen)]:
        mask=m&(gt<3);live=m&(gt<2);dead=m&(gt==2)
        obs[label]=dict(frames=int(mask.sum()),correct=int(((gt==pred)&mask).sum()),live=int(live.sum()),false_dead=int(((pred==2)&live).sum()),
                        wrong_team=int(((pred<2)&(pred!=gt)&live).sum()),dead=int(dead.sum()),false_live=int(((pred<2)&dead).sum()))
    # Exhaustive, descriptive attribution by majority predicted label inside each missed reference spell.
    missed={'mostly_dead':0,'mostly_opponent':0,'mostly_correct_team_but_bad_extent_or_one_to_one':0,'mostly_abstain':0};missed_seen={k:[] for k in missed}
    for i,(a,b,k) in enumerate(truth):
        if matched[i]:continue
        majority=int(np.bincount(filled[a:b],minlength=4).argmax())
        key='mostly_dead' if majority==2 else 'mostly_abstain' if majority==3 else 'mostly_correct_team_but_bad_extent_or_one_to_one' if majority==k else 'mostly_opponent'
        missed[key]+=1;missed_seen[key].append(float(seen[a:b].mean()))
    truth_overlap=np.zeros(len(truth),int);pred_overlap=np.zeros(len(predicted),int)
    for _,i,j in overlaps:truth_overlap[i]+=1;pred_overlap[j]+=1
    transitions=official.transition_scores(official.transitions(gt,runs),official.transitions(filled,runs),25,typed='pair')
    transition={k:v for k,v in transitions.items() if k not in ['pairs','missed','spurious']}
    per_period=[]
    for a,b in spans(data['period']):
        r=metric(gt[a:b],pred[a:b]);per_period.append(dict(period=int(data['period'][a]),**r))
    gaps=[]
    for a,b in spans(data['period']):
        edges=np.diff(np.r_[0,(~seen[a:b]).astype(np.int8),0]);gaps.extend(((np.flatnonzero(edges==-1)-np.flatnonzero(edges==1))/25).tolist())
    return dict(short_spells=bins,observation=obs,missed_at_iou05=missed,
        missed_mean_visible_fraction={k:float(np.mean(v)) if v else None for k,v in missed_seen.items()},
        reference_spells_overlapped_by_multiple_same_team_predictions=int((truth_overlap>1).sum()),
        predictions_overlapping_multiple_same_team_reference_spells=int((pred_overlap>1).sum()),
        transition_pair_tolerance_1s=transition,spell_by_iou=official.spell_scores(gt,filled,runs)['per_iou'],per_period=per_period,
        gap_duration=dict(count=len(gaps),median=float(np.median(gaps)) if gaps else 0,p90=float(np.quantile(gaps,.9)) if gaps else 0,max=max(gaps,default=0)),
        note='Overlap counts are diagnostic associations, not exclusive causal categories; majority attribution is at IoU0.5 only.')


def initialize():
    global CURRENT,PARENT,BUNDLE
    import torch
    from mtah import recovery_candidate,mixed_candidate
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    BUNDLE,CURRENT=recovery_candidate.load_bundle(ROOT/'assets/recovery-candidate/bundle.json')
    _,PARENT=mixed_candidate.load_bundle(ROOT/'assets/pff-mixed-candidate/bundle.json')


def worker(fixture):
    from mtah import degrade,expanded_team,recovery_candidate,mixed_candidate
    from mtah.methods import boost
    from recovery_candidate_runtime import probabilities
    game=fixture['game_id'];folder=OUT/game;folder.mkdir(exist_ok=False);started=time.monotonic()
    paths=raw_paths(game);meta=json.loads(paths['metadata'].read_text());meta=meta[0] if isinstance(meta,list) else meta
    if str(meta['id'])!=game or any(meta[k+'Team']['name']!=fixture[k] for k in ['home','away']):raise ValueError('Fixture identity mismatch')
    data,geo=load_geometry(paths['tracking'],paths['metadata'],paths['rosters']);parts=spans(data['period']);n=len(data['ball'])
    if 'gt' in data:raise ValueError('Reference in inference geometry')
    print('Loaded',game,'geometry',round(time.monotonic()-started,1),flush=True)
    dummy=np.where(data['period']>0,0,3).astype(np.int8);blind=degrade.degrade(dict(data,gt=dummy),seed=7,params=expanded_team.BLIND);blind.pop('gt');blind['period']=data['period']
    predictions={};prob_arrays={};parity=[];observed={'natural_detected':data,'label_blind_seed7':blind}
    for scenario,d in observed.items():
        current,parent=probabilities(d,parts,CURRENT[0],CURRENT[1],CURRENT[2],PARENT)
        if game=='10502':
            direct=recovery_candidate.probabilities(d,parts,CURRENT);old=mixed_candidate.probabilities(d,parts,CURRENT[0],PARENT)
            errors=[max(float(np.max(np.abs(x-y))) for x,y in zip(current,direct)),max(float(np.max(np.abs(x-y))) for x,y in zip(parent,old))]
            if max(errors)!=0:raise ValueError('Paired runtime differs from packaged entrypoints')
            parity.append(dict(scenario=scenario,max_probability_error=errors))
        for name,pair,params in [('weighted_short_gap',current,BUNDLE['decoder']),('boundary102',parent,decoder())]:
            predictions[scenario+'__'+name]=boost.label(*pair,parts,n,params)
        prob_arrays[scenario+'__seen']=np.isfinite(d['ball'][:,:2]).all(axis=1)
        print('Predicted',game,scenario,round(time.monotonic()-started,1),flush=True)
    predpath=folder/'predictions.npz';np.savez_compressed(predpath,**predictions);predicted_at=now()
    gt,alt,ref=tracking_reference(paths['tracking'],data['time'],data['period'],geo['fps'])
    if ref['known_fraction']<.99 or any(ref['class_frames'][k]<=0 for k in ['H','A','D']):raise ValueError('Ineligible reference; retain failure, no dropping')
    np.savez_compressed(folder/'diagnostic_inputs.npz',gt=gt,alternate=alt,period=data['period'],**prob_arrays)
    rows=[];diagnostics=[];historic=json.loads((ROOT/fixture['historical_scores']).read_text());hist={ (r['scenario'],r['reference']):r for r in historic['records'] if r['candidate']=='boundary102'}
    with np.load((ROOT/fixture['historical_scores']).with_name(game+'_predictions.npz')) as hpred:
        comparison=[dict(scenario=s,differing_parent_frames=int(np.sum(predictions[s+'__boundary102']!=hpred[s+'__boundary102']))) for s in SCENARIOS]
    for reference,truth in [('canonical',gt),('sdk_shadow',alt)]:
        for key,pred in predictions.items():
            scenario,name=key.split('__');r=dict(game=game,cohort=fixture['cohort'],scenario=scenario,candidate=name,reference=reference,**metric(truth,pred))
            if name=='boundary102':r['historical_parent_spell_delta']=r['spell_f1_mean']-hist[scenario,reference]['spell_f1_mean']
            rows.append(r)
            if reference=='canonical':diagnostics.append(dict(game=game,cohort=fixture['cohort'],scenario=scenario,candidate=name,**diagnostic(gt,pred,observed[scenario])))
    receipt=dict(game=game,fixture=fixture,geometry=geo,reference=ref,predictions_sha256=sha(predpath),predictions_saved_before_reference_utc=predicted_at,
                 seconds=time.monotonic()-started,paired_packaged_parity=parity,historical_parent_comparison=comparison)
    result=dict(records=rows,diagnostics=diagnostics,source=receipt,freeze_sha256=sha(DOC/'freeze.json'))
    (folder/'result.json').write_text(json.dumps(result,indent=2)+'\n');print('Completed',game,round(receipt['seconds'],1),flush=True)
    return result


def summary(records):
    result=[]
    for cohort in ['bridge6','original_final12','all18']:
        for scenario in SCENARIOS:
            for ref in ['canonical','sdk_shadow']:
                rr=[r for r in records if (cohort=='all18' or r['cohort']==cohort) and r['scenario']==scenario and r['reference']==ref]
                old={r['game']:r for r in rr if r['candidate']=='boundary102'};new={r['game']:r for r in rr if r['candidate']=='weighted_short_gap'}
                expected=18 if cohort=='all18' else 6 if cohort=='bridge6' else 12
                if len(old)!=expected or old.keys()!=new.keys() or len(rr)!=expected*2:raise ValueError('Incomplete or duplicate comparison')
                delta=np.array([new[g]['spell_f1_mean']-old[g]['spell_f1_mean'] for g in old]);macro=float(np.mean([new[g]['macro_f1']-old[g]['macro_f1'] for g in old]))
                result.append(dict(cohort=cohort,scenario=scenario,reference=ref,games=expected,parent_spell=float(np.mean([r['spell_f1_mean'] for r in old.values()])),
                    current_spell=float(np.mean([r['spell_f1_mean'] for r in new.values()])),gain=float(delta.mean()),wins=int((delta>0).sum()),losses=int((delta<0).sum()),worst_gain=float(delta.min()),macro_gain=macro,
                    per_game_gain={g:new[g]['spell_f1_mean']-old[g]['spell_f1_mean'] for g in old},
                    review_guard_pass=bool(delta.mean()>=0 and macro>=-.002 and delta.min()>=-.02)))
    return result


def finish_saved(fixture):
    """Finish only reporting from immutable predictions/reference arrays after report failure."""
    game=fixture['game_id'];folder=OUT/game;predpath=folder/'predictions.npz';inputpath=folder/'diagnostic_inputs.npz'
    with np.load(predpath,allow_pickle=False) as z:predictions=dict(z)
    with np.load(inputpath,allow_pickle=False) as z:inputs=dict(z)
    expected={s+'__'+m for s in SCENARIOS for m in ['boundary102','weighted_short_gap']}
    if set(predictions)!=expected:raise ValueError('Incomplete saved predictions')
    gt,alt=inputs['gt'],inputs['alternate'];period=inputs['period'];valid=period>0
    historic=json.loads((ROOT/fixture['historical_scores']).read_text());prior=historic['source']
    # Metadata comes from the original receipt, whose raw inputs and reader hashes
    # are still identical. Validate actual regenerated reference counts/coverage.
    for truth in [gt,alt]:
        if len(truth)!=prior['geometry']['frames25']:raise ValueError('Changed frame clock')
        if any(int(((truth==i)&valid).sum())!=prior['reference']['class_frames'][label] for i,label in enumerate(['H','A','D','X'])):raise ValueError('Changed reference counts')
    if not all(len(a)==len(gt) for a in predictions.values()):raise ValueError('Prediction length mismatch')
    rows=[];diags=[];comparison=[];hist={(r['scenario'],r['reference']):r for r in historic['records'] if r['candidate']=='boundary102'}
    with np.load((ROOT/fixture['historical_scores']).with_name(game+'_predictions.npz')) as old:
        for scenario in SCENARIOS:comparison.append(dict(scenario=scenario,differing_parent_frames=int(np.sum(predictions[scenario+'__boundary102']!=old[scenario+'__boundary102']))))
    for reference,truth in [('canonical',gt),('sdk_shadow',alt)]:
        for key,pred in predictions.items():
            scenario,name=key.split('__');r=dict(game=game,cohort=fixture['cohort'],scenario=scenario,candidate=name,reference=reference,**metric(truth,pred))
            if name=='boundary102':r['historical_parent_spell_delta']=r['spell_f1_mean']-hist[scenario,reference]['spell_f1_mean']
            rows.append(r)
            if reference=='canonical':
                seen=inputs[scenario+'__seen'];observation=dict(period=period,ball=np.where(seen[:,None],np.zeros((len(gt),3)),np.nan))
                diags.append(dict(game=game,cohort=fixture['cohort'],scenario=scenario,candidate=name,**diagnostic(gt,pred,observation)))
    receipt=dict(game=game,fixture=fixture,geometry=prior['geometry'],reference=prior['reference'],predictions_sha256=sha(predpath),diagnostic_inputs_sha256=sha(inputpath),
        metadata_origin='Original receipt for unchanged raw source and reader; current regenerated reference counts checked exactly',
        recovery='Predictions and reference arrays were saved by original frozen worker before missing-period reporting error. No fitting or inference change.',
        paired_packaged_parity='For10502, original worker asserted zero current/parent probability differences in both scenarios before saving these predictions' if game=='10502' else 'Shared paired runtime; independently checked on10502',
        historical_parent_comparison=comparison)
    result=dict(records=rows,diagnostics=diags,source=receipt,freeze_sha256=sha(DOC/'freeze.json'),reporting_amendment_sha256=sha(DOC/'reporting_amendment.json'))
    (folder/'result.json').write_text(json.dumps(result,indent=2)+'\n');print('Recovered report',game,flush=True);return result


def run(resume=False):
    check();OUT.mkdir(exist_ok=resume);rows=[];diags=[];sources=[];pending=[]
    for g in plan()['fixtures']:
        dest=OUT/g['game_id']/'result.json'
        if dest.exists():
            r=json.loads(dest.read_text());assert r['freeze_sha256']==sha(DOC/'freeze.json');rows.extend(r['records']);diags.extend(r['diagnostics']);sources.append(r['source'])
        elif (OUT/g['game_id']/'diagnostic_inputs.npz').exists():
            r=finish_saved(g);rows.extend(r['records']);diags.extend(r['diagnostics']);sources.append(r['source'])
        else:pending.append(g)
    with ProcessPoolExecutor(max_workers=2,mp_context=get_context('spawn'),initializer=initialize) as pool:
        for f in as_completed([pool.submit(worker,g) for g in pending]):
            r=f.result();rows.extend(r['records']);diags.extend(r['diagnostics']);sources.append(r['source'])
    assert len(rows)==144 and len(diags)==72 and len(sources)==18
    check();result=dict(scope=plan()['scope'],records=rows,diagnostics=diags,source_receipts=sources,summary=summary(rows),freeze_sha256=sha(DOC/'freeze.json'))
    (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n');(DOC/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['summary'],indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run']);p.add_argument('--resume',action='store_true');a=p.parse_args()
    freeze() if a.action=='freeze' else run(a.resume)
