"""Freeze one eligible mixed-source candidate, then consume PFFfinal once."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pff_reader import sha

OUT=ROOT/'reports/pff_mixed_final';FREEZE=ROOT/'docs/experiments/pff-mixed/final_freeze.json'
BUNDLE=ROOT/'assets/pff-mixed-candidate/bundle.json'


def now():return datetime.now(timezone.utc).isoformat()


def freeze():
    from check_evaluation import load_inputs
    policy,_,_=load_inputs()
    if FREEZE.exists() or (OUT/'OPENED.json').exists() or (ROOT/'reports/pff_final/OPENED.json').exists():raise ValueError('Freeze/reserve already used')
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True):raise ValueError('Commit exact source/assets before freezing')
    selection_path=ROOT/'reports/pff_mixed/selection.json';selection=json.loads(selection_path.read_text())
    if selection['status']!='ELIGIBLE_FOR_UNOPENED_FINAL' or not all(selection['decisions'][selection['candidate']]['checks'].values()):raise ValueError('Development/regression requirements failed')
    for file,digest in selection['result_sha256'].items():
        if sha(ROOT/'reports/pff_mixed'/file)!=digest:raise ValueError('Selected development evidence changed')
    reproduction=ROOT/'out/candidates/pff-mixed/verification.json';proof=json.loads(reproduction.read_text());bundle=json.loads(BUNDLE.read_text())
    if not proof['status'].startswith('PASS:') or len(proof['runs'])!=2 or proof['runs'][0]['csv_sha256']!=proof['runs'][1]['csv_sha256'] or proof['bundle_sha256']!=sha(BUNDLE) or proof['candidate']!=selection['candidate']:
        raise ValueError('Exact-file frozen CPU reproduction incomplete')
    if sha(ROOT/'out/candidates/pff-mixed/Ruses Cules.csv')!=proof['runs'][0]['csv_sha256']:raise ValueError('Candidate CSV changed')
    if any(sha(ROOT/path)!=digest for path,digest in proof['inference_source_sha256'].items()):raise ValueError('Inference code changed since exact-file reproduction')
    equality=ROOT/'reports/pff_mixed/deployment_equivalence.json';eq=json.loads(equality.read_text())
    if eq['status']!='PASS' or eq['bundle_sha256']!=sha(BUNDLE) or eq['inference_source_sha256']!=sha(ROOT/'mtah/mixed_candidate.py'):raise ValueError('Deployment/scoring equivalence failed')
    if bundle['candidate']!=selection['candidate'] or sha(BUNDLE)!=selection['bundle_sha256']:raise ValueError('Selected candidate changed')
    ids=policy['reserve']['final_ids']
    for game in ids:
        for folder in ['Tracking','Tracking Data','Metadata','Rosters','Event Data','converted']:
            if any((ROOT/'external/pff'/folder/(game+suffix)).exists() for suffix in ['.json','.jsonl.bz2','_geometry.npz']):raise ValueError('Final content already exists before freeze')
        if (ROOT/'reports/pff_access/references'/('kloppy_'+game+'.npz')).exists():raise ValueError('Final reference already exists')
    files=list((ROOT/'mtah').rglob('*.py'))+list((ROOT/'mtah').rglob('*.json'))
    names=['tools/pff_mixed_final.py','tools/generate_mixed_candidate.py','tools/pff_reader.py','tools/skillcorner_reader.py',
        'tools/fetch_pff_training.py','tools/pff_final.py','tools/pff_development.py','tools/decoder_development.py',
        'tools/quality_audit.py','tools/check_evaluation.py','ref/official/evaluation/possession_metrics.py',
        'docs/evaluation/policy.json','docs/experiments/more-data/pff_protocol.json','docs/experiments/pff/plan_v2.json',
        'docs/experiments/pff-mixed/plan.json','reports/pff_access/browser_complete_listing.json',
        'reports/pff_mixed/selection.json','reports/pff_mixed/deployment_equivalence.json',
        'reports/pff_mixed/old_results.json','reports/pff_mixed/old_provenance.json','reports/pff_mixed/pff_dev_results.json',
        'reports/pff_mixed/reused_results.json','out/candidates/pff-mixed/verification.json','out/candidates/pff-mixed/Ruses Cules.csv',
        'requirements.txt','requirements-validation.txt','requirements-temporal.txt','requirements-pff-research.txt']
    files.extend(ROOT/name for name in names)
    for folder in [ROOT/'assets/v4',BUNDLE.parent]:files.extend(p for p in folder.rglob('*') if p.is_file())
    result={'frozen_utc':now(),'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'candidate':selection['candidate'],'control':'unchanged v4','final_ids':ids,'gate':policy['reserve']['final_gate'],
        'reference':'Unchanged primary kloppy canonical, >=99% coverage/allHAD; fixed SDK-shadow sensitivity mean loss<=.01',
        'files_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sorted(set(files))},
        'final_contents_previously_opened':False,'target_quality_assessed':False,
        'one_shot':'Opening consumes the reserve even after failure; no model switching or post-test selection'}
    FREEZE.write_text(json.dumps(result,indent=2)+'\n');print('Mixed candidate frozen; original final12contents remain unopened',flush=True)


def run():
    if not FREEZE.exists():raise ValueError('Freeze one eligible candidate before any final content')
    frozen=json.loads(FREEZE.read_text())
    for path,digest in frozen['files_sha256'].items():
        if sha(ROOT/path)!=digest:raise ValueError('Frozen source/artifact/evidence changed: '+path)
    OUT.mkdir(exist_ok=True);marker=OUT/'OPENED.json'
    if marker.exists() or (ROOT/'reports/pff_final/OPENED.json').exists():raise ValueError('Original final reserve already consumed')
    marker.write_text(json.dumps({'opened_utc':now(),'freeze_sha256':sha(FREEZE),'status':'Reserve consumed even if subsequent steps fail'},indent=2)+'\n')
    import bz2
    import numpy as np
    import torch
    import requests
    import time
    from mtah import mixed_candidate,v4,degrade,expanded_team
    from mtah.methods import boost,boost_train
    from fetch_pff_training import download
    from pff_reader import load_geometry,tracking_reference
    from pff_development import spans
    from decoder_development import metric
    from pff_final import final_decision
    catalog=json.loads((ROOT/'reports/pff_access/browser_complete_listing.json').read_text())['files']
    protocol=json.loads((ROOT/'docs/experiments/more-data/pff_protocol.json').read_text());registered={g['game_id']:g for g in protocol['final_test']}
    bundle,state=mixed_candidate.load_bundle(BUNDLE);assets=v4.load();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    params=dict(boost_train.decoder_params(),**json.loads((ROOT/'mtah/methods/v2_params.json').read_text())['decoder']);records=[];sources=[]
    try:
        for game in frozen['final_ids']:
            raw={}
            for folder,suffix in [('Tracking Data','.jsonl.bz2'),('Metadata','.json'),('Rosters','.json')]:
                relative=folder+'/'+game+suffix;found=[f for f in catalog if f['path']==relative]
                if len(found)!=1:raise ValueError('Missing/duplicate frozen source file metadata')
                path=ROOT/'external/pff'/relative
                if path.exists():raise ValueError('Unexpected existing final content')
                path.parent.mkdir(exist_ok=True)
                for attempt in [1,2,3]:
                    part=path.with_name(path.name+f'.part-attempt{attempt}')
                    if part.exists():raise ValueError('Unexpected partial reserve file')
                    try:download(found[0],part);part.rename(path);break
                    except requests.RequestException:
                        if attempt==3:raise
                        # Preserve interrupted bytes; retry identical transport
                        # only, before any prediction on this source file.
                        time.sleep(2*attempt)
                raw[folder]=path
            meta=json.loads(raw['Metadata'].read_text());meta=meta[0] if isinstance(meta,list) else meta
            with bz2.open(raw['Tracking Data'],'rt') as stream:first=json.loads(next(stream))
            if float(first['gameRefId'])!=int(game) or str(meta['id'])!=game or meta['homeTeam']['name']!=registered[game]['home'] or meta['awayTeam']['name']!=registered[game]['away']:raise ValueError('Reserve source identity/country mismatch')
            data,geometry=load_geometry(raw['Tracking Data'],raw['Metadata'],raw['Rosters'])
            gt,alt,reference=tracking_reference(raw['Tracking Data'],data['time'],data['period'],geometry['fps'])
            if reference['known_fraction']<.99 or any(reference['class_frames'][c]<=0 for c in ['H','A','D']) or not all(np.any(data['period']==p) for p in [1,2]):raise ValueError('Full-task source invalid; cannot drop a match or change the endpoint')
            parts=spans(data['period']);dummy=np.where(data['period']>0,0,3).astype(np.int8)
            noisy=degrade.degrade(dict(data,gt=dummy),seed=7,params=expanded_team.BLIND);noisy.pop('gt');predictions={}
            for scenario,observed in [('natural_detected',data),('label_blind_seed7',noisy)]:
                control,candidate=mixed_candidate.paired_probabilities(observed,parts,assets,state)
                for name,probabilities in [('v4',control),(bundle['candidate'],candidate)]:predictions[scenario+'__'+name]=boost.label(*probabilities,parts,len(dummy),params)
            pred_file=OUT/(game+'_predictions.npz');np.savez_compressed(pred_file,**predictions)
            for ref_name,truth in [('canonical',gt),('sdk_shadow',alt)]:
                for key,pred in predictions.items():
                    scenario,name=key.split('__');records.append({'game':game,'scenario':scenario,'candidate':name,'reference':ref_name,**metric(truth,pred)})
            receipt={'game':game,'geometry':geometry,'reference':reference,'predictions_sha256':sha(pred_file)};sources.append(receipt)
            (OUT/(game+'_scores.json')).write_text(json.dumps({'records':records[-8:],'source':receipt},indent=2)+'\n');print('Final',game,'complete',flush=True)
        neutral=[dict(r,candidate='source_seed20261113' if r['candidate']==bundle['candidate'] else r['candidate']) for r in records]
        decision=final_decision(neutral,frozen['final_ids'],frozen['gate']);decision['candidate']=bundle['candidate']
        (OUT/'results.json').write_text(json.dumps({'freeze_sha256':sha(FREEZE),'records':records,'source_receipts':sources,'decision':decision},indent=2)+'\n');print(json.dumps(decision),flush=True)
    except Exception as error:
        (OUT/'FAILURE.json').write_text(json.dumps({'status':'FAIL: reserve consumed','reason':str(error),'complete_games':len(sources),'records':records},indent=2)+'\n');raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['freeze','run']);a=p.parse_args();freeze() if a.action=='freeze' else run()
