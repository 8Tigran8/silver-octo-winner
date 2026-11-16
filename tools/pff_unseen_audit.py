"""Prospective evaluation of the six unopened bridge matches, never selection.

This new audit does not alter the original train/dev/final protocol or unblock
bridges in any training/development tool. Predictions precede reference decoding.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from pff_reader import sha

DOC=ROOT/'docs/experiments/unseen-bridge-audit'
OUT=ROOT/'reports/unseen_bridge_audit'
IDS=('10512','10511','10509','10507','10505','10502')


def now():return datetime.now(timezone.utc).isoformat()


def registry():
    protocol=json.loads((ROOT/'docs/experiments/more-data/pff_protocol.json').read_text())
    games={g['game_id']:g for g in protocol['country_bridge_excluded_from_training_and_selection']}
    if set(games)!=set(IDS):raise ValueError('Bridge identity registry changed')
    other={g['game_id'] for split in ['training','development','final_test'] for g in protocol[split]}
    trained=json.loads((ROOT/'assets/pff-mixed-candidate/bundle.json').read_text())['training_games']
    if set(IDS)&other or any('pff/'+g in trained for g in IDS):raise ValueError('Evaluation/training overlap')
    return protocol,games


def ensure_unopened():
    for game in IDS:
        for folder in ['Tracking','Tracking Data','Metadata','Rosters','Event Data','converted']:
            if any((ROOT/'external/pff'/folder/(game+suffix)).exists() for suffix in ['.json','.jsonl.bz2','_geometry.npz']):
                raise ValueError('Bridge contents already present before freeze: '+game)
        if (ROOT/'reports/pff_access/references'/('kloppy_'+game+'.npz')).exists():raise ValueError('Bridge references already present')


def freeze():
    protocol,games=registry();ensure_unopened()
    if (DOC/'freeze.json').exists() or OUT.exists():raise ValueError('Audit already frozen/opened')
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True):raise ValueError('Commit source and plan first')
    files=set((ROOT/'mtah').rglob('*.py'))|set((ROOT/'mtah').rglob('*.json'))
    for folder in ['assets/v4','assets/pff-mixed-candidate']:
        files.update(p for p in (ROOT/folder).rglob('*') if p.is_file())
    files.update(ROOT/name for name in ['tools/pff_unseen_audit.py','tools/pff_reader.py','tools/skillcorner_reader.py',
        'tools/fetch_pff_training.py','tools/pff_development.py','tools/decoder_development.py','tools/pff_mixed_score.py',
        'tools/quality_audit.py','ref/official/evaluation/possession_metrics.py','docs/experiments/more-data/pff_protocol.json',
        'docs/experiments/unseen-bridge-audit/plan.json','reports/pff_access/browser_complete_listing.json',
        'requirements-temporal.txt','requirements-validation.txt','requirements.txt','submission_manifest.json'])
    known={g[side] for split in ['training','development'] for g in protocol[split] for side in ['home','away']}
    record=dict(frozen_utc=now(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                games=[dict(games[g],teams_seen_in_pff_train_dev=[games[g][s] for s in ['home','away'] if games[g][s] in known]) for g in IDS],
                previously_present_content=False,scope='New match holdout; same provider, one previously seen country per game; no training or selection',
                files_sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted(files)})
    (DOC/'freeze.json').write_text(json.dumps(record,indent=2)+'\n');print('Frozen six unopened matches; no contents downloaded',flush=True)


def check_freeze():
    frozen=json.loads((DOC/'freeze.json').read_text())
    for name,digest in frozen['files_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Frozen source/model/reference policy changed: '+name)
    return frozen


def run(resume=False):
    frozen=check_freeze();_,registered=registry()
    if OUT.exists() and not resume:raise ValueError('Already opened; explicit transport/completion resume required')
    OUT.mkdir(exist_ok=True);marker=OUT/'OPENED.json'
    if marker.exists():
        if json.loads(marker.read_text())['freeze_sha256']!=sha(DOC/'freeze.json'):raise ValueError('Resume freeze mismatch')
    else:marker.write_text(json.dumps(dict(opened_utc=now(),freeze_sha256=sha(DOC/'freeze.json'),purpose='One-shot audit, consumed regardless of outcome'),indent=2)+'\n')
    import bz2
    import numpy as np
    import requests
    from mtah import mixed_candidate,v4,degrade,expanded_team
    from mtah.methods import boost
    from pff_mixed_score import decoder
    from pff_reader import load_geometry,tracking_reference
    from pff_development import spans
    from decoder_development import metric
    from fetch_pff_training import download
    catalog=json.loads((ROOT/'reports/pff_access/browser_complete_listing.json').read_text())['files']
    bundle,state=mixed_candidate.load_bundle(ROOT/'assets/pff-mixed-candidate/bundle.json');assets=v4.load()
    records=[];sources=[]
    for game in IDS:
        done=OUT/(game+'_scores.json')
        if done.exists():
            saved=json.loads(done.read_text());records.extend(saved['records']);sources.append(saved['source']);continue
        start=time.monotonic();raw={}
        for folder,suffix in [('Tracking Data','.jsonl.bz2'),('Metadata','.json'),('Rosters','.json')]:
            relative=folder+'/'+game+suffix;found=[x for x in catalog if x['path']==relative]
            if len(found)!=1:raise ValueError('Missing/duplicate registered source')
            path=ROOT/'external/pff'/relative;path.parent.mkdir(exist_ok=True)
            if not path.exists():
                for attempt in range(1,4):
                    part=path.with_name(path.name+'.part-'+str(time.time_ns()))
                    try:download(found[0],part);part.rename(path);break
                    except requests.RequestException:
                        if attempt==3:raise
                        time.sleep(attempt)
            raw[folder]=path
        meta=json.loads(raw['Metadata'].read_text());meta=meta[0] if isinstance(meta,list) else meta
        with bz2.open(raw['Tracking Data'],'rt') as stream:first=json.loads(next(stream))
        if int(float(first['gameRefId']))!=int(game) or str(meta['id'])!=game or any(meta[s+'Team']['name']!=registered[game][s] for s in ['home','away']):raise ValueError('Unexpected fixture identity')
        data,geometry=load_geometry(raw['Tracking Data'],raw['Metadata'],raw['Rosters']);parts=spans(data['period'])
        dummy=np.where(data['period']>0,0,3).astype(np.int8)
        noisy=degrade.degrade(dict(data,gt=dummy),seed=7,params=expanded_team.BLIND);noisy.pop('gt')
        pred_file=OUT/(game+'_predictions.npz')
        if pred_file.exists():
            with np.load(pred_file) as z:predictions=dict(z)
        else:
            predictions={}
            for scenario,observed in [('natural_detected',data),('label_blind_seed7',noisy)]:
                control,current=mixed_candidate.paired_probabilities(observed,parts,assets,state)
                for name,prob in [('v4',control),('boundary102',current)]:
                    predictions[scenario+'__'+name]=boost.label(*prob,parts,len(dummy),decoder())
            np.savez_compressed(pred_file,**predictions)
        # Predictions and their bytes exist before the reference adapter runs.
        prediction_hash=sha(pred_file);predicted_utc=now()
        gt,alt,reference=tracking_reference(raw['Tracking Data'],data['time'],data['period'],geometry['fps'])
        valid=reference['known_fraction']>=.99 and all(reference['class_frames'][c]>0 for c in ['H','A','D']) and all(np.any(data['period']==p) for p in [1,2])
        receipt=dict(game=game,fixture=registered[game],geometry=geometry,reference=reference,reference_eligibility=bool(valid),
                     predictions_sha256=prediction_hash,predictions_saved_before_reference_utc=predicted_utc,
                     source_files={str(p.relative_to(ROOT)):sha(p) for p in raw.values()})
        if not valid:
            (OUT/(game+'_invalid.json')).write_text(json.dumps(receipt,indent=2)+'\n')
            raise ValueError('Reference ineligible; keep failed match visible, do not silently drop')
        rows=[]
        for reference_name,truth in [('canonical',gt),('sdk_shadow',alt)]:
            for key,pred in predictions.items():
                scenario,name=key.split('__');rows.append(dict(game=game,scenario=scenario,candidate=name,reference=reference_name,**metric(truth,pred)))
        records.extend(rows);sources.append(receipt)
        done.write_text(json.dumps(dict(records=rows,source=receipt),indent=2)+'\n')
        print('Unseen match',game,'complete',round(time.monotonic()-start,1),'seconds',flush=True)
    if len(records)!=48:raise ValueError('Incomplete score grid')
    check_freeze()
    (OUT/'results.json').write_text(json.dumps(dict(freeze_sha256=sha(DOC/'freeze.json'),records=records,source_receipts=sources,
        purpose='Report all outcomes; no model/threshold selection from this audit'),indent=2)+'\n')
    print('Six-game audit complete; no production changes',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run']);p.add_argument('--resume',action='store_true');a=p.parse_args()
    freeze() if a.action=='freeze' else run(a.resume)
