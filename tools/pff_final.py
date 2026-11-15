"""Freeze once, then open the original twelve-game PFF reserve exactly once.

No final data access is allowed before a successful freeze. A started run burns
the reserve even when downloading, conversion or scoring fails. No resumptions,
alternative candidates or score-selected reference construction are supported.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pff_reader import sha

OUT=ROOT/'reports/pff_final'
FREEZE=ROOT/'docs/experiments/pff/final_freeze.json'


def now():return datetime.now(timezone.utc).isoformat()


def final_ids():
    policy=json.loads((ROOT/'docs/evaluation/policy.json').read_text())
    return policy,[str(g) for g in policy['reserve']['final_ids']]


def paths():
    files=list((ROOT/'mtah').rglob('*.py'))+list((ROOT/'mtah').rglob('*.json'))+list((ROOT/'tools').glob('pff_*.py'))
    files+=[ROOT/'tools/decoder_development.py',ROOT/'tools/quality_audit.py',ROOT/'ref/official/evaluation/possession_metrics.py',
        ROOT/'docs/evaluation/policy.json',ROOT/'docs/experiments/more-data/pff_protocol.json',ROOT/'docs/experiments/pff/plan_v2.json',
        ROOT/'docs/experiments/strict-fullfold/decision.json',ROOT/'reports/pff_development/results.json',
        ROOT/'reports/pff_development/equivalence_train3816.json',ROOT/'assets/research-balanced/bundle.json']
    for folder,manifest_name in [('assets/v4','manifest.json'),('assets/research-balanced','bundle.json')]:
        directory=ROOT/folder;manifest=directory/manifest_name;files.append(manifest);m=json.loads(manifest.read_text())
        if manifest_name=='manifest.json':files.extend(directory/item['path'] for item in m['files'])
        else:
            for item in m['models']:files.extend([directory/item['path'],directory/item['metadata_path']])
    return sorted(set(files))


def freeze():
    from check_evaluation import load_inputs
    load_inputs()
    if FREEZE.exists():raise ValueError('Original final freeze already exists')
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True):raise ValueError('Commit exact source before freezing')
    dev=json.loads((ROOT/'reports/pff_development/results.json').read_text())
    from pff_development import summarize
    protocol=json.loads((ROOT/'docs/experiments/more-data/pff_protocol.json').read_text());games=[g['game_id'] for g in protocol['development']]
    decision=summarize(dev['records'],games)
    if not all(decision['checks'].values()):raise ValueError('Failed PFF development gate: final remains unopened')
    if dev['tool_sha256']!=sha(ROOT/'tools/pff_development.py') or dev['plan_sha256']!=sha(ROOT/'docs/experiments/pff/plan_v2.json'):
        raise ValueError('Development source changed')
    strict=json.loads((ROOT/'docs/experiments/strict-fullfold/decision.json').read_text())
    if strict['all_stage_wholefold_isolation']!='PASS' or not all(strict['strict_paired']['checks'].values()) or not all(strict['against_historical_frozen_v4']['checks'].values()):
        raise ValueError('Prior quality/isolation gates failed')
    policy,ids=final_ids()
    for game in ids:
        for folder in ['Tracking','Tracking Data','Metadata','Rosters','Event Data','converted']:
            for suffix in ['.json','.jsonl.bz2','_geometry.npz']:
                if (ROOT/'external/pff'/folder/(game+suffix)).exists():raise ValueError('Final content already exists before freeze')
        if (ROOT/'reports/pff_access/references'/('kloppy_'+game+'.npz')).exists():raise ValueError('Final reference already exists')
    files={str(p.relative_to(ROOT)):sha(p) for p in paths()}
    result={'frozen_utc':now(),'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'candidate':'source_seed20261113','control':'v4','files_sha256':files,'final_ids':ids,
        'scenarios':policy['reserve']['final_gate']['scenarios'],'gate':policy['reserve']['final_gate'],
        'reference':'Pinned kloppy canonical and fixed SDK-shadow sensitivity; >=99% known and H/A/D in every match',
        'final_contents_previously_opened':False,'one_shot':'Failure also consumes the reserve; no post-test model replacement'}
    FREEZE.write_text(json.dumps(result,indent=2)+'\n');print('Frozen; final content remains unopened',flush=True)


def final_decision(records,ids,rules):
    from pff_development import summarize
    # Reuse the full-grid calculation; apply original final thresholds below.
    summary=summarize(records,ids)['comparisons'];natural=summary['canonical/natural_detected'];blind=summary['canonical/label_blind_seed7']
    groups={name:sum(natural['per_game_gain'][g] for g in games)/len(games) for name,games in rules['groups'].items()}
    checks={'natural_gain':natural['gain']>=rules['min_natural_mean_spell_gain'],
        'wins':natural['wins']>=rules['min_natural_game_wins'],
        'macro_guard':natural['macro_gain']>=-rules['max_natural_macro_loss'],
        'blind_guard':blind['gain']>=-rules['max_blind_mean_spell_loss'],
        **{'group_'+k:v>0 for k,v in groups.items()},
        **{'sensitivity_'+s:summary['sdk_shadow/'+s]['gain']>=-.01 for s in ['natural_detected','label_blind_seed7']}}
    import numpy as np
    gains=np.array(list(natural['per_game_gain'].values()));rng=np.random.default_rng(20261113)
    interval=np.quantile(gains[rng.integers(0,len(ids),(20000,len(ids)))].mean(axis=1),[.025,.975]).tolist()
    return {'status':'PASS: frozen final quality gates' if all(checks.values()) else 'FAIL: keep verified v4; no post-test selection',
        'checks':checks,'comparisons':summary,'group_gains':groups,'descriptive_paired_game_95ci':interval,
        'limits':'Event-reference proxy; shared national teams/two groups; not universal generalization proof. Reserve consumed.'}


def run():
    if not FREEZE.exists():raise ValueError('Freeze before accessing any final content')
    frozen=json.loads(FREEZE.read_text())
    for path,digest in frozen['files_sha256'].items():
        if sha(ROOT/path)!=digest:raise ValueError('Frozen file changed: '+path)
    OUT.mkdir(exist_ok=True);start=OUT/'OPENED.json'
    if start.exists():raise ValueError('Reserve already consumed; no rerun or runner-up permitted')
    start.write_text(json.dumps({'opened_utc':now(),'freeze_sha256':sha(FREEZE),'status':'CONSUMED even if subsequent steps fail'},indent=2)+'\n')
    # Imports and all content downloads occur only after the once-only marker.
    import gdown
    import numpy as np
    import torch
    from mtah import v4,research_candidate,degrade,expanded_team
    from mtah.methods import boost,boost_train
    from pff_reader import load_geometry,tracking_reference
    from pff_development import paired,spans,BUNDLE
    from decoder_development import metric
    files=json.loads((ROOT/'reports/pff_access/browser_complete_listing.json').read_text())['files']
    ids=frozen['final_ids'];records=[];receipts=[];torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    protocol=json.loads((ROOT/'docs/experiments/more-data/pff_protocol.json').read_text())
    registered={g['game_id']:g for g in protocol['final_test']}
    bundle,states=research_candidate.load_bundle(BUNDLE);assets=v4.load()
    params=dict(boost_train.decoder_params(),**json.loads((ROOT/'mtah/methods/v2_params.json').read_text())['decoder'])
    try:
        for game in ids:
            raw={}
            for folder,suffix in [('Tracking Data','.jsonl.bz2'),('Metadata','.json'),('Rosters','.json')]:
                relative=folder+'/'+game+suffix;entry=[f for f in files if f['path']==relative]
                if len(entry)!=1:raise ValueError('Missing/duplicate public file metadata: '+relative)
                path=ROOT/'external/pff'/relative
                if path.exists():raise ValueError('Unexpected existing reserve content')
                path.parent.mkdir(exist_ok=True);gdown.download(id=entry[0]['id'],output=str(path),quiet=True,use_cookies=False);raw[folder]=path
            import bz2
            metadata=json.loads(raw['Metadata'].read_text());metadata=metadata[0] if isinstance(metadata,list) else metadata
            with bz2.open(raw['Tracking Data'],'rt') as stream:first=json.loads(next(stream))
            if float(first['gameRefId'])!=int(game) or str(metadata['id'])!=game:
                raise ValueError('Source game identity mismatch')
            if metadata['homeTeam']['name']!=registered[game]['home'] or metadata['awayTeam']['name']!=registered[game]['away']:
                raise ValueError('Source teams differ from the original country-isolated split')
            data,geometry=load_geometry(raw['Tracking Data'],raw['Metadata'],raw['Rosters'])
            # Reference construction/source checks are frozen and do not select models.
            gt,alt,reference=tracking_reference(raw['Tracking Data'],data['time'],data['period'],geometry['fps'])
            if reference['known_fraction']<.99 or any(reference['class_frames'][c]<=0 for c in ['H','A','D']) or not all(np.any(data['period']==p) for p in [1,2]):
                raise ValueError('Reserve full-task source invalid; cannot drop match or change endpoint')
            segments=spans(data['period']);dummy=np.where(data['period']>0,0,3).astype(np.int8)
            corrupt=degrade.degrade(dict(data,gt=dummy),seed=7,params=expanded_team.BLIND);corrupt.pop('gt');predictions={}
            for scenario,observed in [('natural_detected',data),('label_blind_seed7',corrupt)]:
                control,candidate=paired(observed,segments,assets,states)
                for name,probs in [('v4',control),('source_seed20261113',candidate)]:
                    predictions[scenario+'__'+name]=boost.label(*probs,segments,len(dummy),params)
            pred_file=OUT/(game+'_predictions.npz');np.savez_compressed(pred_file,**predictions)
            for ref_name,truth in [('canonical',gt),('sdk_shadow',alt)]:
                for name,pred in predictions.items():
                    scenario,candidate=name.split('__');records.append({'game':game,'scenario':scenario,'candidate':candidate,'reference':ref_name,**metric(truth,pred)})
            receipt={'game':game,'geometry':geometry,'reference':reference,'prediction_sha256':sha(pred_file)};receipts.append(receipt)
            (OUT/(game+'_scores.json')).write_text(json.dumps({'records':records[-8:],'source':receipt},indent=2)+'\n');print(game,'final comparison complete',flush=True)
        result=final_decision(records,ids,frozen['gate'])
        (OUT/'results.json').write_text(json.dumps({'freeze_sha256':sha(FREEZE),'records':records,'source_receipts':receipts,'decision':result},indent=2)+'\n');print(json.dumps(result),flush=True)
    except Exception as error:
        (OUT/'FAILURE.json').write_text(json.dumps({'status':'FAIL; reserve consumed','reason':str(error),'completed_games':len(receipts),'records':records},indent=2)+'\n');raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['freeze','run']);a=p.parse_args()
    freeze() if a.action=='freeze' else run()
