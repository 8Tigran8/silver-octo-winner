"""Select exactly one registered model after complete development/regression gates."""
import json
from pathlib import Path
import shutil
import sys
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pff_mixed_train as train
from pff_reader import sha


def main():
    dest=train.OUT/'selection.json'
    if dest.exists():raise ValueError('Do not overwrite a frozen selection')
    if any((ROOT/'reports'/folder/'OPENED.json').exists() for folder in ['pff_final','pff_mixed_final']):raise ValueError('Final already consumed; no candidate reselection')
    plan=train.plan();old=json.loads((train.OUT/'old_results.json').read_text());pff=json.loads((train.OUT/'pff_dev_results.json').read_text());reused=json.loads((train.OUT/'reused_results.json').read_text())
    if any(p['mixed_plan_sha256']!=sha(train.PLAN) for p in [old,pff,reused]):raise ValueError('Results use another registered plan')
    baseline={(r['game'],r['scenario']):r for r in old['records'] if r['candidate']=='strict_v4'};decisions={};eligible=[]
    for spec in plan['candidates']:
        name=spec['name'];history=old['decisions'][name]['historical'];paired=old['decisions'][name]['paired'];dev=pff['decisions'][name];regression=reused['decisions'][name]
        checks={'old_historical':all(history['checks'].values()),'old_strict_paired':all(paired['checks'].values()),
                'PFFdev':all(dev['checks'].values()),'reused_external':all(regression['checks'].values())}
        providers={}
        for source in ['DFL','Metrica','SkillCorner']:
            rows=[r for r in old['records'] if r['candidate']==name and
                (source=='DFL' and '/idsse_' in r['game'] and r['scenario']=='matched_seed0' or
                 source=='Metrica' and '/metrica_' in r['game'] and r['scenario']=='matched_seed0' or
                 source=='SkillCorner' and r['family']=='skillcorner' and r['scenario']=='natural_detected')]
            providers[source]=float(np.mean([r['spell_f1_mean']-baseline[r['game'],r['scenario']]['spell_f1_mean'] for r in rows]));checks['provider_'+source]=providers[source]>=-.01
        score=.5*(history['gain']+dev['comparisons']['canonical/natural_detected']['gain'])
        decisions[name]={'checks':checks,'eligible':all(checks.values()),'selection_score':score,'historical_gain':history['gain'],
            'strict_paired_gain':paired['gain'],'PFFdev_gain':dev['comparisons']['canonical/natural_detected']['gain'],'provider_gains':providers}
        if all(checks.values()):eligible.append(name)
    eligible.sort(key=lambda name:(-decisions[name]['selection_score'],name!='mixed25'))
    result={'registered_plan_sha256':sha(train.PLAN),'decided_utc':datetime.now(timezone.utc).isoformat(),'decisions':decisions,
        'candidate':eligible[0] if eligible else None,'status':'ELIGIBLE_FOR_UNOPENED_FINAL' if eligible else 'KEEP_V4: no candidate passed all development requirements',
        'scope':'Reused old29 and11regression games plus tenPFFdevelopment games; not final confirmation','final_opened':False,
        'result_sha256':{file:sha(train.OUT/file) for file in ['old_results.json','old_provenance.json','pff_dev_results.json','reused_results.json']}}
    if eligible:
        name=eligible[0];state,proof=train.load(name,'full');folder=ROOT/'assets/pff-mixed-candidate'
        if folder.exists():raise ValueError('Preserve existing candidate assets')
        folder.mkdir();source=train.OUT/'models'/(name+'_foldfull');shutil.copy2(source/'model.pt',folder/'model.pt');shutil.copy2(source/'meta.json',folder/'meta.json')
        bundle={'candidate':name,'spec':proof['spec'],'training_games':proof['train'],'training_plan_sha256':sha(train.PLAN),
            'model_path':'model.pt','model_sha256':sha(folder/'model.pt'),'metadata_path':'meta.json','metadata_sha256':sha(folder/'meta.json'),
            'baseline_asset_manifest_sha256':sha(ROOT/'assets/v4/manifest.json'),'normalizer_checked_train_only':True,
            'rule':'Fixed50/50 v3/newtemporal probabilities; unchanged cost9decoder; no PFFdev/final fitting',
            'release_status':'FROZEN CANDIDATE; final quality and exact-file release checks required'}
        (folder/'bundle.json').write_text(json.dumps(bundle,indent=2)+'\n');result['bundle_sha256']=sha(folder/'bundle.json')
    dest.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
