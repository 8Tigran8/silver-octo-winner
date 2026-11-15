"""Predeclared old/new/OOD blending on reused complete game exclusions."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from mtah import robust_temporal
import reliability_score as common
from check_evaluation import evaluate
from decoder_development import metric
PLAN=ROOT/'docs/experiments/robust-followup/plan.json';OUT=ROOT/'reports/robust_followup'


def score(seed=20261115,specs=None,suffix='development'):
    out=OUT/(suffix+'_results.json')
    if out.exists():raise ValueError('Do not overwrite completed followup')
    plan=json.loads(PLAN.read_text());specs=specs or plan['rules'];head=common.prior.v3_team();rows=[]
    frozen={(r['game'],r['scenario']):r for r in json.loads((ROOT/'docs/experiments/temporal/cv_results.json').read_text())['records'] if r['candidate']=='fusion50'}
    long={(r['game'],r['scenario']):r for r in json.loads((ROOT/'reports/reliability/long_results.json').read_text())['records'] if r['candidate']==f'long_seed{seed}_uniform50'}
    params=dict(common.boost_train.decoder_params(),**json.loads((ROOT/'mtah/methods/v2_params.json').read_text())['decoder'])
    with threadpool_limits(1):
        for fold in range(5):
            state=torch.load(ROOT/'reports/reliability/models'/f'seed{seed}_fold{fold}'/'model.pt',map_location='cpu',weights_only=True);mean,scale=state['mean'].numpy(),state['scale'].numpy()
            _,games=common.trial.game_split(f'fold{fold}')
            for game in games:
                family,name=game.split('/');scenarios=['matched_seed0','label_blind_seed1','stress_seed1'] if family=='original' else ['natural_detected','label_blind_seed7']
                for scenario in scenarios:
                    x,xs,meta=common.prior.inputs(game,scenario);bd,bh,gt=common.prior.baseline(game,scenario,head)
                    with np.load(common.trial.OUT/'predictions'/f'fold{fold}'/f'{common.trial.stem(game)}__{scenario}.npz') as z:od,oh=z['pd'],z['ph']
                    with np.load(ROOT/'reports/reliability/predictions'/f'seed{seed}_fold{fold}'/f'{common.trial.stem(game)}__{scenario}.npz') as z:nd,nh=z['pd'],z['ph']
                    for spec in specs:
                        pd,ph=robust_temporal.combine(spec,od,oh,nd,nh,x,mean,scale,meta)
                        pred=common.boost.label(.5*bd+.5*pd,.5*bh+.5*ph,meta['full_segments'],len(gt),params);result=metric(gt,pred)
                        reference=frozen[game,scenario] if spec['name']=='old_control' else long[game,scenario] if spec['name']=='long_control' else None
                        if reference and any(abs(result[k]-reference[k])>1e-12 for k in ['spell_f1_mean','macro_f1']):raise ValueError('Old/long control drift')
                        rows.append({'game':game,'match':name,'family':family,'fold':fold,'scenario':scenario,'candidate':'robust_'+spec['name'],**result})
                print(seed,game,'followup complete',flush=True)
    payload={'plan_sha256':common.trial.sha(common.trial.PLAN),'followup_plan_sha256':common.trial.sha(PLAN),'records':rows};out.write_text(json.dumps(payload,indent=2)+'\n');summary={}
    for spec in specs:
        candidate='robust_'+spec['name'];result=evaluate(payload,candidate,common.provenance('long',seed));(OUT/(suffix+'_'+candidate+'_gate.json')).write_text(json.dumps(result,indent=2)+'\n')
        summary[candidate]={k:result[k] for k in ['status','candidate_composite','gain','descriptive_95ci','game_wins','game_losses']}
        print(candidate,result['status'],round(result['gain'],5),result['game_wins'],flush=True)
    (OUT/(suffix+'_summary.json')).write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':score()
