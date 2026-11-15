"""Score the separately registered spell-aware family, using shared checked I/O."""
import argparse,json,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import reliability_score as common
from decoder_development import metric
from check_evaluation import evaluate


def run(device):
    common.PLAN=ROOT/'docs/experiments/reliability/spell_plan.json';common.OUT=ROOT/'reports/spell_trial'
    out=common.OUT/'results.json'
    if out.exists():raise ValueError('Completed scores must not be overwritten')
    plan=json.loads(common.PLAN.read_text());rows=[];head=common.prior.v3_team()
    params=dict(common.boost_train.decoder_params(),**json.loads((ROOT/'mtah/methods/v2_params.json').read_text())['decoder'])
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    with threadpool_limits(1):
        for fold in range(5):
            _,games=common.trial.game_split(f'fold{fold}')
            for game in games:
                family,name=game.split('/');scenarios=['matched_seed0','label_blind_seed1','stress_seed1'] if family=='original' else ['natural_detected','label_blind_seed7']
                for scenario in scenarios:
                    x,xs,meta=common.prior.inputs(game,scenario);bd,bh,gt=common.prior.baseline(game,scenario,head)
                    pairs=[common.neural(s,fold,x,xs,meta,game,scenario,device) for s in plan['training']['seeds']]
                    candidates=[(f'spell_seed{s}',pair[0],pair[1]) for s,pair in zip(plan['training']['seeds'],pairs)]
                    candidates.append(('spell_ensemble',np.mean([p[0] for p in pairs],axis=0),np.mean([p[1] for p in pairs],axis=0)))
                    for candidate,nd,nh in candidates:
                        pred=common.boost.label(.5*bd+.5*nd,.5*bh+.5*nh,meta['full_segments'],len(gt),params)
                        rows.append({'game':game,'match':name,'family':family,'fold':fold,'scenario':scenario,'candidate':candidate,**metric(gt,pred)})
                print('spell',game,'complete',flush=True)
    payload={'plan_sha256':common.trial.sha(common.trial.PLAN),'research_plan_sha256':common.trial.sha(common.PLAN),'records':rows}
    out.write_text(json.dumps(payload,indent=2)+'\n');summary={}
    for candidate in sorted({r['candidate'] for r in rows}):
        seed=int(candidate[len('spell_seed'):]) if candidate.startswith('spell_seed') else None
        proof=common.provenance('long',seed);result=evaluate(payload,candidate,proof)
        (common.OUT/f'{candidate}_gate.json').write_text(json.dumps(result,indent=2)+'\n')
        summary[candidate]={k:result[k] for k in ['status','candidate_composite','gain','descriptive_95ci','game_wins','game_losses','large_game_losses']}
        print(candidate,result['status'],'gain',round(result['gain'],5),'wins',result['game_wins'],'failed',[k for k,v in result['checks'].items() if not v],flush=True)
    (common.OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',default='mps');a=p.parse_args();run(a.device)
