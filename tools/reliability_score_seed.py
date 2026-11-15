"""Evaluate one completed registered seed on CPU while other fits continue."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import reliability_score as common
from threadpoolctl import threadpool_limits
from decoder_development import metric
from check_evaluation import evaluate
import torch


def run(seed):
    out=common.OUT/f'seed{seed}_cpu_results.json'
    if out.exists():raise ValueError('Completed seed scores must not be overwritten')
    if seed not in json.loads(common.PLAN.read_text())['training']['seeds']:raise ValueError('Unregistered seed')
    for fold in range(5):
        if not (common.OUT/'models'/f'seed{seed}_fold{fold}'/'meta.json').exists():raise ValueError('Incomplete seed fits')
    head=common.prior.v3_team();rows=[];params=dict(common.boost_train.decoder_params(),**json.loads((ROOT/'mtah/methods/v2_params.json').read_text())['decoder']);candidate=f'long_seed{seed}_uniform50'
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    with threadpool_limits(1):
        for fold in range(5):
            _,games=common.trial.game_split(f'fold{fold}')
            for game in games:
                family,name=game.split('/');scenarios=['matched_seed0','label_blind_seed1','stress_seed1'] if family=='original' else ['natural_detected','label_blind_seed7']
                for scenario in scenarios:
                    x,xs,meta=common.prior.inputs(game,scenario);bd,bh,gt=common.prior.baseline(game,scenario,head)
                    nd,nh=common.neural(seed,fold,x,xs,meta,game,scenario,'cpu')
                    pred=common.boost.label(.5*bd+.5*nd,.5*bh+.5*nh,meta['full_segments'],len(gt),params)
                    rows.append({'game':game,'match':name,'family':family,'fold':fold,'scenario':scenario,'candidate':candidate,**metric(gt,pred)})
                print(seed,game,'scored on CPU',flush=True)
    payload={'plan_sha256':common.trial.sha(common.trial.PLAN),'research_plan_sha256':common.trial.sha(common.PLAN),'inference':'CPU, two PyTorch threads, one BLAS thread','records':rows};out.write_text(json.dumps(payload,indent=2)+'\n')
    result=evaluate(payload,candidate,common.provenance('long',seed));(common.OUT/f'seed{seed}_cpu_gate.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['status','gain','candidate_composite','game_wins','game_losses']},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seed',type=int,required=True);a=p.parse_args();run(a.seed)
