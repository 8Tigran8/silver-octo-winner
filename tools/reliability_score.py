"""Score the registered local adaptive and longer-TCN studies; no target data."""
import argparse,json,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from mtah import reliability,temporal
from mtah.methods import boost,boost_train
import temporal_trial as trial
import evaluate_temporal as prior
from decoder_development import metric
from check_evaluation import evaluate
PLAN=ROOT/'docs/experiments/reliability/plan.json';OUT=ROOT/'reports/reliability'


def gate_features(x,meta,columns):
    # Feature schema is shared with the frozen 108-column temporal input.
    gap=x[:,columns.index('gap')];visible=x[:,columns.index('seen_w151')]
    return prior.upsample(gap,visible,meta)


def neural(seed,fold,x,xs,meta,game,scenario,device):
    if device!='cpu':raise ValueError('New-trial inference is fixed to CPU for consistent cached scoring')
    key=f'seed{seed}_fold{fold}';folder=OUT/'models'/key;info=json.loads((folder/'meta.json').read_text())
    train,test=trial.game_split(f'fold{fold}')
    if info['train']!=train or info['excluded']!=test or info['research_plan_sha256']!=trial.sha(PLAN) or info['model_sha256']!=trial.sha(folder/'model.pt'):raise ValueError('Wrong neural fold or model')
    path=OUT/'predictions'/key/f'{trial.stem(game)}__{scenario}.npz';path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        with np.load(path) as z:
            if str(z['model_sha256'])!=info['model_sha256']:raise ValueError('Stale predictions')
            return z['pd'],z['ph']
    state=torch.load(folder/'model.pt',map_location='cpu',weights_only=True);model=temporal.Temporal();model.load_state_dict(state['state_dict']);model.eval()
    pd,ph=temporal.probabilities(model,x,xs,meta['segments'],state['mean'].numpy(),state['scale'].numpy(),device=device)
    pd,ph=prior.upsample(pd,ph,meta);np.savez(path,pd=pd,ph=ph,model_sha256=info['model_sha256']);return pd,ph


def provenance(stage,seed):
    if stage=='adaptive':return json.loads((ROOT/'docs/experiments/temporal/training_provenance.json').read_text())
    result={}
    seeds=json.loads(PLAN.read_text())['training']['seeds'] if seed is None else [seed]
    for fold in range(5):
        metas=[json.loads((OUT/'models'/f'seed{s}_fold{fold}'/'meta.json').read_text()) for s in seeds]
        result[f'fold{fold}']=dict(metas[0],ensemble_seeds=seeds,component_sha256=[m['model_sha256'] for m in metas])
        if len(metas)>1:result[f'fold{fold}']['model_sha256']=hashlib.sha256(json.dumps(result[f'fold{fold}']['component_sha256']).encode()).hexdigest()
    return result


def score(stage,device):
    path=OUT/f'{stage}_results.json'
    if path.exists():raise ValueError('Completed scores must not be overwritten')
    plan=json.loads(PLAN.read_text());rules=plan['rules'];rows=[]
    feature=trial.read_feature(trial.AUDIT,'idsse_J03WMX','matched_seed0');columns=feature['cols'].tolist();del feature
    head=prior.v3_team();params=dict(boost_train.decoder_params(),**json.loads((ROOT/'mtah/methods/v2_params.json').read_text())['decoder'])
    expected={(r['game'],r['scenario']):r for r in json.loads((ROOT/'docs/experiments/temporal/cv_results.json').read_text())['records'] if r['candidate']=='fusion50'}
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    with threadpool_limits(1):
        for fold in range(5):
            _,games=trial.game_split(f'fold{fold}')
            for game in games:
                family,name=game.split('/');scenarios=['matched_seed0','label_blind_seed1','stress_seed1'] if family=='original' else ['natural_detected','label_blind_seed7']
                for scenario in scenarios:
                    x,xs,meta=prior.inputs(game,scenario);bd,bh,gt=prior.baseline(game,scenario,head);gap,visibility=gate_features(x,meta,columns)
                    if stage=='adaptive':
                        with np.load(trial.OUT/'predictions'/f'fold{fold}'/f'{trial.stem(game)}__{scenario}.npz') as z:nd,nh=z['pd'],z['ph']
                        candidates=[(s['name'],s,nd,nh) for s in rules]
                    else:
                        pairs=[neural(s,fold,x,xs,meta,game,scenario,device) for s in plan['training']['seeds']]
                        candidates=[(f'long_seed{s}_uniform50',rules[0],pair[0],pair[1]) for s,pair in zip(plan['training']['seeds'],pairs)]
                        nd=np.mean([p[0] for p in pairs],axis=0);nh=np.mean([p[1] for p in pairs],axis=0)
                        candidates += [('long_ensemble_'+s['name'],s,nd,nh) for s in rules if s['name'] in ['uniform50','gap5','visibility6','confidence']]
                    for candidate,spec,nd,nh in candidates:
                        pd,ph=reliability.mix(spec,bd,bh,nd,nh,gap,visibility)
                        pred=boost.label(pd,ph,meta['full_segments'],len(gt),params);result=metric(gt,pred)
                        if stage=='adaptive' and candidate=='uniform50' and any(abs(result[k]-expected[game,scenario][k])>1e-12 for k in ['spell_f1_mean','macro_f1']):raise ValueError('Frozen v4 control changed')
                        rows.append({'game':game,'match':name,'family':family,'fold':fold,'scenario':scenario,'candidate':candidate,**result})
                print(stage,game,'complete',flush=True)
    payload={'plan_sha256':trial.sha(trial.PLAN),'research_plan_sha256':trial.sha(PLAN),'v4_exact_control':stage=='adaptive','records':rows}
    path.write_text(json.dumps(payload,indent=2)+'\n');decisions={}
    for candidate in sorted({r['candidate'] for r in rows}):
        seed=int(candidate.split('_')[1][4:]) if candidate.startswith('long_seed') else None
        proof=provenance(stage,seed);result=evaluate(payload,candidate,proof);decisions[candidate]=result
        (OUT/f'{candidate}_gate.json').write_text(json.dumps(result,indent=2)+'\n')
        print(candidate,result['status'],'gain',round(result['gain'],5),'wins',result['game_wins'],'failed',[k for k,v in result['checks'].items() if not v],flush=True)
    summary={name:{k:r[k] for k in ['status','candidate_composite','gain','descriptive_95ci','game_wins','game_losses','large_game_losses']} for name,r in decisions.items()}
    (OUT/f'{stage}_summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['adaptive','long']);parser.add_argument('--device',default='cpu');args=parser.parse_args();score(args.stage,args.device)
