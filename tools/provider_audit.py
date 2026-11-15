"""Fixed neural-branch source-transfer and train/CV diagnostics; no selection."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import sys
import time

for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import temporal_trial as trial
import evaluate_temporal as prior
from quality_audit import guard_competition_data,sha
from decoder_development import metric
from mtah import temporal
from mtah.methods import boost,boost_train

OUT=ROOT/'reports/strict_fullfold/provider'
PLAN=ROOT/'docs/experiments/strict-fullfold/provider_plan.json'
PROVIDERS=['DFL','Metrica','SkillCorner']


def provider(game):
    return 'DFL' if '/idsse_' in game else 'Metrica' if '/metrica_' in game else 'SkillCorner'


def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2)+'\n')


def split(omitted):
    games=trial.plan()['training_games']
    train=[g for g in games if provider(g)!=omitted]
    test=[g for g in games if provider(g)==omitted]
    if not train or not test or set(train)&set(test):raise ValueError('Invalid source split')
    return train,test


def register():
    if PLAN.exists():raise ValueError('Provider plan already frozen')
    write(PLAN,{
        'registered_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'Neural branch only. Diagnostic, no change to frozen deployment candidate or threshold. Not independent final testing.',
        'hypothesis':'Quantify source shift and train/CV gaps after fixed candidate, without tuning on held-out-provider errors.',
        'seed':20261113,'source_plan_sha256':sha(ROOT/'docs/experiments/source-balance/plan.json'),
        'splits':{p:{'train':split(p)[0],'excluded':split(p)[1]} for p in PROVIDERS},
        'fits':{
            'control':{'steps':1500,'sampler':'equal nonempty original/SkillCorner families, then uniform games/scenarios'},
            'candidate':{'steps':4500,'sampler':'equal nonempty DFL/Metrica/SkillCorner providers, then uniform games/scenarios'},
        },
        'fixed':'Unchanged architecture/BCE/AdamW/missing masks/augmentation/decoder; normalize training games only; no learned inherited branches in this diagnostic.',
        'scores':'Both natural/matched and blind, original stress; standalone neural output. Full-fit train vs wholefold-out on primary natural/matched scenario, report by provider.',
        'interpretation':'Sources have differing event/phase proxies. Descriptive source transfer is not an organizer hidden-match estimate or absence-of-overfit certificate.',
        'organizer_data':'Blocked; PFF final never opened.',
    })
    print('Provider audit registered',flush=True)


def fit(omitted,kind):
    dest=OUT/'models'/f'{omitted}_{kind}'
    if (dest/'meta.json').exists():
        info=json.loads((dest/'meta.json').read_text())
        if info['plan_sha256']!=sha(PLAN) or info['model_sha256']!=sha(dest/'model.pt'):raise ValueError('Provider cache differs')
        print(omitted,kind,'cached',flush=True);return
    plan=json.loads(PLAN.read_text());cfg=trial.plan()['architecture'];train,test=split(omitted)
    seq=trial.sequences(train);mean,scale=trial.normalization(seq)
    if kind=='candidate':groups=[[g for g in train if provider(g)==p] for p in PROVIDERS]
    else:groups=[[g for g in train if g.startswith(f+'/')] for f in ['original','skillcorner']]
    groups=[g for g in groups if g]
    rng=np.random.default_rng(plan['seed']);torch.manual_seed(plan['seed']);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    model=temporal.Temporal().to('mps')
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'],foreach=False)
    started=time.perf_counter();losses=[]
    for step in range(plan['fits'][kind]['steps']):
        bx=[];by=[];bw=[]
        for _ in range(cfg['batch']):
            group=groups[int(rng.integers(len(groups)))];game=group[int(rng.integers(len(group)))];rows=seq[game];row=rows[int(rng.integers(len(rows)))];spans=row['meta']['segments']
            a,b=spans[int(rng.integers(len(spans)))];core=int(rng.integers(a,max(a+1,b-cfg['core_steps']+1)));ix=np.clip(np.arange(core-64,core+320),a,b-1)
            swap=bool(rng.integers(2));x=row['xs' if swap else 'x'][ix];y=np.array(row['gt'][ix],copy=True)
            if swap:y=np.where(y<2,1-y,y)
            w=np.ones(len(ix),np.float32)
            if game.startswith('skillcorner/'):w[y==2]=.25
            y[:64]=3;y[320:]=3
            bx.append(temporal.normalize(x,mean,scale).T);by.append(y);bw.append(w)
        x=torch.from_numpy(np.stack(bx)).to('mps');y=torch.from_numpy(np.stack(by).astype(np.int64)).to('mps');w=torch.from_numpy(np.stack(bw)).to('mps')
        opt.zero_grad(set_to_none=True);loss=temporal.loss(model(x),y,w);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();losses.append(float(loss.detach().cpu()))
        if (step+1)%1500==0:print(omitted,kind,step+1,'loss',float(np.mean(losses[-100:])),flush=True)
    dest.mkdir(parents=True,exist_ok=True)
    torch.save({'state_dict':model.cpu().state_dict(),'mean':torch.from_numpy(mean),'scale':torch.from_numpy(scale)},dest/'model.pt')
    write(dest/'meta.json',{'plan_sha256':sha(PLAN),'train':train,'excluded':test,'omitted_provider':omitted,'kind':kind,
                          'model_sha256':sha(dest/'model.pt'),'steps':plan['fits'][kind]['steps'],'seed':plan['seed'],
                          'training_loss_last100':float(np.mean(losses[-100:])),'seconds':time.perf_counter()-started})


def load(folder,train,test):
    info=json.loads((folder/'meta.json').read_text())
    if set(info['train'])!=set(train) or set(info['excluded'])!=set(test) or sha(folder/'model.pt')!=info['model_sha256']:
        raise ValueError('Provider/full model training mismatch')
    state=torch.load(folder/'model.pt',map_location='cpu',weights_only=True)
    model=temporal.Temporal();model.load_state_dict(state['state_dict']);model.eval()
    return model,state['mean'].numpy(),state['scale'].numpy()


def predict(states,x,xs,meta):
    model,mean,scale=states
    pd,ph=temporal.probabilities(model,x,xs,meta['segments'],mean,scale,device='cpu')
    return prior.upsample(pd,ph,meta)


def score():
    out=OUT/'results.json'
    if out.exists():raise ValueError('Do not overwrite completed provider diagnostic')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    params=dict(boost_train.decoder_params(),**json.loads((ROOT/'mtah/methods/v2_params.json').read_text())['decoder'])
    records=[];in_sample=[]
    with threadpool_limits(1):
        for omitted in PROVIDERS:
            train,test=split(omitted)
            models={kind:load(OUT/'models'/f'{omitted}_{kind}',train,test) for kind in ['control','candidate']}
            for game in test:
                name=game.split('/')[1];family=game.split('/')[0]
                scenarios=['matched_seed0','label_blind_seed1','stress_seed1'] if family=='original' else ['natural_detected','label_blind_seed7']
                for scenario in scenarios:
                    x,xs,meta=prior.inputs(game,scenario)
                    with np.load(ROOT/'reports/v2_search'/('known' if family=='original' else 'skillcorner')/f'{name}__{scenario}__probs.npz') as z:gt=z['gt']
                    for kind,states in models.items():
                        pd,ph=predict(states,x,xs,meta);pred=boost.label(pd,ph,meta['full_segments'],len(gt),params)
                        records.append({'game':game,'omitted_provider':omitted,'kind':kind,'scenario':scenario,**metric(gt,pred)})
                print('provider',omitted,game,'scored',flush=True)
        allgames=trial.plan()['training_games']
        full=load(ROOT/'reports/source_balance/models/seed20261113_foldfull',allgames,[])
        for game in allgames:
            family,name=game.split('/');scenario='matched_seed0' if family=='original' else 'natural_detected'
            x,xs,meta=prior.inputs(game,scenario)
            with np.load(ROOT/'reports/v2_search'/('known' if family=='original' else 'skillcorner')/f'{name}__{scenario}__probs.npz') as z:gt=z['gt']
            for scope in ['full_fit_training','wholefold_out']:
                if scope=='full_fit_training':pd,ph=predict(full,x,xs,meta)
                else:
                    fold=trial.plan()['folds'][game]
                    folder=ROOT/'reports/source_balance/predictions'/f'seed20261113_fold{fold}'
                    with np.load(folder/f'{trial.stem(game)}__{scenario}.npz') as z:pd,ph=z['pd'],z['ph']
                pred=boost.label(pd,ph,meta['full_segments'],len(gt),params)
                in_sample.append({'game':game,'provider':provider(game),'scope':scope,'scenario':scenario,**metric(gt,pred)})
        write(out,{'plan_sha256':sha(PLAN),'neural_branch_only':True,'fresh_final_test':False,'records':records,'train_vs_cv':in_sample})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['register','train','score']);a=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if a.mode=='register':register()
    else:
        guard_competition_data()
        if a.mode=='train':
            for omitted in PROVIDERS:
                for kind in ['control','candidate']:fit(omitted,kind)
        else:score()
