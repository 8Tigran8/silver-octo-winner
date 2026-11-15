"""Local, predeclared longer TCN fits on the frozen whole-game folds."""
import argparse,json,time,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import temporal_trial as trial
import numpy as np
import torch
PLAN=ROOT/'docs/experiments/reliability/plan.json'
OUT=ROOT/'reports/reliability'


def fit(fold,seed,device):
    p=json.loads(PLAN.read_text());cfg=trial.plan()['architecture']
    if seed not in p['training']['seeds'] or fold not in range(5):raise ValueError('Unregistered fit')
    key=f'seed{seed}_fold{fold}';dest=OUT/'models'/key;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'meta.json').exists():
        meta=json.loads((dest/'meta.json').read_text())
        if meta['research_plan_sha256']!=trial.sha(PLAN) or meta['model_sha256']!=trial.sha(dest/'model.pt'):raise ValueError('Stale or changed fit')
        print(key,'cached verified',flush=True);return
    train,test=trial.game_split(f'fold{fold}');seq=trial.sequences(train);mean,scale=trial.normalization(seq)
    if set(train)&set(test):raise ValueError('Fold leak')
    torch.manual_seed(seed);rng=np.random.default_rng(seed);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    model=trial.temporal.Temporal().to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'],foreach=False)
    groups=[[g for g in train if g.startswith(family+'/')] for family in ['original','skillcorner']]
    started=time.perf_counter();losses=[]
    for step in range(p['training']['steps']):
        bx=[];by=[];bw=[]
        for _ in range(cfg['batch']):
            group=groups[int(rng.integers(len(groups)))];game=group[int(rng.integers(len(group)))];rows=seq[game];row=rows[int(rng.integers(len(rows)))];spans=row['meta']['segments']
            a,b=spans[int(rng.integers(len(spans)))];core=int(rng.integers(a,max(a+1,b-cfg['core_steps']+1)));ix=np.clip(np.arange(core-64,core+320),a,b-1)
            swap=bool(rng.integers(2));x=row['xs' if swap else 'x'][ix];y=np.array(row['gt'][ix],copy=True)
            if swap:y=np.where(y<2,1-y,y)
            w=np.ones(len(ix),np.float32)
            if game.startswith('skillcorner/'):w[y==2]=.25
            y[:64]=3;y[320:]=3
            bx.append(trial.temporal.normalize(x,mean,scale).T);by.append(y);bw.append(w)
        x=torch.from_numpy(np.stack(bx)).to(device);y=torch.from_numpy(np.stack(by).astype(np.int64)).to(device);w=torch.from_numpy(np.stack(bw)).to(device)
        opt.zero_grad(set_to_none=True);loss=trial.temporal.loss(model(x),y,w);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();losses.append(float(loss.detach().cpu()))
        if (step+1)%500==0:print(key,step+1,'loss',round(float(np.mean(losses[-100:])),5),'seconds',round(time.perf_counter()-started,1),flush=True)
    torch.save({'state_dict':model.cpu().state_dict(),'mean':torch.from_numpy(mean),'scale':torch.from_numpy(scale)},dest/'model.pt')
    meta={'train':train,'excluded':test,'plan_sha256':trial.sha(trial.PLAN),'research_plan_sha256':trial.sha(PLAN),'device':device,'seed':seed,'training_steps':p['training']['steps'],'seconds':time.perf_counter()-started,'model_sha256':trial.sha(dest/'model.pt'),'training_loss_last100':float(np.mean(losses[-100:]))}
    (dest/'meta.json').write_text(json.dumps(meta,indent=2)+'\n');print(key,'complete',flush=True)
    del model,opt,seq


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--device',default='mps');args=parser.parse_args()
    for seed in json.loads(PLAN.read_text())['training']['seeds']:
        for fold in range(5):fit(fold,seed,args.device)
