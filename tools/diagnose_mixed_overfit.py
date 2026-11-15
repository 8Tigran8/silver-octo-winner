"""Post-selection full-fit/wholefold-out neural diagnostics, never model tuning."""
import json
import os
from pathlib import Path
import sys
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import temporal_trial as trial
import pff_mixed_train as train
from pff_mixed_score import neural,decoder
from decoder_development import metric
from mtah import mixed_candidate
from mtah.methods import boost
from pff_reader import sha


def main():
    dest=train.OUT/'train_cv_diagnostic.json'
    if dest.exists():raise ValueError('Preserve completed diagnostic')
    selected=json.loads((train.OUT/'selection.json').read_text())['candidate'];bundle,state=mixed_candidate.load_bundle(ROOT/'assets/pff-mixed-candidate/bundle.json');rows=[];params=decoder()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    for fold in range(5):
        out_state,_=train.load(selected,fold);_,games=trial.game_split('fold'+str(fold))
        for game in games:
            scenario='matched_seed0' if game.startswith('original/') else 'natural_detected';source=trial.folder(game,scenario);meta=json.loads((source/'meta.json').read_text());x=np.load(source/'x.npy',mmap_mode='r');xs=np.load(source/'xs.npy',mmap_mode='r');parts=meta['full_segments']
            if game.startswith('original/'):
                f=trial.read_feature(trial.AUDIT,game.split('/')[1],scenario);gt=f['gt'];del f
            else:
                with np.load(ROOT/'reports/v2_search/skillcorner'/(game.split('/')[1]+'.npz')) as z:gt=z['gt']
            result={}
            for kind,model in [('full_fit_on_training_game',state),('wholefold_out',out_state)]:
                # neural() normally downsamples full-resolution features; these
                # cached arrays are already5Hz, so invoke the model directly.
                from mtah import mixed_temporal,v4
                net,mean,scale=model;pd,ph=mixed_temporal.probabilities(net,x,xs,meta['segments'],mean,scale);pd,ph=v4.neural_upsample(pd,ph,parts,len(gt));pred=boost.label(pd,ph,parts,len(gt),params);result[kind]=metric(gt,pred)
            provider='DFL' if '/idsse_' in game else 'Metrica' if '/metrica_' in game else 'SkillCorner';rows.append({'game':game,'provider':provider,'scenario':scenario,'fold':fold,'metrics':result});print(game,'diagnostic complete',flush=True)
    summary={}
    for provider in ['DFL','Metrica','SkillCorner']:
        cells=[r for r in rows if r['provider']==provider];full=float(np.mean([r['metrics']['full_fit_on_training_game']['spell_f1_mean'] for r in cells]));out=float(np.mean([r['metrics']['wholefold_out']['spell_f1_mean'] for r in cells]));summary[provider]={'games':len(cells),'full_fit_spellF1':full,'wholefold_out_spellF1':out,'optimism_gap':full-out}
    result={'candidate':selected,'bundle_sha256':sha(ROOT/'assets/pff-mixed-candidate/bundle.json'),'scope':'Post-selection descriptive29-game diagnostic of standalone neural branch, not the deployed50/50fusion; no model/threshold changes and no additional final-score selection','records':rows,'summary':summary,'limits':'Training-game scores are optimistic. Only two Metrica games; this comparison cannot establish absence of overfit. Primary complete-pipeline gates/final remain separate.'}
    dest.write_text(json.dumps(result,indent=2)+'\n');(ROOT/'docs/experiments/pff-mixed/train_cv_diagnostic.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
