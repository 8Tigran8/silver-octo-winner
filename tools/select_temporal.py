"""Freeze one neural candidate before new-source reference/model scores."""
from pathlib import Path
import json,numpy as np,statistics
from datetime import datetime,timezone
from temporal_trial import ROOT,OUT,PLAN,sha


def main():
    path=OUT/'selection.json'
    if path.exists():raise ValueError('Selection already frozen')
    records=json.loads((OUT/'cv_results.json').read_text())['records'];groups=[('original','matched_seed0'),('original','label_blind_seed1'),('original','stress_seed1'),('skillcorner','natural_detected'),('skillcorner','label_blind_seed7')];scores={}
    for candidate in ['v3','tcn','fusion50']:
        means=[]
        for family,scenario in groups:
            rows=[r for r in records if (r['candidate'],r['family'],r['scenario'])==(candidate,family,scenario)]
            if len(rows)!=(9 if family=='original' else 20):raise ValueError('Incomplete CV records')
            means.append(statistics.mean(r['spell_f1_mean'] for r in rows))
        scores[candidate]={'means':means,'weighted':.25*(means[0]+means[1]+means[3]+means[4])}
    base=scores['v3'];index={(r['candidate'],r['game'],r['scenario']):r['spell_f1_mean'] for r in records}
    for candidate in ['tcn','fusion50']:
        rng=np.random.default_rng(20261113);vectors=[];wins=0;losses=0
        for family,scenarios in [('original',['matched_seed0','label_blind_seed1']),('skillcorner',['natural_detected','label_blind_seed7'])]:
            games=sorted({r['game'] for r in records if r['family']==family})
            v=np.array([.25*sum(index[candidate,g,s]-index['v3',g,s] for s in scenarios) for g in games]);vectors.append(v);wins+=int((v>1e-10).sum());losses+=int((v < -1e-10).sum())
        draws=sum(v[rng.integers(0,len(v),(10000,len(v)))].mean(1) for v in vectors);ci=np.quantile(draws,[.025,.975]).tolist();score=scores[candidate];score.update(gain=score['weighted']-base['weighted'],descriptive_95ci=ci,game_wins=wins,game_losses=losses)
        score['pass']=score['gain']>=.01 and ci[0]>0 and all(a>=b-.01 for a,b in zip(score['means'],base['means']))
    eligible=[n for n in ['tcn','fusion50'] if scores[n]['pass']];selected=max(eligible,key=lambda n:scores[n]['weighted']) if eligible else max(['tcn','fusion50'],key=lambda n:scores[n]['weighted'])
    result={'registered_utc':datetime.now(timezone.utc).isoformat(),'candidate':selected,'development_gate_pass':scores[selected]['pass'],'scores':scores,'plan_sha256':sha(PLAN),'cv_results_sha256':sha(OUT/'cv_results.json'),'full_checkpoint_sha256':sha(OUT/'models/full/model.pt'),'baseline_checkpoint_sha256':sha(ROOT/'out/training_preflight.pkl'),'baseline_team_sha256':sha(ROOT/'reports/v4_search/models/all_nineteen.pkl'),'new_source_scores_seen':False,'release':'v3 until external confirmation and clean reproduction'}
    path.write_text(json.dumps(result,indent=2)+'\n');(ROOT/'docs/experiments/temporal/selection.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
