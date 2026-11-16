"""Post-experiment descriptive diagnosis; never selects or tunes a candidate."""
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
import team_gap_specialist as study
from quality_audit import official
from pff_reader import sha

def main():
    study.install_read_guard();study.check();result={}
    for scenario in study.SCENARIOS:
        summaries={c:dict(spells={b:dict(reference=0,matched=0) for b in ['<1s','1-3s','3-10s','>=10s']},missing={b:dict(live=0,wrong_team=0,false_dead=0) for b in ['<1s','1-3s','3-10s','>=10s']}) for c in ['control','uniform_temporal_focus']}
        for game in study.plan()['development_games']:
            study.guard(game,'development');ref=ROOT/f'reports/pff_access/references/kloppy_{game}.npz';receipt=json.loads((ROOT/f'reports/pff_access/kloppy_{game}_preflight.json').read_text());assert sha(ref)==receipt['reference_cache_sha256']
            with np.load(ref) as z:gt=z['gt']
            with np.load(study.OUT/'development'/f'{game}_{scenario}.npz') as z:seen=z['seen'];parts=z['segments']
            bins=np.full(len(gt),-1,np.int8)
            for a,b in parts:
                mask=~seen[a:b];edges=np.r_[0,np.flatnonzero(mask[1:]!=mask[:-1])+1,b-a]
                for x,y in zip(edges[:-1],edges[1:]):
                    if mask[x]:bins[a+x:a+y]=np.searchsorted([1,3,10],(y-x)/25,side='right')
            runs=official.valid_runs(gt);truth=official.possession_spells(gt,runs);dur=np.array([(b-a)/25 for a,b,_ in truth]);sb=np.searchsorted([1,3,10],dur,side='right')
            with np.load(ROOT/f'reports/dead_gap_focus/{game}_{scenario}_predictions.npz') as predictions:
                for candidate,s in summaries.items():
                    pred=predictions[candidate];sp=official.possession_spells(official.fill_abstentions(pred,runs),runs);matched=official.matched_gt_flags(official.spell_overlaps(truth,sp),len(truth),.5)
                    for i,name in enumerate(s['spells']):
                        s['spells'][name]['reference']+=int((sb==i).sum());s['spells'][name]['matched']+=int(matched[sb==i].sum());live=(gt<2)&(bins==i)
                        s['missing'][name]['live']+=int(live.sum());s['missing'][name]['wrong_team']+=int((live&(pred<2)&(pred!=gt)).sum());s['missing'][name]['false_dead']+=int((live&(pred==2)).sum())
        result[scenario]=summaries
    path=ROOT/'docs/experiments/dead-gap-focus/failure_profile.json';assert not path.exists();path.write_text(json.dumps(dict(scope='Original ten development games; canonical event proxy; exploratory diagnosis after selection rejection, not new acceptance criteria',best_rejected='uniform_temporal_focus',results=result),indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
