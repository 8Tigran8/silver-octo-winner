"""Post-hoc descriptive audit, never a new selection rule or decoder policy."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
import numpy as np
from sklearn.metrics import average_precision_score
from boundary_event_study import OUT,plan,sha,write
from quality_audit import official
from mtah.boundary_events import targets


def main():
    result_file=OUT/'pff_dev_results.json'
    results=json.loads(result_file.read_text())
    cfg=plan();records=[];head=[];source={}
    for game in cfg['pff_dev_ids']:
        reference=ROOT/'reports/pff_access/references'/f'kloppy_{game}.npz'
        predictions=OUT/f'{game}_predictions.npz';source[str(reference.relative_to(ROOT))]=sha(reference);source[str(predictions.relative_to(ROOT))]=sha(predictions)
        with np.load(reference,allow_pickle=False) as z:gt=z['gt']
        runs=official.valid_runs(gt);spells=official.possession_spells(gt,runs)
        durations=np.array([(b-a)/25 for a,b,k in spells]);bins={'under1':durations<1,'1to3':(durations>=1)&(durations<3),'3to10':(durations>=3)&(durations<10),'10plus':durations>=10}
        with np.load(predictions,allow_pickle=False) as z:
            for scenario in ['natural_detected','label_blind_seed7']:
                for name in [p['name'] for p in cfg['policies']]:
                    pred=official.fill_abstentions(z[scenario+'__'+name],runs)
                    predicted=official.possession_spells(pred,runs);overlaps=official.spell_overlaps(spells,predicted)
                    flags=np.stack([official.matched_gt_flags(overlaps,len(spells),t) for t in official.SPELL_IOUS])
                    per_bin={k:{'n_gt':int(mask.sum()),'recall_mean_iou':float(flags[:,mask].mean()) if mask.any() else None} for k,mask in bins.items()}
                    records.append({'game':game,'scenario':scenario,'candidate':name,'n_gt':len(spells),'n_pred':len(predicted),'mean_pred_duration_s':float(np.mean([(b-a)/25 for a,b,k in predicted])),'spells':per_bin})
                cache=OUT/'probabilities'/f'{game}__{scenario}.npz';source[str(cache.relative_to(ROOT))]=sha(cache)
                with np.load(cache,allow_pickle=False) as probabilities:q=probabilities['q'][::5]
                parts=[((a+4)//5,(b+4)//5) for a,b in runs];target=targets(gt[::5],parts);valid=target>=0
                for k,label in [(1,'live_handover'),(2,'live_dead')]:
                    y=target[valid]==k;score=q[valid,k]
                    head.append({'game':game,'scenario':scenario,'boundary_type':label,'positive_fraction':float(y.mean()),'average_precision':float(average_precision_score(y,score)),'n_positive':int(y.sum())})
    summary={}
    for scenario in ['natural_detected','label_blind_seed7']:
        summary[scenario]={}
        for name in [p['name'] for p in cfg['policies']]:
            cells=[r for r in records if r['candidate']==name and r['scenario']==scenario]
            pooled={}
            for label in ['under1','1to3','3to10','10plus']:
                count=sum(r['spells'][label]['n_gt'] for r in cells)
                pooled[label]={'n_gt':count,'recall_mean_iou':sum(r['spells'][label]['n_gt']*(r['spells'][label]['recall_mean_iou'] or 0) for r in cells)/count if count else None}
            summary[scenario][name]={'n_gt':sum(r['n_gt'] for r in cells),'n_pred':sum(r['n_pred'] for r in cells),'per_duration':pooled,'mean_game_pred_duration_s':float(np.mean([r['mean_pred_duration_s'] for r in cells]))}
    seeds={}
    for seed in cfg['seeds']:
        name=f'event1_seed{seed}';seeds[str(seed)]={}
        for scenario in ['natural_detected','label_blind_seed7']:
            cells=[r for r in results['records'] if r['candidate']==name and r['scenario']==scenario and r['reference']=='canonical']
            seeds[str(seed)][scenario]=float(np.mean([r['spell_f1_mean'] for r in cells]))
    write(OUT/'diagnostics.json',{'scope':'Post-hoc descriptive development diagnostics; no policy, fit, threshold or release change. Pooled durations are descriptive; matches are not independent national-team groups. Boundary AP is exact 5 Hz timing, not event-level precision.','results_sha256':sha(result_file),'input_sha256':source,'summary':summary,'seed_scores':seeds,'boundary_ap':head,'records':records})
    print(json.dumps({'summary':summary,'seed_scores':seeds},indent=2))


if __name__=='__main__':main()
