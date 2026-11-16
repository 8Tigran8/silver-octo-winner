"""Summarize the prespecified unseen-match audit; no fitting or selection."""
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from pff_unseen_audit import DOC,OUT,IDS,check_freeze
from pff_reader import load_geometry,tracking_reference,sha
from quality_audit import official
from mtah import degrade,expanded_team


def components(games):
    remaining=set(games);clusters=[]
    while remaining:
        group={remaining.pop()};teams={games[next(iter(group))][s] for s in ['home','away']}
        changed=True
        while changed:
            changed=False
            for game in sorted(remaining):
                sides={games[game][s] for s in ['home','away']}
                if sides&teams:group.add(game);teams|=sides;remaining.remove(game);changed=True
        clusters.append(sorted(group))
    return sorted(clusters)


def paired_summary(rows,games,clusters):
    result={};ix={(r['game'],r['scenario'],r['candidate'],r['reference']):r for r in rows}
    rng=np.random.default_rng(20261116);draw=rng.integers(0,len(games),(10000,len(games)))
    cluster_draw=rng.integers(0,len(clusters),(10000,len(clusters)))
    for ref in ['canonical','sdk_shadow']:
        for scenario in ['natural_detected','label_blind_seed7']:
            old=[ix[g,scenario,'v4',ref] for g in games];new=[ix[g,scenario,'boundary102',ref] for g in games]
            delta=np.array([b['spell_f1_mean']-a['spell_f1_mean'] for a,b in zip(old,new)])
            pergame={g:float(d) for g,d in zip(games,delta)}
            sums=np.array([sum(pergame[g] for g in c) for c in clusters]);sizes=np.array([len(c) for c in clusters])
            cluster_boot=sums[cluster_draw].sum(axis=1)/sizes[cluster_draw].sum(axis=1)
            result[ref+'/'+scenario]=dict(v4_spell=float(np.mean([r['spell_f1_mean'] for r in old])),
                current_spell=float(np.mean([r['spell_f1_mean'] for r in new])),spell_gain=float(delta.mean()),
                wins=int((delta>0).sum()),ties=int((delta==0).sum()),losses=int((delta<0).sum()),worst_gain=float(delta.min()),
                v4_macro=float(np.mean([r['macro_f1'] for r in old])),current_macro=float(np.mean([r['macro_f1'] for r in new])),
                per_class_current={k:float(np.mean([r['per_class'][k]['f1'] for r in new])) for k in ['H','A','D']},
                descriptive_match_bootstrap_gain_95=np.quantile(delta[draw].mean(axis=1),[.025,.975]).tolist(),
                descriptive_shared_team_cluster_bootstrap_gain_95=np.quantile(cluster_boot,[.025,.975]).tolist(),
                per_game_gain=pergame)
    return result


def diagnostics(source):
    game=source['game'];destination=OUT/(game+'_diagnostics.json')
    if destination.exists():return json.loads(destination.read_text())
    for name,digest in source['source_files'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Changed observed source')
    base=ROOT/'external/pff'
    data,info=load_geometry(base/'Tracking Data'/(game+'.jsonl.bz2'),base/'Metadata'/(game+'.json'),base/'Rosters'/(game+'.json'))
    gt,_,_=tracking_reference(base/'Tracking Data'/(game+'.jsonl.bz2'),data['time'],data['period'],info['fps'])
    dummy=np.where(data['period']>0,0,3).astype(np.int8)
    blind=degrade.degrade(dict(data,gt=dummy),seed=7,params=expanded_team.BLIND);blind.pop('gt')
    runs=official.valid_runs(gt);spells=official.possession_spells(gt,runs)
    duration=np.array([(b-a)/25 for a,b,_ in spells]);rows=[]
    predfile=OUT/(game+'_predictions.npz')
    if sha(predfile)!=source['predictions_sha256']:raise ValueError('Changed predictions')
    with np.load(predfile) as predictions:
        for scenario,d in [('natural_detected',data),('label_blind_seed7',blind)]:
            seen=np.isfinite(d['ball'][:,:2]).all(axis=1)
            for name in ['v4','boundary102']:
                pred=predictions[scenario+'__'+name]
                ps=official.possession_spells(official.fill_abstentions(pred,runs),runs)
                matched=official.matched_gt_flags(official.spell_overlaps(spells,ps),len(spells),.5)
                bins={}
                for key,lo,hi in [('under_1s',0,1),('1_to_3s',1,3),('3_to_10s',3,10),('10s_plus',10,np.inf)]:
                    mask=(duration>=lo)&(duration<hi);bins[key]=dict(reference_spells=int(mask.sum()),matched_spells=int(matched[mask].sum()))
                obs={}
                for key,mask in [('ball_seen',seen),('ball_missing',~seen)]:
                    valid=mask&(gt<3);live=mask&(gt<2)
                    obs[key]=dict(graded_frames=int(valid.sum()),correct_frames=int(((gt==pred)&valid).sum()),
                                  reference_live_frames=int(live.sum()),false_dead_frames=int(((pred==2)&live).sum()))
                rows.append(dict(game=game,scenario=scenario,candidate=name,short_spell_bins=bins,observation=obs))
    destination.write_text(json.dumps(rows,indent=2)+'\n');print('Diagnostics',game,'complete',flush=True);return rows


def main():
    check_freeze();result=json.loads((OUT/'results.json').read_text());freeze=json.loads((DOC/'freeze.json').read_text())
    games={g['game_id']:g for g in freeze['games']};clusters=components(games)
    summary=paired_summary(result['records'],list(IDS),clusters)
    rows=[]
    for source in result['source_receipts']:rows.extend(diagnostics(source))
    report=dict(scope='New match holdout with partial country overlap; no training/selection; same provider proxy labels',
                source_result_sha256=sha(OUT/'results.json'),freeze_sha256=sha(DOC/'freeze.json'),
                shared_team_components=clusters,summary=summary,diagnostics=rows,
                uncertainty_warning='Both intervals are descriptive; six matches/four shared-team components, common provider and prior selection preclude a calibrated universal no-overfitting conclusion.')
    (DOC/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    # Publish aggregate scores/source hashes, never per-frame arrays or raw records.
    (DOC/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
