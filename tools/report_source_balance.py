"""Record provider-balanced development and frozen reused-source outcome."""
from pathlib import Path
import json,statistics,shutil
ROOT=Path(__file__).resolve().parents[1];IN=ROOT/'reports/source_balance';OUT=ROOT/'docs/experiments/source-balance'


def main():
    if not (IN/'pipeline_complete.json').exists():raise ValueError('Provider-balanced study is incomplete')
    selection=json.loads((IN/'selection.json').read_text());summary=json.loads((IN/'summary.json').read_text());guards=json.loads((IN/'provider_guards.json').read_text());groups=[]
    if selection.get('candidate'):
        ext=json.loads((IN/'reused_external.json').read_text())
        for cohort in ['wcsf','metrica3','european']:
            for scenario in sorted({r['scenario'] for r in ext['records'] if r['cohort']==cohort}):
                rows=[r for r in ext['records'] if (r['cohort'],r['scenario'])==(cohort,scenario)]
                groups.append({'cohort':cohort,'scenario':scenario,'games':len(rows),'v4':statistics.mean(r['v4_primary'] for r in rows),'candidate':statistics.mean(r['candidate_primary'] for r in rows),'gain':statistics.mean(r['gain'] for r in rows),'wins':sum(r['gain']>0 for r in rows)})
        shutil.copy2(IN/'reused_external.json',OUT/'reused_external.json')
    transfer_pass=bool(groups) and all(r['gain']>=-.01-1e-12 for r in groups)
    for name in ['selection.json','summary.json','results.json','provider_guards.json']:shutil.copy2(IN/name,OUT/name)
    proof={p.parent.name:json.loads(p.read_text()) for p in (IN/'models').glob('*/meta.json')};(OUT/'training_provenance.json').write_text(json.dumps(proof,indent=2)+'\n')
    decision={'candidate':selection.get('candidate'),'development':selection.get('development'),'provider_guard_gains':guards,'reused_sources':groups,'reused_source_guard_pass':transfer_pass,'status':'RESEARCH_CANDIDATE: numeric/provider/reused-source guards pass; independent confirmation and strict inherited-fold audit pending' if transfer_pass else 'NOT_ADOPTED: fixed development/provider or reused-source guard failed','fresh_independent_games':0,'lineage':'New neural fits exclude whole folds; inherited legacy components retain scored-game exclusions, not every other member of that fold.','release':'verified v4 unchanged','organizer_GT_used':False,'organizer_data_used_for_training_or_quality':False,'user_hold':'No submission/messages/push/publication'}
    (OUT/'decision.json').write_text(json.dumps(decision,indent=2)+'\n')
    lines=['# Equal-provider temporal training','','**'+decision['status']+'**','',
      'This study was registered after longer training and OOD gates failed transfer checks on previously used Metrica3. It tests a change to training, not another threshold on that match. All40previously screened games are research now; no independent final test was opened.','',
      'The previous sampler gave original games1/2 and SkillCorner1/2. Since the original family has sevenDFL and twoMetrica games, Metrica received1/9of training draws. This experiment instead samples DFL/Metrica/SkillCorner each1/3, uniformly over training games/scenarios within each provider. Provider identity affects the sampler only, never inference features.','',
      'All registered fits keep the 48-channel, 25-second TCN and use 4500 fixed AdamW updates, the same loss, optimizer, normalization and augmentations as the longer baseline. Train three fixed seeds on five complete neural folds. This isolates provider sampling from the earlier longer-training family; seed trajectories change with the new sampler.','',
      'Candidate policies are three individual uniform50 fusions with v3 and their equal-probability ensemble. Select the best fixed-development candidate passing evaluation-1.0 plus the separately preregistered provider guard: primary mean spellF1 must not fall by>.01on DFL7, Metrica2 or SkillCorner20 separately. Freeze one candidate before full fitting and reused-source checks; no runner-up is chosen afterward.','',
      '| Candidate | Composite spellF1 | Gain vs v4 | Wins /29 | Numeric status |','|---|---:|---:|---:|---|']
    for name,value in sorted(summary.items(),key=lambda item:item[1]['gain'],reverse=True):lines.append(f"| {name} | {value['candidate_composite']:.5f} | {value['gain']:+.5f} | {value['game_wins']} | {value['status']} |")
    lines+=['','## Provider guards','','| Candidate | DFL gain | Metrica gain | SkillCorner gain |','|---|---:|---:|---:|']
    for name,value in guards.items():lines.append(f"| {name} | {value['DFL']:+.5f} | {value['Metrica']:+.5f} | {value['SkillCorner']:+.5f} |")
    if groups:
        lines+=['','## Frozen reused-source checks','','| Source | Scenario | V4 | Candidate | Gain |','|---|---|---:|---:|---:|']
        for row in groups:lines.append(f"| {row['cohort']} ({row['games']}) | {row['scenario']} | {row['v4']:.4f} | {row['candidate']:.4f} | {row['gain']:+.4f} |")
    lines+=['','The primary composite and complete67scenario grid are unchanged. Per-game wins average two primary scenarios; augmentations are not independent games. Intervals in JSON files are descriptive, without correction for repeated selection or team/source dependence.','',
      'All labels and geometry used for fitting/scoring are public external data, never competition GT. WCSF ambiguity/anonymous identity limits, event-derived Metrica references and conditional-only European ownership remain explicit. New neural stages exclude complete folds. The unchanged legacy v3 paths exclude scored games but not every other member of that neural fold, so strict all-stage full-fold compliance and independent confirmation remain pending.','',
      'Keep verified v4 and its local submission package. The user hold bars form submission, outbound messages, push and publication. Default run.py v4 is unchanged. All failures and three seeds remain visible.','']
    (OUT/'REPORT.md').write_text('\n'.join(lines));print(json.dumps(decision,indent=2))


if __name__=='__main__':main()
