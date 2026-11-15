"""Summarize fixed source-held-out neural diagnostics, retaining regressions."""
from pathlib import Path
import json
import statistics
ROOT=Path(__file__).resolve().parents[1]
DOC=ROOT/'docs/experiments/strict-fullfold'


def main():
    dest=DOC/'provider_results.json'
    if dest.exists():raise ValueError('Do not overwrite provider diagnostic report')
    payload=json.loads((ROOT/'reports/strict_fullfold/provider/results.json').read_text())
    rows=payload['records'];index={(r['kind'],r['game'],r['scenario']):r for r in rows}
    summary=[]
    for provider in ['DFL','Metrica','SkillCorner']:
        scenarios=['natural_detected','label_blind_seed7'] if provider=='SkillCorner' else ['matched_seed0','label_blind_seed1','stress_seed1']
        for scenario in scenarios:
            chosen=[r for r in rows if r['omitted_provider']==provider and r['scenario']==scenario and r['kind']=='candidate']
            control=statistics.mean(index['control',r['game'],scenario]['spell_f1_mean'] for r in chosen)
            candidate=statistics.mean(r['spell_f1_mean'] for r in chosen)
            summary.append({'heldout_provider':provider,'scenario':scenario,'games':len(chosen),'control':control,'candidate':candidate,'gain':candidate-control,
                            'wins':sum(r['spell_f1_mean']>index['control',r['game'],scenario]['spell_f1_mean']+1e-10 for r in chosen)})
    gap=[]
    for provider in ['DFL','Metrica','SkillCorner']:
        values={scope:statistics.mean(r['spell_f1_mean'] for r in payload['train_vs_cv'] if r['provider']==provider and r['scope']==scope) for scope in ['full_fit_training','wholefold_out']}
        gap.append({'provider':provider,**values,'train_minus_cv':values['full_fit_training']-values['wholefold_out']})
    transfer_gap=[]
    for provider in ['DFL','Metrica','SkillCorner']:
        held=next(r for r in summary if r['heldout_provider']==provider and r['scenario'] in ['matched_seed0','natural_detected'])
        cv=next(r for r in gap if r['provider']==provider)
        transfer_gap.append({'provider':provider,'wholefold_out':cv['wholefold_out'],'wholeprovider_out':held['candidate'],
                             'provider_out_minus_game_out':held['candidate']-cv['wholefold_out']})
    result={'scope':'Neural branch only; no inherited components enter these source-held-out fits. Retrospective diagnostics, never independent confirmation. Fixed seed20261113; no selection.',
            'source_heldout':summary,'train_vs_cv':gap,'provider_shift':transfer_gap,'raw':payload}
    dest.write_text(json.dumps(result,indent=2)+'\n')
    text=['# Source-transfer and overfit diagnostics','',
          'The frozen candidate is unchanged. These are six additional neural fits: each of three providers is completely excluded, once with v4 neural training and once with the source-balanced longer training. Both models in a pair use the identical remaining training games and train-only normalization. No inherited v3 probability is mixed in, because that would reintroduce the held-out provider. These results describe the neural branch, not the full deployment pipeline.','',
          '| Held-out source | Scenario | Games | Control spellF1 | Candidate spellF1 | Gain | Wins |',
          '|---|---|---:|---:|---:|---:|---:|']
    text += [f"| {r['heldout_provider']} | {r['scenario']} | {r['games']} | {r['control']:.4f} | {r['candidate']:.4f} | {r['gain']:+.4f} | {r['wins']} |" for r in summary]
    text += ['', '## Training versus complete-game-out','',
             'Standalone source-balanced neural branch, same primary matched/natural scenarios. Full-fit training scores are in-sample and cannot be used as validation. The CV score uses complete excluded neural folds. The larger full training pool is an additional difference, so the gap is descriptive.','',
             '| Provider | Full-fit train spellF1 | Wholefold-out spellF1 | Train minus CV |','|---|---:|---:|---:|']
    text += [f"| {r['provider']} | {r['full_fit_training']:.4f} | {r['wholefold_out']:.4f} | {r['train_minus_cv']:+.4f} |" for r in gap]
    text += ['', '## Remaining evidence of source sensitivity','',
             '| Provider | Whole-game-fold-out | Whole-provider-out | Difference |','|---|---:|---:|---:|']
    text += [f"| {r['provider']} | {r['wholefold_out']:.4f} | {r['wholeprovider_out']:.4f} | {r['provider_out_minus_game_out']:+.4f} |" for r in transfer_gap]
    text += ['', 'Metrica contains only two games and its training/CV gap is materially larger than the other providers. This is a warning about possible fitting to those games, not proof that its unseen-game performance will be poor. DFL and SkillCorner accuracy also drops when their whole provider is omitted, despite improving against the control under that same omission. Generalization to a new tracking/label source remains unresolved.','',
             'No new source was opened. References include provider-specific event/phase proxies; source semantics and optical coverage differ. Small training/CV gaps do not establish no overfit, and performance under provider holdout cannot establish hidden organizer-match quality. Every failure stays in this report; no candidate is replaced using these scores. The independent PFF confirmation remains pending.','']
    (DOC/'PROVIDER_REPORT.md').write_text('\n'.join(text))
    print(json.dumps({k:result[k] for k in ['source_heldout','train_vs_cv']},indent=2))


if __name__=='__main__':main()
