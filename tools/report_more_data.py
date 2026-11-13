"""Summarize the frozen 29-game study without declaring a fresh test."""
import json
from pathlib import Path
import statistics
import numpy as np
from more_data_trial import ROOT,FOLDER,PLAN,plan,sha


def main():
    p=plan();data=json.loads((FOLDER/'development_results.json').read_text())
    if data['plan_sha256']!=sha(PLAN):raise ValueError('Changed plan')
    records=data['records'];groups=[('known','matched_seed0'),('known','label_blind_seed1'),('known','stress_seed1'),('development','natural_detected'),('development','label_blind_seed7')]
    old=json.loads((ROOT/'reports/v4_search/development_results.json').read_text())['records']
    previous={(r['game'],r['scenario']):r['spell_f1_mean'] for r in old if r['candidate']=='expanded50'}
    actual={(r['game'],r['scenario']):r['spell_f1_mean'] for r in records if r['candidate']=='v3'}
    if actual.keys()!=previous.keys() or any(abs(actual[k]-previous[k])>1e-12 for k in actual):raise ValueError('Baseline v3 was not reproduced exactly')
    scores={}
    for candidate in ['v3','data29']:
        means=[]
        for cohort,scenario in groups:
            rows=[r for r in records if (r['candidate'],r['cohort'],r['scenario'])==(candidate,cohort,scenario)]
            if len(rows)!=(9 if cohort=='known' else 10):raise ValueError('Incomplete scores')
            means.append(statistics.mean(r['spell_f1_mean'] for r in rows))
        scores[candidate]={'means':means,'weighted':.4*means[0]+.2*means[1]+.2*means[3]+.2*means[4]}
    differences=np.array(scores['data29']['means'])-np.array(scores['v3']['means'])
    gain=scores['data29']['weighted']-scores['v3']['weighted']
    passed=gain>=.005 and (differences>=-.005).all()
    index={(r['candidate'],r['game'],r['scenario']):r['spell_f1_mean'] for r in records}
    vectors=[]
    for family,scenarios in [('original',[('matched_seed0',.4),('label_blind_seed1',.2)]),('skillcorner',[('natural_detected',.2),('label_blind_seed7',.2)])]:
        games=[g for g in p['evaluation_games'] if g.startswith(family+'/')]
        vectors.append(np.array([sum(w*(index['data29',g,s]-index['v3',g,s]) for s,w in scenarios) for g in games]))
    rng=np.random.default_rng(20261113);draws=sum(v[rng.integers(0,len(v),(10000,len(v)))].mean(axis=1) for v in vectors)
    diagnostics={}
    fits=[json.loads(path.read_text()) for path in (FOLDER/'models').glob('exclude__*.json')]
    if len(fits)!=19:raise ValueError('Missing exclusions')
    for family,prefix in [('original_pool','original/'),('SkillCorner','skillcorner/'),
                          ('IDSSE_DFL','original/idsse_'),('Metrica','original/metrica_')]:
        subset=[m for m in fits if m['excluded'].startswith(prefix)]
        diagnostics[family]={}
        for key in ['log_loss','accuracy']:
            diagnostics[family][key]={
                'data29_training_mean':statistics.mean(statistics.mean(d[key] for g,d in m['training_diagnostic'].items() if g.startswith(prefix)) for m in subset),
                'data29_heldout_mean':statistics.mean(m['heldout_diagnostic'][key] for m in subset),
                'v3_heldout_mean':statistics.mean(m['v3_heldout_diagnostic'][key] for m in subset)}
    result={'baseline_reproduced':True,'scores':scores,'weighted_gain':gain,'descriptive_paired_95ci':np.quantile(draws,[.025,.975]).tolist(),'development_gate_pass':bool(passed),'new_fresh_confirmation':False,'promoted':False,'release':'v3 unchanged','diagnostics':diagnostics,'plan_sha256':sha(PLAN)}
    FOLDER.mkdir(exist_ok=True);(FOLDER/'decision.json').write_text(json.dumps(result,indent=2)+'\n')
    out=ROOT/'docs/experiments/more-data';(out/'decision.json').write_text(json.dumps(result,indent=2)+'\n');(out/'development_results.json').write_text(json.dumps(data,indent=2)+'\n')
    provenance={path.stem:json.loads(path.read_text()) for path in (FOLDER/'models').glob('*.json')}
    (out/'training_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    counts=json.loads((ROOT/'docs/experiments/expanded-team/training_data_counts.json').read_text())
    counts += [{'game':m['game'],'live_team_frames':m['reference_live_frames']} for m in
               (json.loads(path.read_text()) for path in (FOLDER/'rows').glob('*.json'))]
    if len(counts)!=29:raise ValueError('Incomplete source counts')
    (out/'training_data_counts.json').write_text(json.dumps(counts,indent=2)+'\n')
    lines=['# Fixed-capacity 19 → 29 game study','','**Verified release remains v3.** No fresh confirmation has been completed in this cycle.','',
      'The only experimental change is adding ten prior SkillCorner confirmation games to the team-head training pool. They are now training data, not a test set. The expanded head retains 108 features, original HGB capacity, augmentations, sampling and a fixed 50/50 blend with the original nine-game stacked team probabilities. The dead-probability channel and decoder stay fixed.','',
      f'The 29-game head uses {sum(r["live_team_frames"] for r in counts):,} unique live-team source frames and {provenance["all_twenty_nine"]["sampled_training_rows"]:,} sampled augmented/symmetric examples, versus 747,644 examples for the nineteen-game head. Provider frame counts are not independent statistical samples.','',
      'One full model and nineteen leave-one-whole-game-out heads are fitted. Every augmented and team-swapped sibling of a scored game is excluded from its head. Both v3 and data29 are evaluated on the same original19 excluded games; baseline scores match the prior study exactly. Historical feature/configuration choices reused these games, so this is reused-development evidence, not an unbiased final generalization estimate.','',
      '| Cohort | V3 spellF1 | Data29 spellF1 | Difference |','|---|---:|---:|---:|']
    for i,(cohort,scenario) in enumerate(groups):lines.append(f'| {cohort}/{scenario} | {scores["v3"]["means"][i]:.4f} | {scores["data29"]["means"][i]:.4f} | {differences[i]:+.4f} |')
    bounds=result['descriptive_paired_95ci']
    lines += ['',f'Weighted gain: **{gain:+.4f}**; descriptive paired 95% game-bootstrap interval [{bounds[0]:+.4f}, {bounds[1]:+.4f}]. Fixed development gate pass: **{passed}**. Gate requires gain ≥0.005 and no cohort mean loss >0.005. Original and SkillCorner games are resampled separately, preserving paired scenario differences within each game; 10,000 draws, seed 20261113. Intervals do not account for historical model selection.','',
      '## Descriptive overfit diagnostic','',
      'For each excluded fit, score at most 2,048 evenly spaced augmented/symmetric H/A rows per game. Training means weight games equally and include only the same source family as the held-out cohort; held-out rows use the same augmentation pipeline. These binary-head losses/accuracies are not the official decoded spellF1. A train/held-out gap can indicate overfit or match shift; a small gap cannot prove absence of overfit.','',
      '| Held-out family | Data29 training loss | Data29 held-out loss | V3 held-out loss | Data29 training accuracy | Data29 held-out accuracy | V3 held-out accuracy |','|---|---:|---:|---:|---:|---:|---:|']
    for family,d in diagnostics.items():
        l=d['log_loss'];a=d['accuracy'];lines.append(f'| {family} | {l["data29_training_mean"]:.4f} | {l["data29_heldout_mean"]:.4f} | {l["v3_heldout_mean"]:.4f} | {a["data29_training_mean"]:.4f} | {a["data29_heldout_mean"]:.4f} | {a["v3_heldout_mean"]:.4f} |')
    lines += ['','## Release and further data','',
      'The prior European9 have also been scored already; they cannot become fresh again. A new third-provider confirmation is required before promoting a replacement. PFF officially released 64 World Cup2022 games, but access to the OpenSportsLab mirror requires login, contact-sharing consent and owner approval. No gated file or organizer match material enters this experiment. No competition public/private GT is read. The verified v3 CSV/package remains unchanged.','',
      'Plan: [plan.json](plan.json). Full aggregate scores, game exclusions, input-row/model hashes and diagnostics are alongside this report. Raw data/weights/arrays remain local. Original source references and licences are documented in the v3 method.','',
      'Reproduce with pinned original/SkillCorner sources and prior baseline feature/probability caches:','',
      '```bash','.venv/bin/python tools/more_data_trial.py prepare --jobs 4','.venv/bin/python tools/more_data_trial.py fit --jobs 2','.venv/bin/python tools/more_data_trial.py score --jobs 4','.venv/bin/python tools/report_more_data.py','```','']
    (out/'REPORT.md').write_text('\n'.join(lines));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
