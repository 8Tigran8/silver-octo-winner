"""Publish aggregate PFF receipts, never raw geometry or reference labels."""
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pff_reader import sha


def main():
    out=ROOT/'docs/experiments/pff';panels=[];sections=[]
    for name,label in [('development','Development: ten new games'),('final','Final: twelve country-isolated games')]:
        source=ROOT/'reports'/('pff_'+name)/'results.json'
        if not source.exists():continue
        data=json.loads(source.read_text());decision=data['decision'];panels.append((label,decision['comparisons']['canonical/natural_detected']))
        shutil.copy2(source,out/(name+'_results.json'))
        lines=[f'## {label}',f"\n**{decision['status']}**\n",'| Scenario/reference | V4 | Candidate | Gain | Game wins |',
            '|---|---:|---:|---:|---:|']
        for key,row in decision['comparisons'].items():
            lines.append(f"| {key} | {row['baseline']:.5f} | {row['candidate']:.5f} | {row['gain']:+.5f} | {row['wins']} |")
        lines+=['','| Check | Result |','|---|---|']+[f"| {k} | {'PASS' if v else 'FAIL'} |" for k,v in decision['checks'].items()]
        lines+=['','| Game | V4 natural spellF1 | Candidate | Gain |','|---|---:|---:|---:|']
        rows={(r['game'],r['candidate']):r for r in data['records'] if r['reference']=='canonical' and r['scenario']=='natural_detected'}
        for game,gain in decision['comparisons']['canonical/natural_detected']['per_game_gain'].items():
            lines.append(f"| {game} | {rows[game,'v4']['spell_f1_mean']:.5f} | {rows[game,'source_seed20261113']['spell_f1_mean']:.5f} | {gain:+.5f} |")
        if name=='final':
            lines+=['',f"Group mean gains: {decision['group_gains']}. Descriptive paired whole-game95%interval: {decision['descriptive_paired_game_95ci']}. Shared national teams and two groups limit independent-sample interpretations."]
        sections.append('\n'.join(lines))
    if not panels:raise ValueError('No completed PFF comparison; do not fabricate a result')
    body='''# Original PFF transfer validation

One frozen provider-balanced candidate (source_seed20261113) is compared with unchanged v4. Neither model is fitted on PFF. Candidate selection predates all PFF quality scores. The original36training/10development/12final/6excluded-bridge split is unchanged. All earlier40screened games are reused research; PFF development is new-provider evidence but becomes development once examined.

## Source and annotation boundaries

Original provider access followed an explicitly authorized contact/newsletter form. The Hugging Face mirror request remains pending and no gated mirror files are used. Source: [PFF enhanced World Cup release](https://www.blog.fc.pff.com/blog/enhanced-2022-world-cup-dataset). Vendor Tracking Specification2.2 and pinned [PySport/kloppy loader](https://github.com/PySport/kloppy/blob/a9651eba7c4dbbe2d2fc6d4cffb4aea63fadac11/kloppy/infra/serializers/tracking/pff.py) define the separately reconstructed ball state/owning team.

Features use only raw VISIBLE ball/player observations; no ESTIMATED tracks, event-informed ballsSmoothed, event fields, identities, source labels or reference-conditioned corruption. Independent period numbers and frame/video timestamps define prediction spans. All supplied periods are evaluated, including missing tracking frames inside them. This is coverage of each supplied source clip, not a claim that extra time or penalties are included. Public match IDs, source team identities and raw hashes are verified.

Reference H/A/D is a provider-event-derived proxy, not corrected frame-by-frame competition truth. OUT/END marks dead, kickoff/OTB marks live, and home_ball supplies ownership; other events preserve state. Canonical references reset unknown state at each independent period and mask contradictory duplicates. A fixed last-row SDK-state shadow supplies a separate sensitivity check. Neither reference is selected by model scores. Training3816/3813 and dev3821 were source-format pilots only. The failed one-owner-per-sequence prototype remains preserved separately; numeric thresholds and candidate were not changed to fix it.

## Fixed procedure

All ten development games must pass the registered gates before final content is opened: natural mean spell gain>=.02, at least7wins, macro loss<=.01, each class loss<=.02, blind mean spell loss<=.01. Final uses evaluation-1.0: mean natural gain>=.02, at least9/12wins, positive gain in both groupsA/H, macro loss<=.01, blind mean loss<=.01. SDK sensitivity loss<=.01 is an additional conservative guard. Missing/invalid games cannot be dropped or converted into conditional H/A evaluation.

Paired probabilities exactly reproduce both frozen entrypoints on training3816 (zero maximum difference). Predictions are reference-blind; development predictions are saved before reference caches are opened. The official scorer is unchanged. Individual augmentations are not counted as independent games. Final requires a committed code/model/reader/plan freeze before any final content, and a once-only opening marker that consumes the reserve even after failure. No runner-up or post-test model choice is allowed.

No organizer-match ground truth is used for PFF scoring, fitting, calibration or candidate selection. Raw PFF data, reference arrays and frame predictions are ignored by Git and not redistributed. Publication and submission readiness remain separate from these numeric gates.

'''
    (out/'REPORT.md').write_text(body+'\n\n'.join(sections)+'\n\n![Natural-track paired gains](game_comparison.png)\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(len(panels),1,figsize=(11,3.5*len(panels)),squeeze=False)
    for ax,(label,summary) in zip(axes[:,0],panels):
        ids=list(summary['per_game_gain']);gains=list(summary['per_game_gain'].values());x=np.arange(len(ids))
        ax.bar(x,gains,color=['#19816b' if g>=0 else '#b84448' for g in gains]);ax.axhline(0,color='#333333',linewidth=.7)
        ax.axhline(.02,color='#777777',linestyle='--',linewidth=.8,label='Mean improvement gate +0.02')
        ax.set_xticks(x,ids);ax.set_ylabel('Candidate minus v4 spellF1');ax.set_title(f"{label}: mean {summary['gain']:+.4f}, {summary['wins']}/{len(ids)}wins",loc='left');ax.legend(fontsize=8)
    fig.text(.01,.01,'Provider event-derived H/A/D references; all supplied source periods; no claim of universal transfer.',fontsize=8)
    fig.tight_layout(rect=(0,.04,1,1));fig.savefig(out/'game_comparison.png',dpi=160);plt.close(fig)
    (out/'report_sources.json').write_text(json.dumps({name:sha(ROOT/'reports'/('pff_'+name)/'results.json') for name in ['development','final'] if (ROOT/'reports'/('pff_'+name)/'results.json').exists()},indent=2)+'\n')


if __name__=='__main__':main()
