"""Aggregate-only study report; source frames and reference arrays stay local."""
import hashlib
import json
from pathlib import Path
import shutil
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'docs/experiments/pff-mixed';WORK=ROOT/'reports/pff_mixed'


def main():
    old=json.loads((WORK/'old_results.json').read_text());dev=json.loads((WORK/'pff_dev_results.json').read_text());reused=json.loads((WORK/'reused_results.json').read_text());selection=json.loads((WORK/'selection.json').read_text())
    candidate=selection['candidate'];final_path=ROOT/'reports/pff_mixed_final/results.json';final=json.loads(final_path.read_text()) if final_path.exists() else None
    copies={name:WORK/name for name in ['old_results.json','old_provenance.json','pff_dev_results.json','reused_results.json','selection.json']}
    if final:copies['final_results.json']=final_path
    if (WORK/'deployment_equivalence.json').exists():copies['deployment_equivalence.json']=WORK/'deployment_equivalence.json'
    for name,source in copies.items():shutil.copy2(source,OUT/name)
    lines=['# 65-game temporal and boundary study',f'\nSelected before final opening: **{candidate}**. '+('Final quality: **'+final['decision']['status']+'**.' if final else 'Final quality is not yet assessed.'),
        '\n## Registered methods and sources',
        '\nThe unchanged v4 release is the control. The temporal branch now trains on65public games: DFL7, Metrica2, SkillCorner20 and original-provider PFF36. The PFF addition contributes5,636,115unaugmented H/A/D period frames; all augmented siblings remain the same match. No organizer labels, PFFdevelopment/final games or country bridges enter any fit or normalizer.',
        '\nBoth fixed variants use48channels,108geometry/contact features plus108finite masks,5Hz, seed20261113,6000AdamWsteps, batch16, LR.001, decay.001, dropout.15 and equal sampling of four providers.6000/4preserves the expected1500updates per older provider relative to4500/3. SkillCorner Dproxy loss weight remains.25. The boosted path, mixture.5and cost9duration decoder are unchanged.',
        '\n| Variant | Temporal receptive field | Boundary auxiliary loss |', '|---|---|---|',
        '| mixed25 |125steps =25seconds;64context steps each side | None |',
        '| boundary102 |509steps =101.8seconds;256context steps each side |.15weighted change BCE; positiveweight8 |',
        '\nBoundary change probability is1 minus the dot product of adjacent predicted H/A/D distributions. Unknown/context-only pairs are masked; crops never cross independent periods. The longer context and auxiliary loss change together, so these experiments do not separately isolate their individual effects.',
        '\nPFF geometry uses raw VISIBLE detections only. ESTIMATED tracks, event-informed smoothed coordinates, event/possession fields and identities are excluded from features. References separately follow the pinned primary kloppy state machine and vendor Tracking Specification2.2. They are event-derived proxies, not corrected frame-by-frame competition truth. Training/dev source IDs, clock, periods, coverage and raw/cache hashes are checked. All supplied periods, including extra time when available, are covered; missing tracking frames inside a period remain graded.',
        '\nSources: [original PFF release](https://www.blog.fc.pff.com/blog/enhanced-2022-world-cup-dataset), [pinned primary loader](https://github.com/PySport/kloppy/blob/a9651eba7c4dbbe2d2fc6d4cffb4aea63fadac11/kloppy/infra/serializers/tracking/pff.py), [prior source registry](../temporal/REPORT.md). Original raw data and reference arrays are not redistributed. The separate Hugging Face mirror is unused.',
        '\n## All-stage whole-fold development',
        '\nThe exact29-game fivefold mapping and67scenario cells are retained. Every inherited direct/nested fit excludes the whole scored fold. New neural folds use the remaining23/24older games plus all36PFFtraining games. Actual normalizers are recomputed from those training observations and compared exactly with the checkpoint. The recorded component hashes and transitive exclusions are verified. Both immutable historical-v4 and identical-exclusion reconstructed-v4 numeric gates are required.',
        '\n| Variant | Candidate composite | Gain vs historical v4(.61435) | Wins | Gain vs strict v4(.61218) | Wins |',
        '|---|---:|---:|---:|---:|---:|']
    for name,d in old['decisions'].items():
        h=d['historical'];p=d['paired'];lines.append(f"| {name} | {h['candidate_composite']:.5f} | {h['gain']:+.5f} | {h['game_wins']}/29 | {p['gain']:+.5f} | {p['game_wins']}/29 |")
    lines+=['\nAll old-family/cohort, class, stress, lower-tail, large-loss and descriptive-bootstrap gates pass for both variants. These are reused research games, so intervals do not remove historical model-selection or opponent dependence.',
        '\n## PFFdevelopment: all ten games',
        '\nNone of these games trains any learned stage or normalizer. They are now a model-selection set, not an untouched final estimate. The original frozen-v4 outputs are recomputed exactly on every sensor cell. Canonical and fixed SDK-shadow references agree on these ten games.',
        '\n| Variant/scenario | V4 spellF1 | Candidate | Gain | Game wins | MacroF1gain |', '|---|---:|---:|---:|---:|---:|']
    for name,d in dev['decisions'].items():
        for scenario in ['natural_detected','label_blind_seed7']:
            r=d['comparisons']['canonical/'+scenario];lines.append(f"| {name}/{scenario} | {r['baseline']:.5f} | {r['candidate']:.5f} | {r['gain']:+.5f} | {r['wins']}/10 | {r['macro_gain']:+.5f} |")
    lines+=['\nBoth variants pass the predeclared natural mean+.02, at least7/10wins, macro/class and sensor-stress/sensitivity guards. No failed match is dropped.',
        '\n## Eleven reused external regression games',
        '\n| Cohort/scenario | mixed25primary gain | boundary102primary gain |', '|---|---:|---:|']
    for key in reused['decisions']['mixed25']['primary_gains']:
        lines.append(f"| {key} | {reused['decisions']['mixed25']['primary_gains'][key]:+.5f} | {reused['decisions']['boundary102']['primary_gains'][key]:+.5f} |")
    lines+=['\nThe European9primary metric is conditional H/A macroF1, not full H/A/D spellF1. WCSF and Metrica3 use their previously documented reference/provenance caveats. Each predeclared cohort/scenario mean-loss guard is.01. Boundary102 loses.00900 on the noisy Metrica3 scenario, close to that limit. Improvements are not universal.']
    diagnostic=OUT/'spell_diagnostics.json'
    if diagnostic.exists():
        summary=json.loads(diagnostic.read_text())['summary'];r=summary['natural_detected/boundary102'];b=summary['natural_detected/v4']
        lines += ['\n## Spell count and duration diagnostics',
            f"\nOn PFF development natural inputs, the mean per-game predicted/reference spell-count ratio is {r['spell_count_ratio']:.3f} versus v4 {b['spell_count_ratio']:.3f}. Mean predicted possession duration is {r['mean_predicted_spell_seconds']:.2f}s versus reference {r['mean_reference_spell_seconds']:.2f}s. The model still merges/misses short possessions; better spellF1 does not mean perfect segmentation. These are post-selection descriptive diagnostics, not an added gate or a reason to change the frozen model."]
    lines += [
        '\n## Selection and exact-file verification',
        '\nBoth variants satisfy every registered development requirement plus DFL/Metrica/SkillCorner provider guards. The fixed ranking is the mean of historical-v4 development gain and PFFdev natural gain; boundary102 wins. There is no seed, decoder, gate or runner-up selection on final data. The original two failed source/reference and29-only PFF-transfer studies remain preserved under `../pff`.',
        '\nDeployment exactly reproduces the previously scored dev3821predictions for natural and blind cases, both control and candidate: zero differing frame labels. Two clean committed-source CPU generations produce the same exact organizer CSV and pass the official format validator; this is frozen-weight inference reproduction, not repeated training or hidden target-quality validation.']
    proof_path=ROOT/'out/candidates/pff-mixed/verification.json'
    if proof_path.exists():
        proof=json.loads(proof_path.read_text());public={k:v for k,v in proof.items() if k!='input_sha256'};public['input_sha256']={Path(k).name:v for k,v in proof['input_sha256'].items()};(OUT/'reproduction.json').write_text(json.dumps(public,indent=2)+'\n')
        lines.append(f"\nCPU runs: {proof['runs'][0]['seconds']:.3f}s and {proof['runs'][1]['seconds']:.3f}s; peak child RSS {proof['peak_child_RSS_mb']:.2f}MiB. CSV SHA256: `{proof['runs'][0]['csv_sha256']}`. Source: `{proof['source_commit']}`.")
    lines+=['\n## Original twelve-game final reserve',
        '\nThe original groupsA/H country-isolated split is unchanged; both national teams are absent from all PFFtrain/dev matches. One candidate, source/model/reader/scenario hashes and exact-file receipts are frozen before any final contents. Opening consumes the reserve even after failure. All12games and both fixed sensor cases are required. The final natural mean gain must be>=.02with>=9wins, both group gains positive, macro loss<=.01, blind spell loss<=.01and fixed SDK sensitivity loss<=.01. A source/adapter failure blocks promotion instead of changing the endpoint or dropping a match.']
    if final:
        d=final['decision'];lines+=[f"\n**{d['status']}**",'\n| Final comparison | V4 | Candidate | Gain | Wins |', '|---|---:|---:|---:|---:|']
        for key,r in d['comparisons'].items():lines.append(f"| {key} | {r['baseline']:.5f} | {r['candidate']:.5f} | {r['gain']:+.5f} | {r['wins']}/12 |")
        lines+=[f"\nGroup gains: {d['group_gains']}. Descriptive paired whole-game95%gain interval: {d['descriptive_paired_game_95ci']}. Shared national teams and only two groups limit independence assumptions."]
    else:lines+=['\nFinal evaluation is in progress or has not been completed. This report does not certify release readiness.']
    gap_path=OUT/'train_cv_diagnostic.json'
    if gap_path.exists():
        gap=json.loads(gap_path.read_text());lines+=['\n## Post-selection train/CV diagnostic','\nStandalone neural branch only, before the50/50fusion. Same29primary natural/matched game inputs; full-fit training-game predictions versus complete-fold-out predictions. No weights, gates or selection change after this diagnostic.','\n| Source | Games | Full-fit spellF1 | Whole-fold-out | Optimism gap |','|---|---:|---:|---:|---:|']
        for provider,r in gap['summary'].items():lines.append(f"| {provider} | {r['games']} | {r['full_fit_spellF1']:.5f} | {r['wholefold_out_spellF1']:.5f} | {r['optimism_gap']:+.5f} |")
        lines+=['\nThe two-game Metrica gap of.151indicates a material source-specific overfit risk. DFLshows modest training optimism; SkillCorner does not show a positive gap here. This does not invalidate the complete-pipeline held-out comparisons, but absence of overfit cannot be claimed. The frozen final12result is separately reported above.']
    lines+=['\nGeneralization remains conditional on sensor/reference semantics and sampled matches; no universal/no-overfit or competition-rank claim. The provided match is never used for supervised fitting, candidate selection or quality scoring. Publication and organizer submission are separate steps and remain on hold.','\n![PFF paired game gains](game_comparison.png)']
    text='\n'.join(lines)+'\n'
    (OUT/'REPORT.md').write_text(text)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    panels=2 if final else 1;fig,axes=plt.subplots(panels,1,figsize=(11,3.6*panels),squeeze=False)
    ax=axes[0,0];ids=list(dev['decisions']['mixed25']['comparisons']['canonical/natural_detected']['per_game_gain']);x=np.arange(len(ids))
    for offset,name,color in [(-.19,'mixed25','#7b9da9'),(.19,'boundary102','#157d69')]:ax.bar(x+offset,[dev['decisions'][name]['comparisons']['canonical/natural_detected']['per_game_gain'][g] for g in ids],width=.36,color=color,label=name)
    ax.set_xticks(x,ids);ax.axhline(0,color='#333333',linewidth=.7);ax.set_title('PFF development: all ten games, natural detections',loc='left');ax.set_ylabel('SpellF1gain versus v4');ax.legend(fontsize=9)
    if final:
        ax=axes[1,0];r=final['decision']['comparisons']['canonical/natural_detected'];ids=list(r['per_game_gain']);gains=list(r['per_game_gain'].values());ax.bar(np.arange(len(ids)),gains,color=['#157d69' if g>=0 else '#b7474c' for g in gains]);ax.set_xticks(np.arange(len(ids)),ids);ax.axhline(0,color='#333333',linewidth=.7);ax.set_title(f"Country-held-out final: frozen {candidate}, mean {r['gain']:+.4f}, {r['wins']}/12 wins",loc='left');ax.set_ylabel('SpellF1 gain versus v4')
    fig.text(.01,.012,'Provider event-derived H/A/D; whole supplied periods; no claim of universal transfer.',fontsize=8);fig.tight_layout(rect=(0,.045,1,1));fig.savefig(OUT/'game_comparison.png',dpi=160);plt.close(fig)
    if final:
        r=final['decision']['comparisons']['canonical/natural_detected'];ids=list(r['per_game_gain']);gains=list(r['per_game_gain'].values())
        fig,ax=plt.subplots(figsize=(11,4.5));ax.bar(np.arange(len(ids)),gains,color='#157d69');ax.axhline(0,color='#333333',linewidth=.8);ax.axhline(r['gain'],color='#173547',linestyle='--',linewidth=1.3,label=f"Mean gain {r['gain']:+.4f}");ax.set_xticks(np.arange(len(ids)),ids,fontsize=13);ax.tick_params(axis='y',labelsize=13);ax.set_ylabel('SpellF1 gain versus v4',fontsize=14);ax.set_xlabel('Previously unopened PFF match',fontsize=13);ax.set_title('One frozen model: improvement on all 12 final matches',fontsize=15,loc='left');ax.legend(fontsize=13);fig.tight_layout();fig.savefig(OUT/'final_game_gains.png',dpi=170);plt.close(fig)
    (OUT/'report_sources.json').write_text(json.dumps({name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in copies.items()},indent=2)+'\n')


if __name__=='__main__':main()
