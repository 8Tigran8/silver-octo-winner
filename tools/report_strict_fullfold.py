"""Publish the strict audit, including failures, without changing release assets."""
from pathlib import Path
import json
import sys
import statistics
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from check_evaluation import evaluate,evaluate_paired_audit,sha
from strict_fullfold import assert_component_isolation,fold_games,original_names
from mtah import expanded_team as expanded

OUT=ROOT/'reports/strict_fullfold';DOC=ROOT/'docs/experiments/strict-fullfold'


def write(path,obj):
    path.write_text(json.dumps(obj,indent=2)+'\n')


def main():
    if (DOC/'decision.json').exists():raise ValueError('Do not overwrite completed strict-audit decision')
    payload=json.loads((OUT/'results.json').read_text());proof=json.loads((OUT/'provenance.json').read_text())
    for fold in range(5):
        meta=proof[f'fold{fold}'];train,test=fold_games(fold);assert_component_isolation(meta,test)
        expected_original={g for g in train if g.startswith('original/')}
        expected_team=set(expanded.games())-set(test)
        if set(meta['stage1_train'])!=expected_original or set(meta['stack_train'])!=expected_original or set(meta['team_train'])!=expected_team:
            raise ValueError('Inherited fitting pool differs from frozen specification')
        if {p['inner_test'] for p in meta['inner_provenance']}!=expected_original:raise ValueError('Incomplete nested fits')
        for inner in meta['inner_provenance']:
            if set(inner['train'])!=expected_original-{inner['inner_test']}:raise ValueError('Incorrect nested pool')
        for neural in meta['neural'].values():
            if set(neural['train'])!=set(train) or set(neural['excluded'])!=set(test) or not neural['normalizer_recomputed_train_only']:
                raise ValueError('Neural provenance/normalization incomplete')
    historical=evaluate(payload,'strict_source',proof)
    paired=evaluate_paired_audit(payload,'strict_source','strict_v4',proof)
    control=evaluate(payload,'strict_v4',proof)
    rows=payload['records'];index={(r['candidate'],r['game'],r['scenario']):r for r in rows}
    providers={}
    for provider,prefix in [('DFL','original/idsse_'),('Metrica','original/metrica_'),('SkillCorner','skillcorner/')]:
        chosen=[r for r in rows if r['candidate']=='strict_source' and r['game'].startswith(prefix) and r['scenario'] in ['matched_seed0','natural_detected']]
        providers[provider]={'games':len(chosen),'paired_primary_gain':statistics.mean(r['spell_f1_mean']-index['strict_v4',r['game'],r['scenario']]['spell_f1_mean'] for r in chosen)}
    provider_pass=min(x['paired_primary_gain'] for x in providers.values())>=-.01-1e-12
    result={
        'candidate':'source_seed20261113','all_stage_wholefold_isolation':'PASS',
        'scope':'Retrospective audit on29reused research matches; this is not untouched confirmation.',
        'strict_paired':{k:paired[k] for k in ['status','baseline_composite','candidate_composite','gain','descriptive_95ci','game_wins','game_losses','checks']},
        'against_historical_frozen_v4':{k:historical[k] for k in ['status','baseline_composite','candidate_composite','gain','game_wins','game_losses','checks']},
        'strict_reconstructed_v4_composite':control['candidate_composite'],
        'provider_guards':providers,'provider_guards_pass':provider_pass,
        'numeric_audit_pass':paired['status']=='ELIGIBLE_FOR_CONFIRMATION' and historical['status']=='ELIGIBLE_FOR_CONFIRMATION' and provider_pass,
        'independent_confirmation':'PENDING; no fresh final data opened',
        'release':'Keep verified v4; no research promotion before unopened confirmation.',
        'source_sha256':{'scores':sha(OUT/'results.json'),'provenance':sha(OUT/'provenance.json'),'plan':sha(DOC/'plan.json'),'checker':sha(ROOT/'tools/check_evaluation.py')},
    }
    write(DOC/'decision.json',result);write(DOC/'results.json',payload);write(DOC/'provenance.json',proof)
    write(DOC/'paired_gate.json',paired);write(DOC/'historical_gate.json',historical)
    write(DOC/'period_attestation.json',json.loads((OUT/'period_attestation.json').read_text()))
    text=[
        '# Complete inherited-fold audit', '',
        '**All-stage isolation: PASS. Independent final confirmation: PENDING. Verified v4 remains the release.**','',
        'The source-balanced seed20261113 candidate was frozen before this audit. The original fivefold mapping,29games,67scenario cells, official metric, thresholds and decoder are unchanged. No runner-up/seed selection or post-audit threshold tuning occurs.','',
        'Earlier scores excluded each scored game from inherited branches, but other members of its neural fold could remain. Here every original-game first-stage fit, every inner fit feeding the stack, the stack, inherited nineteen-game team head, neural checkpoint and normalization excludes the complete outer fold. Actual checkpoint hashes and training ancestry were checked. Neural normalizers were recomputed using outer training games only. Independent geometry/metadata period spans equal cached spans on all29games.','',
        'Only inherited cross-validation weights are rebuilt. The deployed full-data training pools and frozen full candidate weights do not change. Therefore the previously repeated candidate CSV and v4 package remain byte-for-byte untouched.','',
        '| Comparison | Control spellF1 | Candidate spellF1 | Gain | Paired game wins |',
        '|---|---:|---:|---:|---:|',
        f"| Same complete-fold reconstruction | {paired['baseline_composite']:.5f} | {paired['candidate_composite']:.5f} | {paired['gain']:+.5f} | {paired['game_wins']}/29 |",
        f"| Historical frozen v4 (different inherited exclusions) | {historical['baseline_composite']:.5f} | {historical['candidate_composite']:.5f} | {historical['gain']:+.5f} | {historical['game_wins']}/29 |",'',
        f"The strict paired descriptive95%game-bootstrap interval is [{paired['descriptive_95ci'][0]:+.5f}, {paired['descriptive_95ci'][1]:+.5f}]. Games average their two primary scenarios before win counting. Resampling does not correct repeated model selection, shared teams or provider/label dependence.",'',
        '| Provider | Primary games | Strict paired mean gain |','|---|---:|---:|',
    ]
    text += [f"| {name} | {value['games']} | {value['paired_primary_gain']:+.5f} |" for name,value in providers.items()]
    text += ['',f"Both numeric comparisons and provider guards pass: **{result['numeric_audit_pass']}**.",'',
             '## Failures and limits','',
             'The immutable historical comparator and reconstructed strict comparator are reported separately; the historical baseline was not rewritten. All numeric pass/fail checks and game/cohort losses are retained in paired_gate.json and historical_gate.json. Both development gates must pass; neither produces a final-test PASS. All40previously screened games remain research, and proxy references differ among providers. PFF access/label eligibility and an unopened final test remain required. No organizer labels were used, no competition quality score was computed and no submission/message/publication occurs in this audit.','']
    failed=[k for k,v in paired['checks'].items() if not v]
    if failed:text += ['Failed strict numeric checks: '+', '.join(failed),'']
    (DOC/'REPORT.md').write_text('\n'.join(text))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
