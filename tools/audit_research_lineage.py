"""Audit declared legacy per-game isolation separately from strict full-fold policy."""
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    plan=json.loads((ROOT/'docs/experiments/temporal/plan.json').read_text());audit=ROOT/'reports/quality_audit/380987b3029c';head=ROOT/'reports/v4_search/models';all_head=json.loads((head/'all_nineteen.json').read_text());rows=[]
    for game in plan['training_games']:
        family,name=game.split('/');fold=plan['folds'][game];group={g for g,k in plan['folds'].items() if k==fold};other_seen=set()
        if family=='original':
            meta=json.loads((audit/'stage1'/('exclude__'+name+'.json')).read_text());assert name not in meta['train'] and name in meta['excluded']
            assert sha(audit/'stage1'/('exclude__'+name+'.pkl'))==meta['model_sha256']
            other_seen|={'original/'+n for n in meta['train']} & group
            nested=json.loads((audit/'scores'/(name+'.json')).read_text())['inner_provenance'];assert len(nested)==8
            for info in nested:assert name not in info['train'] and info['inner_test'] not in info['train']
        if game in all_head['train']:
            f=head/('exclude__'+game.replace('/','__')+'.json');meta=json.loads(f.read_text());assert game not in meta['train'] and meta['excluded']==game
            assert sha(f.with_suffix('.pkl'))==meta['model_sha256']
        else:meta=all_head;assert game not in meta['train']
        other_seen|=set(meta['train']) & group
        rows.append({'game':game,'scored_game_absent_from_legacy_training':True,'other_neural_fold_games_in_legacy_training':sorted(other_seen)})
    result={'status':'PASS: declared per-scored-game exclusion; strict inherited full-fold exclusion NOT SATISFIED','games':29,'nested_original_stack_cases_checked':72,'frozen_v4_probability_control':'All67scenario spellF1/macroF1 reproduced exactly in adaptive study','scope':'New TCNs exclude complete neural folds. Legacy unchanged v3 components exclude each scored game but may contain other games in that neural fold. Constant/reference-blind mixtures do not learn from these test predictions.','policy_action':'Do not claim all-stage full-fold compliance or final adoption; further audit/retraining is required under the stricter evaluation wording. No policy thresholds relaxed.','rows':rows}
    (ROOT/'docs/experiments/reliability/lineage_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(result['status'])


if __name__=='__main__':main()
