"""Freeze one equal-provider model after complete development results."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,subprocess
ROOT=Path(__file__).resolve().parents[1]


def main():
    folder=ROOT/'reports/source_balance';path=folder/'selection.json'
    if path.exists():raise ValueError('Selection already frozen')
    summary=json.loads((folder/'summary.json').read_text());rows=json.loads((folder/'results.json').read_text())['records']
    baseline={(r['game'],r['scenario']):r for r in json.loads((ROOT/'docs/experiments/temporal/cv_results.json').read_text())['records'] if r['candidate']=='fusion50'}
    import statistics
    provider={}
    for candidate in summary:
        provider[candidate]={}
        for name,prefix in [('DFL','original/idsse_'),('Metrica','original/metrica_'),('SkillCorner','skillcorner/')]:
            subset=[r for r in rows if r['candidate']==candidate and r['game'].startswith(prefix) and r['scenario'] in ['matched_seed0','natural_detected']]
            provider[candidate][name]=statistics.mean(r['spell_f1_mean']-baseline[r['game'],r['scenario']]['spell_f1_mean'] for r in subset)
    (folder/'provider_guards.json').write_text(json.dumps(provider,indent=2)+chr(10))
    eligible=[n for n,r in summary.items() if r['status']=='ELIGIBLE_FOR_CONFIRMATION' and min(provider[n].values())>=-.01-1e-12]
    if not eligible:result={'candidate':None,'status':'No source-balanced candidate passed development; v4 unchanged'}
    else:
        name=max(eligible,key=lambda n:summary[n]['candidate_composite']);seeds=[int(name[len('source_seed'):])] if name.startswith('source_seed') else [20261113,20261114,20261115]
        result={'candidate':name,'family':'source','seeds':seeds,'rule':{'name':'uniform50','kind':'fixed','dead':.5,'home':.5},'development':summary[name],'status':'Frozen source-balanced research candidate; reused-source guards and independent confirmation pending'}
    result.update(frozen_utc=datetime.now(timezone.utc).isoformat(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),summary_sha256=hashlib.sha256((folder/'summary.json').read_bytes()).hexdigest(),external_source_balance_scores_seen=False,release='v4 unchanged',user_hold='No submission/messages/push/publication')
    path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
