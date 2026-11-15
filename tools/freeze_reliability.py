"""Freeze one numerically eligible research candidate before full fitting."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,subprocess
ROOT=Path(__file__).resolve().parents[1]


def main():
    out=ROOT/'reports/reliability/selection.json'
    if out.exists():raise ValueError('Selection already frozen')
    files=[('adaptive',ROOT/'reports/reliability/adaptive_summary.json'),('long',ROOT/'reports/reliability/long_summary.json'),('spell',ROOT/'reports/spell_trial/summary.json')]
    if not all(p.exists() for _,p in files):raise ValueError('Complete every registered study before selecting')
    candidates={family+'/'+key:value for family,path in files for key,value in json.loads(path.read_text()).items()}
    eligible=[k for k,v in candidates.items() if v['status']=='ELIGIBLE_FOR_CONFIRMATION']
    if not eligible:result={'candidate':None,'release':'v4 unchanged; no numerically eligible candidate'}
    else:
        name=max(eligible,key=lambda n:candidates[n]['candidate_composite']);family,candidate=name.split('/')
        seeds=[int(candidate.split('_')[1][4:])] if candidate.startswith('long_seed') else [int(candidate[len('spell_seed'):])] if candidate.startswith('spell_seed') else [20261113,20261114,20261115]
        rule=candidate[len('long_ensemble_'):] if candidate.startswith('long_ensemble_') else 'uniform50'
        spec=next(s for s in json.loads((ROOT/'docs/experiments/reliability/plan.json').read_text())['rules'] if s['name']==rule)
        result={'candidate':name,'family':family,'seeds':seeds,'rule':spec,'development':candidates[name],'status':'Frozen research candidate; inherited-lineage audit and independent confirmation pending','release':'v4 unchanged'}
    result.update(frozen_utc=datetime.now(timezone.utc).isoformat(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),summary_sha256={family:hashlib.sha256(path.read_bytes()).hexdigest() for family,path in files},external_candidate_scores_seen_this_cycle=False,user_hold='Local only; no submission, messages, push or publication')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
