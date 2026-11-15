"""Freeze one changed rule on development; controls cannot repair known transfer failure."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
ROOT=Path(__file__).resolve().parents[1]


def main():
    folder=ROOT/'reports/robust_followup';out=folder/'selection.json'
    if out.exists():raise ValueError('Followup selection already frozen')
    summary=json.loads((folder/'development_summary.json').read_text());plan=json.loads((ROOT/'docs/experiments/robust-followup/plan.json').read_text())
    # Old/long controls are known endpoints, not new corrective candidates.
    eligible=[name for name,r in summary.items() if r['status']=='ELIGIBLE_FOR_CONFIRMATION' and name not in ['robust_old_control','robust_long_control']]
    if not eligible:result={'candidate':None,'status':'No changed rule passed development; retain v4'}
    else:
        name=max(eligible,key=lambda n:summary[n]['candidate_composite']);rule=next(r for r in plan['rules'] if 'robust_'+r['name']==name)
        result={'candidate':name,'temporal_blend':rule,'development':summary[name],'status':'Frozen local rule; seed diagnostics and reused-source transfer guards pending'}
    result.update(frozen_utc=datetime.now(timezone.utc).isoformat(),summary_sha256=hashlib.sha256((folder/'development_summary.json').read_bytes()).hexdigest(),parent_bundle_sha256=hashlib.sha256((ROOT/'assets/research-v5/bundle.json').read_bytes()).hexdigest(),external_followup_scores_seen=False,release='v4 unchanged',user_hold='No submission/messages/push/publication')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
