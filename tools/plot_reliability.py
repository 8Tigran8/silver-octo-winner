"""Plot aggregate, reused-development comparisons for the selected local candidate."""
from pathlib import Path
import json,argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--study',choices=['long','source','strict'],default='long');args=p.parse_args()
D=ROOT/'docs/experiments'/({'source':'source-balance','long':'reliability','strict':'strict-fullfold'}[args.study])
if args.study=='strict':r=json.loads((D/'paired_gate.json').read_text())
else:
    S=ROOT/'reports'/('source_balance' if args.study=='source' else 'reliability')
    s=json.loads((S/'selection.json').read_text());key=s['candidate'].split('/')[-1];r=json.loads((S/(key+'_gate.json')).read_text())
fig,(left,right)=plt.subplots(1,2,figsize=(13,9),gridspec_kw={'width_ratios':[1.05,1]});fig.patch.set_facecolor('white')
games=sorted(r['games'],key=lambda x:x['game']);y=np.arange(len(games));delta=[g['gain'] for g in games]
labels=[g['game'].replace('original/idsse_','DFL ').replace('original/metrica_','Metrica ').replace('skillcorner/','SC ') for g in games]
left.barh(y,delta,color='#087f8c');left.set_yticks(y,labels,fontsize=8);left.invert_yaxis();left.axvline(0,color='#555',lw=.8);left.set_xlabel('Paired per-game spellF1 gain');left.set_title(f"Selected candidate: {r['game_wins']} wins / {r['game_losses']} losses",fontsize=12);left.grid(axis='x',alpha=.18)
cohorts=r['cohorts'];names=['Original matched (9)','Original blind (9)','Original stress (9)','Skill natural (20)','Skill blind (20)'];yy=np.arange(5)
right.barh(yy-.18,[c['baseline'] for c in cohorts],height=.34,color='#a2aab3',label='v4 under same exclusions' if args.study=='strict' else 'v4');right.barh(yy+.18,[c['candidate'] for c in cohorts],height=.34,color='#087f8c',label='Local candidate')
right.set_yticks(yy,names,fontsize=9);right.invert_yaxis();right.set_xlim(0,.8);right.grid(axis='x',alpha=.18);right.set_xlabel('Mean official spellF1');right.set_title('Fixed source and sensor cohorts',fontsize=12);right.legend(loc='lower right',frameon=False)
fig.suptitle({'source':'Provider-balanced training','long':'Longer temporal training','strict':'All-stage whole-fold audit'}[args.study]+f": {r['baseline_composite']:.4f} to {r['candidate_composite']:.4f}",fontsize=16,y=.98)
footnote=('All learned stages, inner fits and normalization exclude the entire scored fold.\nReused research games; independent confirmation remains pending. Game wins average two primary scenarios.' if args.study=='strict' else 'Reused research games, not an untouched test. Legacy components exclude each scored game;\nnew temporal fits exclude the entire scored fold. Independent confirmation and stricter inherited-fold review remain pending.')
fig.text(.5,.025,footnote,ha='center',fontsize=8,color='#555')
fig.tight_layout(rect=[0,.07,1,.95]);fig.savefig(D/'game_comparison.png',dpi=150);plt.close(fig)
