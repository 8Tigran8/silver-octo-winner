"""Plot aggregate, reused-development comparisons for the selected local candidate."""
from pathlib import Path
import json,argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--study',choices=['long','source'],default='long');args=p.parse_args()
D=ROOT/'docs/experiments'/('source-balance' if args.study=='source' else 'reliability');S=ROOT/'reports'/('source_balance' if args.study=='source' else 'reliability')
s=json.loads((S/'selection.json').read_text());key=s['candidate'].split('/')[-1];r=json.loads((S/(key+'_gate.json')).read_text())
fig,(left,right)=plt.subplots(1,2,figsize=(13,9),gridspec_kw={'width_ratios':[1.05,1]});fig.patch.set_facecolor('white')
games=sorted(r['games'],key=lambda x:x['game']);y=np.arange(len(games));delta=[g['gain'] for g in games]
labels=[g['game'].replace('original/idsse_','DFL ').replace('original/metrica_','Metrica ').replace('skillcorner/','SC ') for g in games]
left.barh(y,delta,color='#087f8c');left.set_yticks(y,labels,fontsize=8);left.invert_yaxis();left.axvline(0,color='#555',lw=.8);left.set_xlabel('Paired per-game spellF1 gain');left.set_title('Selected candidate: 29 wins / 0 losses',fontsize=12);left.grid(axis='x',alpha=.18)
cohorts=r['cohorts'];names=['Original matched (9)','Original blind (9)','Original stress (9)','Skill natural (20)','Skill blind (20)'];yy=np.arange(5)
right.barh(yy-.18,[c['baseline'] for c in cohorts],height=.34,color='#a2aab3',label='v4');right.barh(yy+.18,[c['candidate'] for c in cohorts],height=.34,color='#087f8c',label='Local candidate')
right.set_yticks(yy,names,fontsize=9);right.invert_yaxis();right.set_xlim(0,.8);right.grid(axis='x',alpha=.18);right.set_xlabel('Mean official spellF1');right.set_title('Fixed source and sensor cohorts',fontsize=12);right.legend(loc='lower right',frameon=False)
fig.suptitle(('Provider-balanced training' if args.study=='source' else 'Longer temporal training')+f": {r['baseline_composite']:.4f} to {r['candidate_composite']:.4f}",fontsize=16,y=.98)
fig.text(.5,.025,'Reused research games, not an untouched test. Legacy components exclude each scored game;\nnew temporal fits exclude the entire scored fold. Independent confirmation and stricter inherited-fold review remain pending.',ha='center',fontsize=8,color='#555')
fig.tight_layout(rect=[0,.07,1,.95]);fig.savefig(D/'game_comparison.png',dpi=150);plt.close(fig)
