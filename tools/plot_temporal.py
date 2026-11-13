"""Plot aggregate game-isolated results; never read tracking/reference arrays."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'docs/experiments/temporal'
r=json.loads((OUT/'cv_results.json').read_text())['records'];games=sorted({z['game'] for z in r});delta=[]
for game in games:
 primary=['matched_seed0','label_blind_seed1'] if game.startswith('original/') else ['natural_detected','label_blind_seed7']
 mean=lambda c:np.mean([z['spell_f1_mean'] for z in r if z['game']==game and z['candidate']==c and z['scenario'] in primary])
 delta.append(mean('fusion50')-mean('v3'))
fig,axes=plt.subplots(1,2,figsize=(13,10),gridspec_kw={'width_ratios':[1.1,1]})
y=np.arange(len(games));axes[0].barh(y,delta,color=['#13824a' if v>=0 else '#c67d26' for v in delta]);axes[0].axvline(0,color='#555',linewidth=.8);axes[0].set_yticks(y,[g.replace('original/idsse_','DFL ').replace('original/metrica_','Metrica ').replace('skillcorner/','SC ') for g in games],fontsize=9);axes[0].invert_yaxis();axes[0].set_xlabel('Paired spellF1 change: two-scenario game mean');axes[0].set_title('All29 game-excluded comparisons:26wins /3losses');axes[0].grid(axis='x',alpha=.2)
s=json.loads((OUT/'selection.json').read_text())['scores'];labels=['Original matched (9)','Original label-blind (9)','Original stress (9)','Skill natural (20)','Skill label-blind (20)'];y=np.arange(5)
axes[1].barh(y-.18,s['v3']['means'],.35,label='V3',color='#8b949e');axes[1].barh(y+.18,s['fusion50']['means'],.35,label='V4 / temporal fusion',color='#147d92');axes[1].set_yticks(y,labels,fontsize=9);axes[1].invert_yaxis();axes[1].set_xlim(0,1);axes[1].set_xlabel('Mean official spellF1');axes[1].set_title('Five predefined source/sensor cohorts');axes[1].legend(frameon=False,loc='lower right');axes[1].grid(axis='x',alpha=.2)
fig.suptitle('Temporal fusion: whole-game validation evidence',fontsize=15);fig.text(.5,.015,'Five-fold neural CV on reused research games; all augmentation siblings excluded together. Not29fresh tests.',ha='center',fontsize=10);fig.tight_layout(rect=[0,.04,1,.96]);fig.savefig(OUT/'game_comparison.png',dpi=150);plt.close(fig)
