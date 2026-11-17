"""Unlabelled input diagnostics; no prediction, reference loading or fitting."""
import os,sys,json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
def guard(event,args):
 if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
  name=os.fsdecode(args[0])
  if '/evaluation/gt/' in name or 'possession_gt' in name or '/references/' in name:raise PermissionError('Possession reference access blocked')
sys.addaudithook(guard)
from mtah import atd
from mtah.config import HALVES,PITCH
from provider_stress import corrupt_geometry

def q(x):
 x=np.asarray(x);x=x[np.isfinite(x)]
 return {str(p):float(np.quantile(x,p)) for p in [.5,.9,.95,.99]} if len(x) else {}
def stats(d,parts):
 players=d['players'][...,:2]*np.asarray(d.get('pitch',PITCH));ball=d['ball'][:,:2]
 live=np.zeros(len(ball),bool)
 gap=[];speeds=[];curve=[]
 for a,b in parts:
  live[a:b]=True;seen=np.isfinite(ball[a:b]).all(axis=1);edge=np.diff(np.r_[0,(~seen).astype(np.int8),0]);gap.extend((np.flatnonzero(edge==-1)-np.flatnonzero(edge==1))/25)
  xy=players[a:b];speeds.extend(np.linalg.norm(np.diff(xy,axis=0),axis=2).ravel()*25)
  curve.extend(np.linalg.norm(xy[2:]-2*xy[1:-1]+xy[:-2],axis=2).ravel())
 good=np.isfinite(players[live]).all(axis=2);count=good.sum(axis=1);provided=d['players'][live,:,2]
 return dict(ball_visible_fraction=float(np.isfinite(ball[live]).all(axis=1).mean()),player_count_mean=float(count.mean()),player_count_quantiles=q(count),all_players_missing_fraction=float((count==0).mean()),ball_gap_seconds=q(gap),adjacent_coordinate_speed_mps=q(speeds),provided_speed_mps=q(provided),complete_triplet_second_difference_m=q(curve))
def main():
 os.environ['MTAH_DATA_DIR']=str(ROOT.parent/'data')
 # config was imported above: expose only the two allowed tactical input paths.
 import mtah.config as cfg
 cfg.DATA_DIR=str(ROOT.parent/'data')
 from mtah import atd
 target=atd.load();parts=[(a,b+1) for _,a,b in HALVES];target['pitch']=np.array(PITCH)
 sourcepath=ROOT/'external/pff/converted/3816_geometry.npz'
 preflight=json.loads((ROOT/'reports/pff_access/kloppy_3816_preflight.json').read_text())
 assert hashlib.sha256(sourcepath.read_bytes()).hexdigest()==preflight['geometry_cache_sha256']
 with np.load(sourcepath,allow_pickle=False) as z:source={k:z[k] for k in z.files}
 assert 'gt' not in source
 edges=np.r_[0,np.flatnonzero(np.diff(source['period']))+1,len(source['period'])];sourceparts=[(int(a),int(b)) for a,b in zip(edges[:-1],edges[1:]) if source['period'][a]>0]
 result={'target':stats(target,parts),'training3816':stats(source,sourceparts),'source_stress':{},'stationary_probe':{},'reference_labels_read':False,'quality_scored':False}
 n=25*120;p=np.zeros((n,22,3),np.float32);p[:,:,:2]=.5;ball=np.full((n,3),.5,np.float32);period=np.ones(n,np.int8);pitch=np.array(PITCH)
 presets=json.loads((ROOT/'tools/provider_stress_presets.json').read_text())['presets']
 print('preset keys',list(presets),flush=True)
 for name in ['hz10_noise0p5','hz15_noise0p5','hz10_noise2','hz15_noise2']:
  spec=presets[name];native=dict(players=source['players'],ball=source['ball'],period=source['period'],timestamps=np.arange(len(source['ball']))/25,source_frame=np.arange(len(source['ball'])))
  damaged,_=corrupt_geometry(native,source['players'],source['ball'],source['period'],source['pitch'],25,spec,20261113);damaged['pitch']=source['pitch'];result['source_stress'][name]=stats(damaged,sourceparts)
  native=dict(players=p,ball=ball,period=period,timestamps=np.arange(n)/25,source_frame=np.arange(n));damaged,receipt=corrupt_geometry(native,p,ball,period,pitch,25,spec,20261113)
  result['stationary_probe'][name]={'all_players_missing_fraction':float((~np.isfinite(damaged['players'][...,:2]).all(axis=2).any(axis=1)).mean()),'reported_player_speed_mps':q(damaged['players'][:,:,2]),'note':'True speed is zero. White coordinate noise creates these apparent velocities; no clipping/correlation applied.'}
 result['input_sha256']={str(p.relative_to(ROOT.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT.parent/'data/tactical/BAR-ATH_tactical_atd.xml',ROOT.parent/'data/tactical/BAR-ATH_tactical_atd.txt',sourcepath]}
 result['code_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();result['plan_sha256']=hashlib.sha256((ROOT/'docs/experiments/sensor-relevance/plan.json').read_bytes()).hexdigest()
 (ROOT/'docs/experiments/sensor-relevance/results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
