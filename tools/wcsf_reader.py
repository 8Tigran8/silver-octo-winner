"""Research-only unseen WCSF match: predictor geometry separated from references."""
from pathlib import Path
import json,zipfile
import numpy as np
import pandas as pd
from skillcorner_reader import upsample_observations
ROOT=Path(__file__).resolve().parents[1]
GEOMETRY=['x','y','group_id','timestamp_ms','x_ball','y_ball','player_id']


def unanimous_reference(df):
    alive=df.AliveFlag.to_numpy();owner=df.PossessionFlag.to_numpy();df=df.copy()
    df['reference_label']=np.where(alive==0,2,np.where((alive==1)&(owner==1),0,np.where((alive==1)&(owner==2),1,3))).astype(np.int8)
    grouped=df.groupby('timestamp_ms')['reference_label'];values=grouped.first();conflict=grouped.nunique(dropna=False)>1;values.loc[conflict]=3
    return values,int(conflict.sum())


def half(path):
    with zipfile.ZipFile(path) as z:
        name=next(n for n in z.namelist() if n.endswith('.csv') and not n.startswith('__'))
        with z.open(name) as f:df=pd.read_csv(f,usecols=GEOMETRY)
    df=df[df.group_id.isin([1,2])];times=np.sort(df.timestamp_ms.unique());start=int(times[0]);end=int(times[-1])
    if not np.allclose((times-start)/100,np.round((times-start)/100)):raise ValueError('Unexpected WCSF clock grid')
    n=(end-start)//100+1;ids=sorted({(int(r.group_id),int(r.player_id)) for r in df[['group_id','player_id']].drop_duplicates().itertuples(index=False)})
    teams=np.array([g-1 for g,p in ids],np.int8);slots={identity:i for i,identity in enumerate(ids)}
    balls={g:df[df.group_id==g].groupby('timestamp_ms')[['x_ball','y_ball']].first() for g in [1,2]}
    common=balls[1].index.intersection(balls[2].index);a=balls[1].loc[common].to_numpy();b=balls[2].loc[common].to_numpy()
    signs=np.array([min([1,-1],key=lambda s:float(np.nanmedian(abs(a[:,j]-s*b[:,j])))) for j in [0,1]])
    errors=np.nanmedian(abs(a-signs*b),axis=0)
    if not np.isfinite(errors).all() or (errors>.05).any():raise ValueError('Cannot reconcile team-relative coordinate frames')
    players=np.full((n,len(ids),3),np.nan,np.float32);ball=np.full((n,3),np.nan,np.float32)
    for group in [1,2]:
        rows=df[df.group_id==group];ix=np.round((rows.timestamp_ms.to_numpy()-start)/100).astype(int)
        slot=np.array([slots[group,int(p)] for p in rows.player_id]);xy=rows[['x','y']].to_numpy()
        if group==2:xy=xy*signs
        if rows.duplicated(['timestamp_ms','player_id']).any():raise ValueError('Repeated player timestamp')
        players[ix,slot,:2]=xy
        b=balls[group];ix=np.round((b.index.to_numpy()-start)/100).astype(int);xy=b.to_numpy()
        if group==2:xy=xy*signs
        keep=~np.isfinite(ball[ix,:2]).all(1);ball[ix[keep],:2]=xy[keep]
    speed=np.linalg.norm(np.diff(players[:,:,:2],axis=0),axis=2)*10;players[1:,:,2]=speed
    pitch=np.array([105.,68.]);players[:,:,:2]=players[:,:,:2]/pitch+.5;ball[:,:2]=ball[:,:2]/pitch+.5
    pos=np.arange(int((n-1)*2.5)+1)/2.5
    return {'players':upsample_observations(players,pos),'ball':upsample_observations(ball,pos),'slot_team':teams,'pitch':pitch}, {'source_start_ms':start,'source_end_ms':end,'source_frames':n,'upsampled_frames':len(pos),'away_coordinate_signs':signs.tolist(),'median_ball_disagreement_m':errors.tolist(),'slots':ids}


def geometry(folder):
    parts=[];metadata=[]
    for name in ['positions_FIRST_HALF.csv.zip','positions_SECOND_HALF.csv.zip']:
        data,meta=half(Path(folder)/name);parts.append(data);metadata.append(meta)
    ids=sorted({tuple(identity) for m in metadata for identity in m['slots']});teams=np.array([g-1 for g,p in ids],np.int8);slot={v:i for i,v in enumerate(ids)}
    n=sum(m['upsampled_frames'] for m in metadata)+250;players=np.full((n,len(ids),3),np.nan,np.float32);ball=np.full((n,3),np.nan,np.float32);period=np.zeros(n,np.int8)
    offset=0;spans=[]
    for i,(data,meta) in enumerate(zip(parts,metadata)):
        end=offset+len(data['ball']);ball[offset:end]=data['ball']
        for j,identity in enumerate(meta['slots']):players[offset:end,slot[tuple(identity)]]=data['players'][:,j]
        period[offset:end]=i+1;spans.append([offset,end]);offset=end+250
    return {'players':players,'ball':ball,'period':period,'slot_team':teams,'pitch':np.array([105.,68.])},{'halves':metadata,'segments':spans,'purpose':'Predictor positions/clock only; AliveFlag/PossessionFlag/team_game_phase excluded from geometry reads'}


def labels(folder,selection_path):
    from quality_audit import sha
    selection=json.loads(Path(selection_path).read_text())
    if selection['plan_sha256']!=sha(ROOT/'docs/experiments/temporal/plan.json'):raise ValueError('No frozen candidate')
    gt_parts=[];quality=[]
    for name in ['positions_FIRST_HALF.csv.zip','positions_SECOND_HALF.csv.zip']:
        with zipfile.ZipFile(Path(folder)/name) as z:
            name=next(n for n in z.namelist() if n.endswith('.csv') and not n.startswith('__'))
            with z.open(name) as f:df=pd.read_csv(f,usecols=['timestamp_ms','AliveFlag','PossessionFlag'])
        d,conflicts=unanimous_reference(df)
        # Source rows occasionally disagree at the same timestamp. Never choose
        # a majority or a model-favoured label. Unanimous-reference frames only.
        start=int(d.index.min());n=(int(d.index.max())-start)//100+1;gt=np.full(n,3,np.int8);ix=np.round((d.index.to_numpy()-start)/100).astype(int);gt[ix]=d.to_numpy()
        pos=np.arange(int((n-1)*2.5)+1)/2.5;up=gt[np.floor(pos).astype(int)];gt_parts.append(up)
        quality.append({'source_timestamps':len(d),'ambiguous_source_frames':conflicts,'upsampled_frames':len(up),'known_reference_frames':int((up<3).sum()),'unknown_reference_frames':int((up==3).sum())})
    out=ROOT/'reports/temporal/external/wcsf_reference_quality.json'
    out.write_text(json.dumps({'policy':'Map each source row toH/A/D, require unanimity within timestamp; contradictory/unknown frames remainX. Dead ownership does not matter. No majority vote or model dependence.','halves':quality},indent=2)+'\n')
    return np.concatenate([gt_parts[0],np.full(250,3,np.int8),gt_parts[1]])


if __name__=='__main__':
    out=ROOT/'reports/temporal/external';out.mkdir(parents=True,exist_ok=True)
    data,meta=geometry(ROOT/'external/wcsf');np.savez_compressed(out/'wcsf_geometry.npz',**data);(out/'wcsf_geometry.json').write_text(json.dumps(meta,indent=2)+'\n');print('Geometry prepared without processing reference flags')
