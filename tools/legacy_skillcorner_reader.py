"""Read the public 2020 SkillCorner release without inventing dead-ball GT.

Only provider-owned H/A frames have reference labels. Missing possession is X,
not D. Model period spans derive from the independent tracking period field.
Anonymous tracks use finite team slots; no ownership annotation enters features.
"""
import json
from pathlib import Path
import numpy as np
from skillcorner_reader import upsample_observations


def period_spans(period):
    spans=[]
    for value in (1,2):
        edges=np.diff(np.r_[0,(period==value).astype(np.int8),0])
        spans.extend(zip(np.flatnonzero(edges==1).tolist(),np.flatnonzero(edges==-1).tolist()))
    return sorted(spans)


def read_match(folder):
    folder=Path(folder)
    meta=json.loads((folder/"match_data.json").read_text())
    frames=json.loads((folder/"structured_data.json").read_text())
    n=max(r["frame"] for r in frames)+1
    pitch=np.array([meta["pitch_length"],meta["pitch_width"]],float)
    home=meta["home_team"]["id"];away=meta["away_team"]["id"]
    teams={p["trackable_object"]:0 if p["team_id"]==home else 1 for p in meta["players"] if p["team_id"] in (home,away)}
    ball_id=meta["ball"]["trackable_object"]
    ball=np.full((n,3),np.nan,np.float32)
    players=np.full((n,46,3),np.nan,np.float32)
    track_codes=np.full((n,46),-1,np.int32);identities={}
    period=np.zeros(n,np.int8);gt=np.full(n,3,np.int8)
    slots={};previous={};last=-1
    for row in frames:
        index=row["frame"]
        if index<=last:raise ValueError("Legacy frame order is invalid")
        last=index;period[index]=row.get("period") or 0
        group=row.get("possession",{}).get("group")
        if group is not None:
            if group not in ("home team","away team","home goalkeeper","away goalkeeper"):
                raise ValueError("Unknown possession-reference group")
            if period[index] in (1,2):gt[index]=0 if group.startswith("home") else 1
        objects=[]
        for item in row["data"]:
            identity=item.get("trackable_object");group=item.get("group_name","")
            if identity==ball_id or group=="ball":
                ball[index]=[item.get(k) if item.get(k) is not None else np.nan for k in ("x","y","z")]
                continue
            team=teams.get(identity)
            if team is None:
                team=0 if group.startswith("home") else 1 if group.startswith("away") else None
            if team is None:continue
            key=(team,item.get("track_id",identity))
            objects.append((key,item))
        active={k for k,_ in objects}
        slots={k:v for k,v in slots.items() if k in active}
        current={}
        for key,item in objects:
            team=key[0]
            if key not in slots:
                free=[s for s in range(team*23,(team+1)*23) if s not in slots.values()]
                if not free:raise ValueError("More than 23 visible tracks per team")
                slots[key]=free[0]
            slot=slots[key];xy=np.array([item["x"],item["y"]],float)
            if key not in identities:identities[key]=len(identities)
            track_codes[index,slot]=identities[key]
            players[index,slot,:2]=xy/pitch+.5
            if key in previous and previous[key][0]==index-1:
                players[index,slot,2]=np.linalg.norm(xy-previous[key][1])*10
            current[key]=(index,xy)
        previous=current
    ball[:,:2]=ball[:,:2]/pitch+.5
    times=np.arange(int((n-1)*2.5)+1)/2.5;source=np.floor(times).astype(int)
    resampled=upsample_observations(players,times)
    right=np.minimum(source+1,n-1)
    switched=(track_codes[source]!=track_codes[right]) & ((times-source)[:,None]>0)
    resampled[switched]=np.nan
    return {"ball":upsample_observations(ball,times),"players":resampled,
            "slot_team":np.repeat([0,1],23).astype(np.int8),"pitch":pitch,
            "gt":gt[source],"period":period[source]}
