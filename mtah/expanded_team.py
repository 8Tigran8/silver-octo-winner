"""Public-game team-head training; competition observations never enter fitting."""
from pathlib import Path
import json
import numpy as np
from . import degrade,features2,metrics,possession_context as context
from .config import ROOT
from tools.skillcorner_reader import read_match

ROOT=Path(ROOT)
CONFIG=Path(__file__).with_name("expanded_team_training.json")
ORIGINAL_SCENARIOS={
    "matched_seed0":(0,{},12),
    "label_blind_seed1":(1,{"mode":"v1","ball_gap_mean_s":6.,"frag_mean_s":107.,
        "frag_gap_mean_s":6.,"skip_ball_if_seen_below":0.},12),
    "clean_source":(None,None,12)}
BLIND={"mode":"v1","skip_ball_if_seen_below":0.,"ball_seen_target":.4,
       "ball_gap_mean_s":6.,"frag_mean_s":107.,"frag_gap_mean_s":6.,
       "pos_noise_m":.3,"speed_noise":.6}
SKILL_SCENARIOS={"natural_detected":(None,None,8),"label_blind_seed7":(7,BLIND,8)}


def games():
    config=json.loads(CONFIG.read_text())
    return ["original/"+n for n in config["original"]]+["skillcorner/"+n for n in config["skillcorner"]]


def assert_training_split(all_games,train,excluded):
    if len(all_games)!=len(set(all_games)) or len(train)!=len(set(train)):
        raise ValueError("Duplicate training game")
    expected=set(all_games)-({excluded} if excluded is not None else set())
    if set(train)!=expected or (excluded is not None and excluded in train):
        raise ValueError("Scored match leaked into expanded training")


def public_game(game):
    if game not in games():raise ValueError("Only whitelisted public training games may be loaded")
    family,name=game.split("/")
    if family=="original":
        with np.load(ROOT/"external/conv"/(name+".npz"),allow_pickle=False) as z:return {k:z[k] for k in z.files}
    return read_match(ROOT/"external/skillcorner"/name,name)


def feature_pair(data,segments=None):
    segments=segments if segments is not None else metrics.valid_runs(data["gt"])
    x,names=features2.build(data,segments);xs,_=features2.build(data,segments,swap=True)
    x,columns=context.extend(x,names,segments);xs,swapped=context.extend(xs,names,segments)
    if columns!=swapped:raise ValueError("Feature schema mismatch")
    return x,xs,columns


def training_rows(game,save_skill_evaluation=None):
    clean=public_game(game)
    family,name=game.split("/")
    scenarios=ORIGINAL_SCENARIOS if family=="original" else SKILL_SCENARIOS
    parts=[];columns=None
    for scenario,(seed,params,stride) in scenarios.items():
        data=clean if seed is None else degrade.degrade(clean,seed=seed,params=params)
        x,xs,cols=feature_pair(data)
        if columns is not None and columns!=cols:raise ValueError("Augmentation schema changed")
        columns=cols
        parts.append(context.rows(x,xs,data["gt"],stride=stride))
        if family=="skillcorner" and save_skill_evaluation is not None:
            path=Path(save_skill_evaluation)/f"{name}__{scenario}.npz"
            path.parent.mkdir(parents=True,exist_ok=True)
            np.savez(path,X=x,Xs=xs,gt=data["gt"],seen=np.isfinite(data["ball"][:,0]),cols=np.array(cols))
    return np.concatenate([p[0] for p in parts]),np.concatenate([p[1] for p in parts]),columns


def taper_weight(seen,segments,scale=1.):
    weight=np.zeros(len(seen))
    for a,b in segments:
        since,until,_=features2._gap_info(seen[a:b])
        weight[a:b]=scale*(1-np.exp(-since))*(1-np.exp(-until))
    return weight


def blend_team(old,new,seen,segments,spec):
    weight=(taper_weight(seen,segments,spec["weight"]) if spec["gate"]=="taper_1s" else spec["weight"])
    return (1-weight)*old+weight*new
