"""Reference-blind, team-symmetric reliability mixtures for external research."""
import numpy as np


def weights(spec,base_dead,base_home,neural_dead,neural_home,gap,visibility):
    kind=spec['kind'];n=len(base_dead)
    if kind=='fixed':return np.full(n,spec['dead']),np.full(n,spec['home'])
    if kind=='gap':
        q=np.clip(np.nan_to_num(gap,nan=120.,posinf=120.,neginf=0.)/spec['scale_seconds'],0,1)
        weight=spec['low']+(spec['high']-spec['low'])*q
        return weight,weight.copy()
    if kind=='visibility':
        q=1-np.clip(np.nan_to_num(visibility,nan=.5,posinf=1.,neginf=0.),0,1)
        weight=spec['low']+(spec['high']-spec['low'])*q
        return weight,weight.copy()
    if kind=='confidence':
        def channel(base,neural):return np.clip(.5+.4*(np.abs(2*neural-1)-np.abs(2*base-1)),spec['low'],spec['high'])
        return channel(base_dead,neural_dead),channel(base_home,neural_home)
    raise ValueError('Unregistered reliability rule')


def mix(spec,base_dead,base_home,neural_dead,neural_home,gap,visibility):
    wd,wh=weights(spec,base_dead,base_home,neural_dead,neural_home,gap,visibility)
    return (1-wd)*base_dead+wd*neural_dead,(1-wh)*base_home+wh*neural_home
