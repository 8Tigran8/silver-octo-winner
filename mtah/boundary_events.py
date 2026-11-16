"""Research-only geometry boundary head and time-varying duration decoder."""
import numpy as np
from .methods import boost


def temporal_features(x, names, segments, delta_names):
    """No labels, events, model outputs or cross-period context are accepted."""
    columns = {name: i for i, name in enumerate(names)}
    indices = [columns[name] for name in delta_names]
    pieces = [np.asarray(x, np.float32)]
    for step in [1, 5]:
        before = np.full((len(x), len(indices)), np.nan, np.float32)
        after = before.copy()
        for a, b in segments:
            t = np.arange(a, b)
            values = x[a:b, indices]
            before[a:b] = values - x[np.maximum(t-step, a)[:, None], indices]
            after[a:b] = x[np.minimum(t+step, b-1)[:, None], indices] - values
        pieces.extend([before, after])
    return np.concatenate(pieces, axis=1)


def targets(gt, segments):
    """Supervision-only path; unknowns and first-in-period pairs remain -1."""
    out = np.full(len(gt), -1, np.int8)
    for a, b in segments:
        previous, current = gt[a:b-1], gt[a+1:b]
        valid = (previous < 3) & (current < 3)
        change = previous != current
        value = np.where(~change, 0, np.where((previous < 2) & (current < 2), 1, 2))
        out[a+1:b] = np.where(valid, value, -1)
    return out


def probabilities(models, direct, swapped):
    if any(not np.array_equal(model.classes_, [0, 1, 2]) for model in models):
        raise ValueError('Boundary class schema mismatch')
    individual = [.5*(model.predict_proba(direct)+model.predict_proba(swapped)) for model in models]
    return np.mean(individual, axis=0), individual


def transition_costs(q, prior, params, strength):
    q = np.clip(q, 1e-7, 1-1e-7)
    prior = np.clip(prior, 1e-7, 1-1e-7)
    evidence = np.maximum(0, np.log(q[:, 1:]/q[:, :1]) - np.log(prior[1:]/prior[0]))
    n = len(q)
    costs = np.full((n, 4, 4), np.inf)
    for a, b, key, kind in [(0,1,'cost_ha',0),(1,0,'cost_ha',0),
                            (0,2,'cost_in',1),(1,3,'cost_in',1),
                            (2,0,'cost_same',1),(3,1,'cost_same',1),
                            (2,1,'cost_other',1),(3,0,'cost_other',1)]:
        costs[:, a, b] = np.clip(params[key]-strength*evidence[:, kind], 1, params[key])
    for i in range(4):costs[:, i, i] = 0
    return costs


def duration_path(emissions, costs, lengths):
    """Same duration chain/tie rules as boost, with a cost matrix at each step."""
    total = len(emissions)
    lengths = np.maximum(1, np.asarray(lengths, int))
    first = np.cumsum(np.r_[0, lengths[:-1]])
    last = first + lengths - 1
    size = int(lengths.sum())
    base = np.repeat(np.arange(4), lengths)
    previous = np.arange(size)-1
    is_first = np.zeros(size, bool);is_first[first] = True
    negative = -1e18
    value = np.full(size, negative);value[first] = emissions[0]
    entry = np.zeros((total,4), np.int8);stayed = np.zeros((total,4), bool)
    costs = np.where(np.isfinite(costs), costs, 1e18)
    single = lengths == 1
    for t in range(1,total):
        last_value = value[last]
        candidate = np.where(is_first, negative, value[np.maximum(previous,0)])
        stay_better = last_value >= candidate[last]
        candidate[last] = np.where(stay_better,last_value,candidate[last])
        transitions = last_value[:,None]-costs[t]
        np.fill_diagonal(transitions, negative)
        best = transitions.max(axis=0);entry[t] = transitions.argmax(axis=0)
        take = best > candidate[first]
        candidate[first] = np.where(take,best,candidate[first])
        stayed[t] = np.where(single,~take,stay_better)
        value = candidate+emissions[t][base]
    path = np.zeros(total,np.int8)
    state = int(np.argmax(value));k = int(base[state]);j = state-first[k]
    for t in range(total-1,-1,-1):
        path[t] = k
        if t==0:break
        if j==lengths[k]-1 and stayed[t,k]:continue
        if j>0:j-=1;continue
        k=int(entry[t,k]);j=lengths[k]-1
    return path


def label(pd, ph, boundary, prior, segments, n, params, strength):
    if strength==0:
        return boost.label(pd,ph,segments,n,params)
    out=np.full(n,3,np.int8)
    stride=int(params['stride'])
    for a,b in segments:
        dead=np.clip(pd[a:b],1e-4,1-1e-4);home=np.clip(ph[a:b],1e-4,1-1e-4)
        emissions=np.stack([np.log1p(-dead)+np.log(home),np.log1p(-dead)+np.log1p(-home),np.log(dead),np.log(dead)],axis=1)*params['emit_scale']
        blocks=(b-a+stride-1)//stride;pad=blocks*stride-(b-a)
        if pad:emissions=np.concatenate([emissions,np.repeat(emissions[-1:],pad,axis=0)])
        emissions=emissions.reshape(blocks,stride,4).sum(axis=1)
        q=boundary[a:b:stride]
        costs=transition_costs(q,prior,params,strength)
        live=max(1,round(params['min_team_s']*25/stride));dead_steps=max(1,round(params['min_dead_s']*25/stride))
        path=duration_path(emissions,costs,[live,live,dead_steps,dead_steps])
        path=np.repeat(path,stride)[:b-a]
        out[a:b]=np.where(path>=2,2,path)
    return out
