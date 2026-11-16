"""Второй ярус (стекинг): вероятности первого яруса + их окна во времени -> уточнённые вероятности.

Первый ярус смотрит на кадр и короткие окна признаков; второй видит, как вероятности «мёртв» и
«хозяева» ведут себя секунды до и после — это ближе к тому, как считается владение целиком.
Учится на вероятностях «все, кроме одного» внешних матчей (честных, не на своих же данных).
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .config import H, A, D, X

WINS = (5, 25, 75, 250)
SHIFTS = (25, 75, 200)
MODEL = {"max_iter": 150, "learning_rate": 0.08, "max_leaf_nodes": 15, "min_samples_leaf": 400,
         "l2_regularization": 1.0, "max_bins": 128}
STRIDE = 4


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _smooth(x, w):
    k = np.ones(w) / w
    pad = w // 2
    xp = np.concatenate([np.repeat(x[:1], pad), x, np.repeat(x[-1:], w - 1 - pad)])
    return np.convolve(xp, k, mode="valid")


def _shift(x, s):
    out = np.empty_like(x)
    if s > 0:
        out[s:] = x[:-s]
        out[:s] = x[0]
    else:
        out[:s] = x[-s:]
        out[s:] = x[-1]
    return out


def features(pD, pH, swap=False):
    if swap:
        pH = 1.0 - pH
    cols = []
    for p in (_logit(pD), _logit(pH)):
        cols.append(p)
        for w in WINS:
            cols.append(_smooth(p, w))
        for s in SHIFTS:
            sm = _smooth(p, 25)
            cols.append(_shift(sm, s))
            cols.append(_shift(sm, -s))
    return np.stack(cols, axis=1).astype(np.float32)


def rows(pD, pH, gt, stride=STRIDE):
    idx = np.arange(0, len(gt), stride)
    idx = idx[gt[idx] != X]
    F, Fs = features(pD, pH), features(pD, pH, swap=True)
    g = gt[idx]
    Xd = np.concatenate([F[idx], Fs[idx]])
    yd = np.concatenate([g == D, g == D]).astype(np.int8)
    live = idx[(g == H) | (g == A)]
    gl = gt[live]
    Xt = np.concatenate([F[live], Fs[live]])
    yt = np.concatenate([gl == H, gl == A]).astype(np.int8)
    return Xd, yd, Xt, yt


def fit(rows_list):
    Xd = np.concatenate([r[0] for r in rows_list])
    yd = np.concatenate([r[1] for r in rows_list])
    Xt = np.concatenate([r[2] for r in rows_list])
    yt = np.concatenate([r[3] for r in rows_list])
    clf = lambda: HistGradientBoostingClassifier(random_state=0, early_stopping=False, **MODEL)  # noqa: E731
    return clf().fit(Xd, yd), clf().fit(Xt, yt)


def predict(models, pD, pH, segments=None):
    if segments is not None:
        qD = np.full(len(pD), .5); qH = qD.copy()
        for a, b in segments:
            qD[a:b], qH[a:b] = predict(models, pD[a:b], pH[a:b])
        return qD, qH
    md, mt = models
    F, Fs = features(pD, pH), features(pD, pH, swap=True)
    qD = 0.5 * (md.predict_proba(F)[:, 1] + md.predict_proba(Fs)[:, 1])
    qH = 0.5 * (mt.predict_proba(F)[:, 1] + 1.0 - mt.predict_proba(Fs)[:, 1])
    return qD, qH
