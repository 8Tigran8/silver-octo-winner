"""Внешние матчи: загрузка, порча под наш матч, оценка любого метода теми же метриками.

Это единственные данные, на которых разрешено подбирать параметры (разъяснения организаторов 04.11).

    from mtah import external
    games = external.load_all()                     # {имя: испорченный матч + ответы}
    external.score(games["idsse_J03WMX"], labels)    # spellF1 / macroF1
"""

import glob
import hashlib
import json
import os

import numpy as np

from . import degrade, features, metrics
from .config import ROOT, CACHE_DIR

CONV_DIR = os.path.join(ROOT, "external", "conv")
# Проверочные матчи: на них ничего не подбираем, только смотрим итог.
HOLDOUT = ("idsse_J03WR9", "metrica_1", "metrica_2")


def names():
    return sorted(os.path.splitext(os.path.basename(p))[0]
                  for p in glob.glob(os.path.join(CONV_DIR, "*.npz")))


def load(name, seed=0, use_cache=True, params=None):
    """Испорченный матч с признаками. Кэш в out/cache, порча детерминирована по зерну и параметрам."""
    src = os.path.join(CONV_DIR, name + ".npz")
    p = dict(degrade.DEFAULTS, **(params or {}))
    tag = hashlib.md5(json.dumps(p, sort_keys=True).encode()).hexdigest()[:6]
    cache = os.path.join(CACHE_DIR, "ext_{}_{}_s{}.npz".format(name, tag, seed))
    if use_cache and os.path.isfile(cache) and os.path.getmtime(cache) > os.path.getmtime(src):
        z = np.load(cache, allow_pickle=False)
        d = {k: z[k] for k in z.files}
        d["pitch"] = tuple(d["pitch"])
    else:
        z = np.load(src)
        clean = {k: z[k] for k in z.files}
        d = degrade.degrade(clean, seed=seed, params=params)
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez(cache, **{k: np.asarray(v) for k, v in d.items()})
    d["name"] = name
    d["segments"] = metrics.valid_runs(d["gt"])
    return d


def with_features(d):
    d["features"] = features.build(d)
    return d


def load_all(seed=0, only=None):
    out = {}
    for n in names():
        if only and n not in only:
            continue
        out[n] = with_features(load(n, seed))
    return out


def score(d, labels):
    pred = metrics.codes_from_labels(labels)
    r = metrics.evaluate(d["gt"], pred)
    return {"spell_f1": r["spell_f1"], "macro_f1": r["macro_f1"],
            "n_pred": r["n_pred_spells"], "n_gt": r["n_gt_spells"]}
