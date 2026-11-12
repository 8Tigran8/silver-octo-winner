"""Обучение итоговых моделей v1 на ВСЕХ внешних матчах (оба яруса) и параметры декодера.

Никаких данных нашего матча: только external/conv/*.npz (IDSSE/DFL, Metrica sample), испорченные
mtah/degrade.py с фиксированным зерном. Выбор вариантов и параметров — tools/exp_v1.py, только по
внешним матчам (каждый проверяется моделями, обученными без него); итог записан в v1_params.json.

Схема:
1. Признаки (mtah/features2.py) на каждый внешний матч, прямые и зеркальные.
2. Первый ярус: 9 моделей «все, кроме одного» (дают честные вероятности для второго яруса)
   и одна модель на всех 9 (её применяем к нашему матчу).
3. Второй ярус (mtah/stack.py) учится на честных вероятностях всех 9 матчей.

Воспроизводимость: зерно порчи фиксировано, бустинг random_state=0 без ранней остановки, каждая
модель учится в отдельном процессе в ОДИН поток (порядок сложений не зависит от числа ядер).
Время: ~10–15 мин на 8 ядрах.
"""

import json
import os
from multiprocessing import get_context

import numpy as np

from .. import external, features2, stack
from ..config import CACHE_DIR
from . import boost

HERE = os.path.dirname(os.path.abspath(__file__))
PARAMS_FILE = os.path.join(HERE, "v1_params.json")
SEED = 0


def config():
    with open(PARAMS_FILE) as fh:
        return json.load(fh)


def decoder_params():
    return dict(boost.DECODER, **config().get("decoder", {}))


def _feat_path(name):
    return os.path.join(CACHE_DIR, "v1feat_{}.npz".format(name))


def _make_features(name):
    os.environ["OMP_NUM_THREADS"] = "1"
    d = external.load(name, seed=SEED)
    X, _ = features2.build(d, d["segments"])
    Xs, _ = features2.build(d, d["segments"], swap=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(_feat_path(name), X=X, Xs=Xs, gt=d["gt"])
    return name


def _fit_job(args):
    from threadpoolctl import threadpool_limits
    train, test = args
    with threadpool_limits(1):
        rows = []
        for n in train:
            z = np.load(_feat_path(n))
            rows.append(boost.training_rows(z["X"], z["Xs"], z["gt"]))
        models = boost.fit(rows)
        del rows
        if test is None:
            return None, models
        z = np.load(_feat_path(test))
        return test, boost.predict(models, z["X"], z["Xs"])


def train(names=None, log=print):
    """-> {'stage1': модели на всех матчах, 'stage2': второй ярус или None}."""
    names = names or external.names()
    if not names:
        raise IOError("нет внешних матчей в external/conv — сначала tools/ext_idsse.py и tools/ext_metrica.py")
    cfg = config()
    ctx = get_context("fork")
    with ctx.Pool(min(8, os.cpu_count() or 1)) as pool:
        pool.map(_make_features, names)
        log("признаки внешних матчей готовы ({})".format(len(names)))
        jobs = [(names, None)]
        if cfg.get("stack"):
            jobs += [([m for m in names if m != n], n) for n in names]
        res = pool.map(_fit_job, jobs)
    stage1 = res[0][1]
    log("первый ярус обучен на {} внешних матчах".format(len(names)))
    stage2 = None
    if cfg.get("stack"):
        from threadpoolctl import threadpool_limits
        oof = {n: p for n, p in res[1:]}
        rows = []
        for n in names:
            gt = np.load(_feat_path(n))["gt"]
            rows.append(stack.rows(oof[n][0], oof[n][1], gt))
        with threadpool_limits(1):
            stage2 = stack.fit(rows)
        log("второй ярус обучен на честных вероятностях {} матчей".format(len(names)))
    return {"stage1": stage1, "stage2": stage2}
