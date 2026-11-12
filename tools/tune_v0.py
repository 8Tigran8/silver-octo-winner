"""Подбор параметров v0 только на внешних матчах (правило организаторов: не на нашем матче).

Обучающие: все внешние, кроме external.HOLDOUT. Цель: средний spellF1 по обучающим.
Случайный поиск вокруг текущих значений, затем покоординатное уточнение. Итог печатается и
пишется в out/tune_v0.json (лучшие параметры + счёт на каждом матче, базовое решение рядом).

    .venv/bin/python tools/tune_v0.py [--n 400] [--seed 0]
    .venv/bin/python tools/tune_v0.py --n 0      # только счёт текущих параметров на внешних матчах
"""

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mtah import external                     # noqa: E402
from mtah.config import OUT_DIR               # noqa: E402
from mtah.methods import baseline, viterbi    # noqa: E402

# (нижняя, верхняя граница, логарифмическая шкала)
SPACE = {
    "r_control": (0.2, 6.0, True),
    "team_gain": (0.05, 5.0, True),
    "margin_scale": (0.5, 8.0, True),
    "w_out": (0.5, 10.0, True),
    "w_missing": (0.05, 3.0, True),
    "missing_s": (0.5, 8.0, True),
    "w_slow": (0.05, 4.0, True),
    "slow_speed": (0.5, 3.5, False),
    "alive_bias": (-0.5, 1.5, False),
    "cost_ha": (0.2, 80.0, True),
    "cost_d": (0.1, 80.0, True),
}
GAMES = {}


def spell(d, p):
    sc = viterbi.emissions(d["features"], p)
    lab = viterbi.label_segments(sc, p, d["segments"], len(d["gt"]))
    return external.score(d, lab)


def objective(p):
    full = dict(viterbi.DEFAULTS, **p)
    res = {n: spell(d, full) for n, d in GAMES.items() if n not in external.HOLDOUT}
    return float(np.mean([r["spell_f1"] for r in res.values()])), p


def sample(rng, center=None, spread=1.0):
    p = {}
    for k, (lo, hi, log) in SPACE.items():
        if center is None:
            u = rng.random()
        else:
            c = center[k]
            u0 = (np.log(c) - np.log(lo)) / (np.log(hi) - np.log(lo)) if log else (c - lo) / (hi - lo)
            u = float(np.clip(u0 + rng.normal(0, 0.15 * spread), 0, 1))
        p[k] = float(np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo)))) if log else lo + u * (hi - lo)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    GAMES.update(external.load_all(seed=0))
    train = [n for n in GAMES if n not in external.HOLDOUT]
    print("обучающие: {}; проверочные: {}".format(train, list(external.HOLDOUT)), flush=True)
    rng = np.random.default_rng(args.seed)
    start = {k: viterbi.DEFAULTS[k] for k in SPACE}
    with Pool(8) as pool:
        best = pool.map(objective, [start])[0]
        print("исходные значения: {:.4f}".format(best[0]), flush=True)
        # 1) широкий случайный поиск
        for s, p in pool.imap_unordered(objective, [sample(rng) for _ in range(args.n)], chunksize=4):
            if s > best[0]:
                best = (s, p)
        print("после широкого поиска: {:.4f} ({:.0f} с)".format(best[0], time.time() - t0), flush=True)
        # 2) уточнение вокруг лучшего, шаг сужается
        for spread in (1.0, 0.5, 0.25):
            cands = [sample(rng, best[1], spread) for _ in range(args.n // 4)]
            for s, p in pool.imap_unordered(objective, cands, chunksize=4):
                if s > best[0]:
                    best = (s, p)
            print("уточнение {}: {:.4f}".format(spread, best[0]), flush=True)

    params = dict(viterbi.DEFAULTS, **best[1])
    report = {"params": params, "train_mean_spell": best[0], "games": {}}
    for n, d in GAMES.items():
        v = spell(d, params)
        b = external.score(d, baseline.run(d, window=50))
        report["games"][n] = {"role": "проверка" if n in external.HOLDOUT else "подбор",
                              "v0": v, "baseline50": b}
    for seed in (1, 2):   # другая порча тех же проверочных матчей — устойчивость
        for n in external.HOLDOUT:
            d = external.with_features(external.load(n, seed=seed))
            report["games"]["{}_s{}".format(n, seed)] = {
                "role": "проверка, другая порча", "v0": spell(d, params),
                "baseline50": external.score(d, baseline.run(d, window=50))}
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "tune_v0.json"), "w") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)
    print("\n{:<26} {:<22} {:>9} {:>9} {:>9} {:>9}".format(
        "матч", "роль", "v0 spell", "база sp", "v0 macro", "база mac"))
    for n, r in report["games"].items():
        print("{:<26} {:<22} {:>9.3f} {:>9.3f} {:>9.3f} {:>9.3f}".format(
            n, r["role"], r["v0"]["spell_f1"], r["baseline50"]["spell_f1"],
            r["v0"]["macro_f1"], r["baseline50"]["macro_f1"]))
    print("\nпараметры:", json.dumps({k: round(v, 3) for k, v in params.items()}, ensure_ascii=False))
    print("время: {:.0f} с".format(time.time() - t0))


if __name__ == "__main__":
    main()
