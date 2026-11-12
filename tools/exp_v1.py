"""Эксперименты v1 только на внешних матчах: признаки -> модели по схеме «все, кроме одного» -> декодер.

Правило: наш матч здесь не участвует. Каждый внешний матч получает вероятности от моделей, обученных
на остальных (честная проверка). Параметры декодера подбираются по 6 матчам DFL из «подбора»;
external.HOLDOUT (J03WR9, Metrica 1–2) только смотрим.

    .venv/bin/python tools/exp_v1.py oof [--variant base] [--no-metrica-train]
    .venv/bin/python tools/exp_v1.py tune [--variant base] [--n 120]
    .venv/bin/python tools/exp_v1.py score [--variant base] [--params '{...}']
"""

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np   # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mtah import external, features2, metrics      # noqa: E402
from mtah.config import CACHE_DIR, OUT_DIR, X       # noqa: E402
from mtah.methods import boost, viterbi, baseline   # noqa: E402

NAMES = external.names()


# «Стресс»: другая порча (зерно 1, края дыр размыты вдвое сильнее) — только для проверки устойчивости.
STRESS = {"dead_jitter_s": 4.0, "p_dead_hide": 0.85}


def feat_path(name, seed=0):
    return os.path.join(CACHE_DIR, "f2_{}_s{}.npz".format(name, seed))


def make_features(args):
    name, seed = args
    path = feat_path(name, seed)
    if os.path.isfile(path):
        return path
    d = external.load(name, seed=seed % 100, params=STRESS if seed >= 100 else None)
    Xf, cols = features2.build(d, d["segments"])
    Xs, _ = features2.build(d, d["segments"], swap=True)
    kick = boost.kick_frames(d, boost.DECODER)
    np.savez(path, X=Xf, Xs=Xs, gt=d["gt"], kick=kick, cols=np.array(cols))
    return path


def load_feat(name, seed=0):
    z = np.load(feat_path(name, seed))
    return {k: z[k] for k in z.files}


def oof_one(args):
    test, train_names, variant, seeds = args
    rows = []
    for n in train_names:
        for s in seeds:
            f = load_feat(n, s)
            rows.append(boost.training_rows(f["X"], f["Xs"], f["gt"]))
    models = boost.fit(rows)
    res = {}
    for s, tag in ((0, ""), (101, "_stress")):
        f = load_feat(test, s)
        pD, pH = boost.predict(models, f["X"], f["Xs"])
        res["pD" + tag], res["pH" + tag] = pD, pH
    out = os.path.join(CACHE_DIR, "oof_{}_{}.npz".format(variant, test))
    np.savez(out, **res)
    imp = None
    return test, out, imp


def cmd_oof(args):
    t0 = time.time()
    seeds = [int(s) for s in args.seeds.split(",")]
    jobs = [(n, s) for n in NAMES for s in seeds + [101]]
    with Pool(8) as pool:
        pool.map(make_features, jobs)
    print("признаки готовы за {:.0f} с".format(time.time() - t0), flush=True)
    tasks = []
    for test in NAMES:
        train = [n for n in NAMES if n != test]
        if args.no_metrica_train:
            train = [n for n in train if not n.startswith("metrica")]
        tasks.append((test, train, args.variant, seeds))
    with Pool(args.jobs) as pool:
        for test, out, _ in pool.imap_unordered(oof_one, tasks):
            print("  {} готов ({:.0f} с)".format(test, time.time() - t0), flush=True)


DATA = {}


def _init(variant, stress=False):
    tag, s = ("_stress", 101) if stress else ("", 0)
    for n in NAMES:
        f = load_feat(n, s)
        z = np.load(os.path.join(CACHE_DIR, "oof_{}_{}.npz".format(variant, n)))
        DATA[n] = {"gt": f["gt"], "kick": f["kick"], "pD": z["pD" + tag], "pH": z["pH" + tag],
                   "segments": metrics.valid_runs(f["gt"])}


def eval_one(args):
    name, p = args
    g = DATA[name]
    lab = boost.label(g["pD"], g["pH"], g["segments"], len(g["gt"]), p, g["kick"])
    r = metrics.evaluate(g["gt"], lab)
    return name, r["spell_f1"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"]


def evaluate(pool, p, names):
    res = pool.map(eval_one, [(n, p) for n in names])
    return {n: {"spell": s, "macro": m, "n_pred": a, "n_gt": b} for n, s, m, a, b in res}


TUNE = [n for n in NAMES if n not in external.HOLDOUT]

SPACE = {   # (низ, верх, лог)
    "emit_scale": (0.05, 2.0, True),
    "cost_ha": (0.5, 60.0, True),
    "cost_in": (0.5, 60.0, True),
    "cost_same": (0.5, 60.0, True),
    "cost_other": (0.5, 60.0, True),
    "min_team_s": (0.2, 4.0, True),
    "min_dead_s": (0.2, 4.0, True),
}


def sample(rng, center=None, spread=1.0):
    p = {}
    for k, (lo, hi, log) in SPACE.items():
        if center is None:
            u = rng.random()
        else:
            c = center[k]
            u0 = (np.log(c) - np.log(lo)) / (np.log(hi) - np.log(lo)) if log else (c - lo) / (hi - lo)
            u = float(np.clip(u0 + rng.normal(0, 0.12 * spread), 0, 1))
        p[k] = float(np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo)))) if log else lo + u * (hi - lo)
    return p


def cmd_tune(args):
    t0 = time.time()
    rng = np.random.default_rng(0)
    base = dict(boost.DECODER, **json.loads(args.params))
    with Pool(8, initializer=_init, initargs=(args.variant,)) as pool:
        def obj(p):
            full = dict(base, **p)
            r = evaluate(pool, full, TUNE)
            return float(np.mean([v["spell"] for v in r.values()]))
        start = {k: base[k] for k in SPACE}
        best = (obj(start), start)
        print("начало: {:.4f}".format(best[0]), flush=True)
        for i in range(args.n):
            p = sample(rng) if i < args.n // 3 else sample(rng, best[1], 1.0 if i < 2 * args.n // 3 else 0.4)
            s = obj(p)
            if s > best[0]:
                best = (s, p)
                print("  {:>4} {:.4f} {} ({:.0f} с)".format(i, s, {k: round(v, 3) for k, v in p.items()},
                                                          time.time() - t0), flush=True)
        full = dict(base, **best[1])
        report(pool, full, args.variant)


def report(pool, p, variant):
    r = evaluate(pool, p, NAMES)
    print("\n{:<16} {:<9} {:>7} {:>7} {:>6} {:>6}".format("матч", "роль", "spell", "macro", "наших", "верных"))
    for n in NAMES:
        v = r[n]
        print("{:<16} {:<9} {:>7.3f} {:>7.3f} {:>6} {:>6}".format(
            n, "проверка" if n in external.HOLDOUT else "подбор", v["spell"], v["macro"], v["n_pred"], v["n_gt"]))
    hold = [r[n]["spell"] for n in external.HOLDOUT]
    tune = [r[n]["spell"] for n in TUNE]
    allm = [r[n]["spell"] for n in NAMES]
    print("среднее spell: подбор {:.4f}, проверка {:.4f}, все 9 {:.4f} (разброс {:.3f}–{:.3f}); macro все {:.4f}".format(
        np.mean(tune), np.mean(hold), np.mean(allm), min(allm), max(allm),
        np.mean([r[n]["macro"] for n in NAMES])))
    print("параметры:", json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in p.items()}))
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "exp_v1_{}.json".format(variant)), "w") as fh:
        json.dump({"params": p, "games": r}, fh, ensure_ascii=False, indent=1)


def cmd_score(args):
    p = dict(boost.DECODER, **json.loads(args.params))
    with Pool(8, initializer=_init, initargs=(args.variant, args.stress)) as pool:
        report(pool, p, args.variant + ("_stress" if args.stress else ""))


def v0_one(name):
    d = external.with_features(external.load(name, seed=0))
    p = dict(viterbi.DEFAULTS)
    lab = viterbi.label_segments(viterbi.emissions(d["features"], p), p, d["segments"], len(d["gt"]))
    r = external.score(d, lab)
    b = external.score(d, baseline.run(d, window=50))
    return name, r, b


def cmd_v0(args):
    with Pool(8) as pool:
        res = pool.map(v0_one, NAMES)
    print("{:<16} {:>8} {:>8} {:>8} {:>8}".format("матч", "v0 sp", "v0 mac", "база sp", "база mac"))
    for n, r, b in res:
        print("{:<16} {:>8.3f} {:>8.3f} {:>8.3f} {:>8.3f}".format(n, r["spell_f1"], r["macro_f1"],
                                                                b["spell_f1"], b["macro_f1"]))
    for title, sel in (("подбор", TUNE), ("проверка", list(external.HOLDOUT)), ("все 9", NAMES)):
        rr = [r for n, r, b in res if n in sel]
        bb = [b for n, r, b in res if n in sel]
        print("{}: v0 spell {:.4f} macro {:.4f}; база spell {:.4f}".format(
            title, np.mean([r["spell_f1"] for r in rr]), np.mean([r["macro_f1"] for r in rr]),
            np.mean([b["spell_f1"] for b in bb])))


def stack_one(args):
    from mtah import stack
    test, variant = args
    oof = {n: np.load(os.path.join(CACHE_DIR, "oof_{}_{}.npz".format(variant, n))) for n in NAMES}
    gts = {n: load_feat(n, 0)["gt"] for n in NAMES}
    rows = [stack.rows(oof[n]["pD"], oof[n]["pH"], gts[n]) for n in NAMES if n != test]
    models = stack.fit(rows)
    res = {}
    for tag in ("", "_stress"):
        res["pD" + tag], res["pH" + tag] = stack.predict(models, oof[test]["pD" + tag], oof[test]["pH" + tag])
    np.savez(os.path.join(CACHE_DIR, "oof_{}s2_{}.npz".format(variant, test)), **res)
    return test


def cmd_stack(args):
    t0 = time.time()
    with Pool(args.jobs) as pool:
        for t in pool.imap_unordered(stack_one, [(n, args.variant) for n in NAMES]):
            print("  {} ({:.0f} с)".format(t, time.time() - t0), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["oof", "tune", "score", "v0", "stack"])
    ap.add_argument("--variant", default="base")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--no-metrica-train", action="store_true")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--params", default="{}")
    ap.add_argument("--stress", action="store_true")
    args = ap.parse_args()
    {"stack": cmd_stack, "oof": cmd_oof, "tune": cmd_tune, "score": cmd_score, "v0": cmd_v0}[args.cmd](args)


if __name__ == "__main__":
    main()
