"""Признаки v1 на кадр для модели (бустинг). Только трекинг: позиции, скорости, дыры у мяча.

Одинаково считаются для нашего матча и для испорченных внешних (mtah/degrade.py), поэтому модель,
обученная на внешних, применима к нашему. Ответы нашего матча здесь не участвуют.

Симметрия команд: build(d, swap=True) считает те же признаки, как будто хозяева и гости поменялись
местами (метки H <-> A). Это и аугментация при обучении, и способ убрать перекос «хозяева чаще с мячом»
при применении (среднее прямого и зеркального ответа).

Направление атаки хозяев на каждом тайме — по среднему положению команд за тайм (автоматически,
из трекинга): координаты мяча и команд даём «в сторону ворот соперника хозяев».
"""

import warnings

import numpy as np

from .config import PITCH, FPS

WINDOWS = (13, 51, 151)          # окна сглаживания: 0.5 с, 2 с, 6 с (по центру)


def _runs(mask):
    m = np.concatenate(([0], mask.astype(np.int8), [0]))
    e = np.diff(m)
    return np.flatnonzero(e == 1), np.flatnonzero(e == -1)


def _gap_info(seen):
    """Для кадров без мяча: сколько секунд с последнего мяча, до следующего, длина дыры."""
    n = len(seen)
    since = np.zeros(n)
    until = np.zeros(n)
    total = np.zeros(n)
    for a, b in zip(*_runs(~seen)):
        k = np.arange(b - a)
        since[a:b] = k + 1
        until[a:b] = (b - a) - k
        total[a:b] = b - a
    return since / FPS, until / FPS, total / FPS


def _interp(x, seen):
    """Линейная подстановка мяча в дырах (по краям — ближайшее известное)."""
    idx = np.flatnonzero(seen)
    if not len(idx):
        return np.zeros_like(x)
    return np.interp(np.arange(len(x)), idx, x[idx])


def _smooth(x, w):
    """Скользящее среднее по центру с учётом NaN (NaN -> не участвует)."""
    ok = np.isfinite(x)
    v = np.where(ok, x, 0.0)
    k = np.ones(min(w, len(x)))
    num = np.convolve(v, k, mode="same")
    den = np.convolve(ok.astype(float), k, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.maximum(den, 1), np.nan)


def _attack_dir(cx_home, cx_away, segments, n, L):
    """+1, если хозяева на этом куске атакуют в сторону x = L (их середина в среднем левее гостей)."""
    out = np.ones(n)
    for a, b in segments:
        dh, da = np.nanmean(cx_home[a:b]), np.nanmean(cx_away[a:b])
        out[a:b] = 1.0 if not (np.isfinite(dh) and np.isfinite(da)) or dh <= da else -1.0
    return out


def build(d, segments, swap=False):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        n = len(d["ball"])
        result = None
        previous = 0
        for a, b in segments:
            if not (previous <= a < b <= n):
                raise ValueError("Periods must be ordered, disjoint and within input")
            previous = b
            local = dict(d, ball=d["ball"][a:b], players=d["players"][a:b])
            part, names = _build(local, [(0, b-a)], swap)
            if result is None:
                result = np.full((n, len(names)), np.nan, np.float32)
            result[a:b] = part
        if result is None:
            raise ValueError("At least one nonempty period required")
        return result, names


def _team_stats(pl, sp, mask, ball_i, L):
    """Для одной команды: ближайший и второй к мячу, сколько рядом, середина, ширина, скорость."""
    xs = np.where(mask, pl[:, :, 0], np.nan)
    ys = np.where(mask, pl[:, :, 1], np.nan)
    dist = np.sqrt((xs - ball_i[:, None, 0]) ** 2 + (ys - ball_i[:, None, 1]) ** 2)
    ds = np.sort(np.where(np.isfinite(dist), dist, 1e3), axis=1)
    d1, d2 = ds[:, 0], ds[:, 1]
    d1[d1 >= 1e3] = np.nan
    d2[d2 >= 1e3] = np.nan
    order = np.argmin(np.where(np.isfinite(dist), dist, 1e3), axis=1)
    # Duplicate/equidistant detections must not make slot order a predictor.
    closest = np.min(np.where(np.isfinite(dist), dist, np.inf), axis=1)
    tied = mask & np.isfinite(dist) & (dist == closest[:, None])
    near_speed = np.nanmean(np.where(tied, sp, np.nan), axis=1)
    out = {
        "d1": d1, "d2": d2,
        "n3": (dist < 3).sum(axis=1).astype(float),
        "n7": (dist < 7).sum(axis=1).astype(float),
        "n15": (dist < 15).sum(axis=1).astype(float),
        "cx": np.nanmean(xs, axis=1), "cy": np.nanmean(ys, axis=1),
        "sx": np.nanstd(xs, axis=1), "sy": np.nanstd(ys, axis=1),
        "xmin": np.nanmin(xs, axis=1), "xmax": np.nanmax(xs, axis=1),
        "nvis": mask.sum(axis=1).astype(float),
        "speed": np.nanmean(np.where(mask, sp, np.nan), axis=1),
        "near_speed": near_speed,
    }
    return out


def _build(d, segments, swap):
    L, W = d.get("pitch", PITCH)
    n = len(d["ball"])
    team = d["slot_team"].astype(int)
    if swap:
        team = 1 - team
    ball = d["ball"][:, :2] * np.array([L, W])
    pl = d["players"][:, :, :2] * np.array([L, W])
    sp = np.where(np.isfinite(d["players"][:, :, 2]), d["players"][:, :, 2], np.nan)
    vis = np.isfinite(pl).all(axis=2)
    seen = np.isfinite(ball).all(axis=1)

    bi = np.stack([_interp(ball[:, 0], seen), _interp(ball[:, 1], seen)], axis=1)
    h = _team_stats(pl, sp, vis & (team[None, :] == 0), bi, L)
    a = _team_stats(pl, sp, vis & (team[None, :] == 1), bi, L)
    dirn = _attack_dir(h["cx"], a["cx"], segments, n, L)

    def fx(x):            # x -> «в сторону ворот гостей», 0 = центр поля
        return (x - L / 2) * dirn

    since, until, total = _gap_info(seen)
    v = np.zeros((n, 2))
    v[1:] = (bi[1:] - bi[:-1]) * FPS
    vs = np.stack([_smooth(v[:, 0], 5), _smooth(v[:, 1], 5)], axis=1)
    bspeed = np.sqrt((vs ** 2).sum(axis=1))
    bspeed_raw = np.sqrt((v ** 2).sum(axis=1))
    out_line = seen & ((ball[:, 0] < -0.5) | (ball[:, 0] > L + 0.5) | (ball[:, 1] < -0.5) | (ball[:, 1] > W + 0.5))
    edge_x = np.minimum(bi[:, 0], L - bi[:, 0])
    edge_y = np.minimum(bi[:, 1], W - bi[:, 1])
    allspeed = np.nanmean(np.where(vis, sp, np.nan), axis=1)
    nvis = vis.sum(axis=1).astype(float)

    f = {
        "seen": seen.astype(float),
        "since": np.minimum(since, 60), "until": np.minimum(until, 60), "gap": np.minimum(total, 120),
        "bx": fx(bi[:, 0]), "by": np.abs(bi[:, 1] - W / 2),
        "edge_x": edge_x, "edge_y": edge_y, "out": out_line.astype(float),
        "bvx": vs[:, 0] * dirn, "bspeed": bspeed, "bspeed_raw": np.minimum(bspeed_raw, 60),
        "allspeed": allspeed, "nvis": nvis,
        "h_d1": h["d1"], "h_d2": h["d2"], "a_d1": a["d1"], "a_d2": a["d2"],
        "margin": a["d1"] - h["d1"],
        "dmin": np.fmin(h["d1"], a["d1"]),
        "h_n3": h["n3"], "a_n3": a["n3"], "h_n7": h["n7"], "a_n7": a["n7"],
        "h_n15": h["n15"], "a_n15": a["n15"],
        "h_cx": fx(h["cx"]), "a_cx": fx(a["cx"]), "cx_diff": fx(h["cx"]) - fx(a["cx"]),
        "h_sx": h["sx"], "a_sx": a["sx"], "h_sy": h["sy"], "a_sy": a["sy"],
        "h_last": np.where(dirn > 0, h["xmin"], L - h["xmax"]),     # самый глубокий игрок хозяев от своей линии
        "a_last": np.where(dirn > 0, L - a["xmax"], a["xmin"]),
        "h_nvis": h["nvis"], "a_nvis": a["nvis"],
        "h_speed": h["speed"], "a_speed": a["speed"],
        "h_near_speed": h["near_speed"], "a_near_speed": a["near_speed"],
        "ball_minus_hcx": fx(bi[:, 0]) - fx(h["cx"]), "ball_minus_acx": fx(bi[:, 0]) - fx(a["cx"]),
    }
    # движение команд вперёд за 1 с (атакующая команда обычно идёт вперёд)
    for k in ("h_cx", "a_cx"):
        x = f[k]
        dx = np.full(n, np.nan)
        dx[FPS:] = x[FPS:] - x[:-FPS]
        f[k + "_v"] = dx
    # сглаженные окна по ключевым признакам
    closer = np.sign(np.nan_to_num(f["margin"]))
    closer[~seen] = 0.0
    base = {"margin": f["margin"], "dmin": f["dmin"], "seen": f["seen"], "allspeed": allspeed,
            "cx_diff": f["cx_diff"], "bx": f["bx"], "closer": closer, "bspeed": bspeed,
            "h_n7": f["h_n7"], "a_n7": f["a_n7"], "nvis": nvis, "out": f["out"]}
    for k, x in base.items():
        for w in WINDOWS:
            f["{}_w{}".format(k, w)] = _smooth(x, w)
    # прошлое и будущее по отдельности (кто был ближе 2 с назад и будет через 2 с)
    for k in ("closer", "seen", "allspeed"):
        s = _smooth(base[k], 51)
        past = np.full(n, np.nan)
        fut = np.full(n, np.nan)
        past[50:] = s[:-50]
        fut[:-50] = s[50:]
        f[k + "_past2"] = past
        f[k + "_fut2"] = fut
    names = sorted(f)
    X = np.stack([np.asarray(f[k], dtype=np.float32) for k in names], axis=1)
    return X, names
