"""«Порча» чистого внешнего трекинга под наш матч (автоматический трекинг Metrica без правки).

Что известно о нашем матче из DATA.md организаторов (и только это):
- мяч виден на 59.3% оцениваемых кадров, пропадает чаще в толкучке и в воздухе;
- 46 безымянных слотов на 22 игроков, слоты появляются, пропадают и меняют личность;
- в среднем 20.5 объектов в кадре во время игры.

Всё остальное (длина обрывков, длина дыр у мяча) — наши догадки, они в DEFAULTS.
Порча детерминирована: одно зерно -> один и тот же испорченный матч.
"""

import numpy as np

FPS = 25
DEFAULTS = {
    "ball_seen_target": 0.593,   # DATA.md: доля оцениваемых кадров с мячом
    "ball_gap_mean_s": 1.6,      # Г: средняя длина дыры у мяча
    "crowd_r": 3.0,              # м: «толкучка» — игроки в этом радиусе от мяча
    "crowd_k": 3,                # сколько игроков рядом считается толкучкой
    "crowd_mult": 3.0,           # во столько раз чаще теряется мяч в толкучке
    "air_z": 1.8,                # м: мяч выше — «в воздухе»
    "air_mult": 4.0,
    "slots_per_team": 23,        # 46 слотов на матч
    "frag_mean_s": 15.0,         # Г: средняя длина куска трека до потери личности
    "frag_gap_mean_s": 1.0,      # Г: пауза, пока трекер снова подхватит игрока
    "pos_noise_m": 0.15,
    "speed_noise": 0.3,          # м/с
    "skip_ball_if_seen_below": 0.70,  # источник уже дырявый (Metrica sample) — мяч не трогаем
    # v2 (10.11, сверка tools/sverka_porchi.py, tools/gapstat.py): у нашего трекинга мяч пропадает
    # в основном, когда игроки стоят (пауза), дыры длинные (средняя 17 с, медиана 11 с, 0.94 на 1000 кадров),
    # как у Metrica sample. Поэтому мяч прячем прежде всего в остановках, а в игре — реже.
    "mode": "v2",
    "p_dead_hide": 0.95,         # доля остановок, где мяч пропадает
    "dead_lag_s": 0.5,           # мяч ещё виден столько после начала остановки (средняя)
    "dead_lead_s": 1.0,          # и появляется столько до её конца (средняя)
    "dead_jitter_s": 2.0,        # края дыр не совпадают с краями остановки точно: у нас дыры чаще
                                 # начинаются посреди поля (34%), чем у порчи без размытия (61%)
    "live_gap_mean_s": 6.0,      # средняя дыра в игре
    "frag_median_s": 35.0,       # кусок трека: медиана и средняя как у нашего трекинга (логнормальное)
    "frag_mean_s2": 107.0,
    "frag_gap_mean_s2": 6.0,     # пауза между кусками: в кадре в среднем 20.5 объекта, как у нас
}


def _ball_dropout(ball, players, slot_team, pitch, graded, rng, p):
    L, W = pitch
    seen = np.isfinite(ball[:, 0])
    if seen[graded].mean() < p["skip_ball_if_seen_below"]:
        return ball
    b = ball[:, :2] * np.array([L, W])
    pl = players[:, :, :2] * np.array([L, W])
    with np.errstate(invalid="ignore"):
        dist = np.sqrt(((pl - b[:, None, :]) ** 2).sum(axis=2))
        crowd = (np.nan_to_num(dist, nan=1e9) < p["crowd_r"]).sum(axis=1) >= p["crowd_k"]
        air = np.nan_to_num(ball[:, 2], nan=0.0) > p["air_z"]
    weight = np.ones(len(ball))
    weight[crowd] *= p["crowd_mult"]
    weight[air] *= p["air_mult"]
    gap_mean = p["ball_gap_mean_s"] * FPS
    u = rng.random(len(ball))
    lengths = rng.exponential(gap_mean, len(ball)).astype(int) + 1

    def apply(scale):
        miss = np.zeros(len(ball), dtype=bool)
        starts = np.flatnonzero(u < np.minimum(1.0, scale * weight / gap_mean))
        for s in starts:
            miss[s:s + lengths[s]] = True
        return miss

    # подбираем частоту дыр так, чтобы доля видимого мяча совпала с целью
    lo, hi = 0.0, 50.0
    for _ in range(30):
        mid = (lo + hi) / 2
        share = (seen & ~apply(mid))[graded].mean()
        if share > p["ball_seen_target"]:
            lo = mid
        else:
            hi = mid
    out = ball.copy()
    out[apply((lo + hi) / 2)] = np.nan
    return out


def _ball_dropout_v2(ball, players, slot_team, pitch, graded, gt, rng, p):
    """Мяч пропадает прежде всего в остановках (как у трекинга Metrica), плюс дыры в игре до нужной доли."""
    L, W = pitch
    seen = np.isfinite(ball[:, 0])
    if seen[graded].mean() < p["skip_ball_if_seen_below"]:
        return ball
    n = len(ball)
    dead_miss = np.zeros(n, dtype=bool)
    e = np.diff(np.concatenate(([0], (gt == 2).astype(np.int8), [0])))
    for a, b in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
        if rng.random() < p["p_dead_hide"]:
            j = p.get("dead_jitter_s", 0.0) * FPS
            s = a + int(rng.exponential(p["dead_lag_s"] * FPS)) + (int(rng.normal(0, j)) if j else 0)
            t = b - int(rng.exponential(p["dead_lead_s"] * FPS)) + (int(rng.normal(0, j)) if j else 0)
            s, t = max(s, 0), min(t, n)
            if t > s:
                dead_miss[s:t] = True
    b = ball[:, :2] * np.array([L, W])
    pl = players[:, :, :2] * np.array([L, W])
    with np.errstate(invalid="ignore"):
        dist = np.sqrt(((pl - b[:, None, :]) ** 2).sum(axis=2))
        crowd = (np.nan_to_num(dist, nan=1e9) < p["crowd_r"]).sum(axis=1) >= p["crowd_k"]
        air = np.nan_to_num(ball[:, 2], nan=0.0) > p["air_z"]
    weight = np.where(gt == 2, 0.0, 1.0)
    weight[crowd] *= p["crowd_mult"]
    weight[air] *= p["air_mult"]
    gap_mean = p["live_gap_mean_s"] * FPS
    u = rng.random(n)
    lengths = rng.exponential(gap_mean, n).astype(int) + 1

    def apply(scale):
        miss = dead_miss.copy()
        for s in np.flatnonzero(u < np.minimum(1.0, scale * weight / gap_mean)):
            miss[s:s + lengths[s]] = True
        return miss

    lo, hi = 0.0, 50.0
    if (seen & ~apply(0.0))[graded].mean() <= p["ball_seen_target"]:
        hi = 0.0                 # одни остановки уже прячут достаточно (матч с долгими паузами)
    else:
        for _ in range(30):
            mid = (lo + hi) / 2
            if (seen & ~apply(mid))[graded].mean() > p["ball_seen_target"]:
                lo = mid
            else:
                hi = mid
    out = ball.copy()
    out[apply((lo + hi) / 2)] = np.nan
    return out


def _frag_length(rng, p):
    if p.get("mode") == "v2":
        mu = np.log(p["frag_median_s"] * FPS)
        sigma = np.sqrt(2 * max(np.log(p["frag_mean_s2"] / p["frag_median_s"]), 1e-6))
        return int(rng.lognormal(mu, sigma)) + FPS
    return int(rng.exponential(p["frag_mean_s"] * FPS)) + FPS


def _fragment_players(players, slot_team, rng, p):
    """Чистые треки людей -> безымянные слоты: куски треков ложатся в случайные свободные слоты своей команды."""
    n, s_in, _ = players.shape
    k = p["slots_per_team"]
    out = np.full((n, 2 * k, 3), np.nan)
    busy_until = np.zeros(2 * k, dtype=np.int64)       # слот занят до этого кадра
    frags = []
    for j in range(s_in):
        on = np.isfinite(players[:, j, 0])
        if not on.any():
            continue
        t, last = int(np.flatnonzero(on)[0]), int(np.flatnonzero(on)[-1])
        while t <= last:
            length = _frag_length(rng, p)
            frags.append((t, min(t + length, last + 1), j))
            gap_s = p["frag_gap_mean_s2"] if p.get("mode") == "v2" else p["frag_gap_mean_s"]
            t += length + int(rng.exponential(gap_s * FPS))
    frags.sort()
    for a, b, j in frags:
        team = int(slot_team[j])
        pool = np.arange(team * k, (team + 1) * k)
        free = pool[busy_until[pool] <= a]
        slot = int(rng.choice(free)) if len(free) else int(pool[np.argmin(busy_until[pool])])
        out[a:b, slot] = players[a:b, j]
        busy_until[slot] = b
    pitch_noise = rng.normal(0, 1, out[:, :, :2].shape)
    return out, pitch_noise


def degrade(clean, seed=0, params=None):
    """clean: dict из external/conv/*.npz -> dict того же вида, что atd.load() у настоящего матча."""
    p = dict(DEFAULTS, **(params or {}))
    rng = np.random.default_rng(seed)
    L, W = clean["pitch"]
    graded = clean["gt"] != 3
    if p.get("mode") == "v2":
        ball = _ball_dropout_v2(clean["ball"], clean["players"], clean["slot_team"], (L, W), graded,
                                clean["gt"], rng, p)
    else:
        ball = _ball_dropout(clean["ball"], clean["players"], clean["slot_team"], (L, W), graded, rng, p)
    players, noise = _fragment_players(clean["players"], clean["slot_team"], rng, p)
    players[:, :, 0] += noise[:, :, 0] * p["pos_noise_m"] / L
    players[:, :, 1] += noise[:, :, 1] * p["pos_noise_m"] / W
    players[:, :, 2] += rng.normal(0, p["speed_noise"], players.shape[:2])
    slot_team = np.repeat(np.array([0, 1], dtype=np.int8), p["slots_per_team"])
    return {"ball": ball, "players": players, "slot_team": slot_team,
            "pitch": (float(L), float(W)), "gt": clean["gt"]}


def summary(d):
    graded = d["gt"] != 3
    n_obj = np.isfinite(d["players"][:, :, 0]).sum(axis=1)
    return {"мяч виден": round(float(np.isfinite(d["ball"][graded, 0]).mean()), 3),
            "объектов в кадре": round(float(n_obj[graded].mean()), 1)}
