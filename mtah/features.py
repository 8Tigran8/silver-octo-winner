"""Признаки на кадр из трекинга. Только позиции, без личностей игроков (слоты — обрывки треков).

Все расстояния в метрах. NaN там, где мяча или игроков нет.
"""

import warnings

import numpy as np

from .config import PITCH, FPS


def _run_length(mask):
    """Для каждого кадра: длина непрерывного куска True, в который он входит (0 для False)."""
    m = mask.astype(np.int8)
    edges = np.diff(np.concatenate(([0], m, [0])))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    out = np.zeros(len(mask), dtype=np.int32)
    for a, b in zip(starts, stops):
        out[a:b] = b - a
    return out


def build(d):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # пустые кадры (перерыв) дают NaN — это норма
        return _build(d)


def _build(d):
    L, W = d.get("pitch", PITCH)       # у внешних матчей своё поле
    ball = d["ball"][:, :2] * np.array([L, W])
    pl = d["players"][:, :, :2] * np.array([L, W])
    speed = d["players"][:, :, 2]
    team = d["slot_team"]
    seen = ~np.isnan(ball[:, 0])

    # расстояние от мяча до ближайшего игрока каждой команды
    dist = np.sqrt(((pl - ball[:, None, :]) ** 2).sum(axis=2))           # (N, S)
    with np.errstate(all="ignore"):
        d_home = np.nanmin(np.where(team[None, :] == 0, dist, np.nan), axis=1)
        d_away = np.nanmin(np.where(team[None, :] == 1, dist, np.nan), axis=1)
        mean_speed = np.nanmean(speed, axis=1)

    # мяч за пределами поля (с запасом 0.5 м)
    m = 0.5
    ball_out = seen & ((ball[:, 0] < -m) | (ball[:, 0] > L + m) | (ball[:, 1] < -m) | (ball[:, 1] > W + m))

    # скорость мяча по соседним кадрам, м/с
    v = np.full(len(ball), np.nan)
    step = np.sqrt(((ball[1:] - ball[:-1]) ** 2).sum(axis=1)) * FPS
    v[1:] = step

    missing_run = _run_length(~seen) / float(FPS)                       # секунды без мяча подряд

    # центр масс команд по длине поля (для будущих признаков формы)
    with np.errstate(all="ignore"):
        cx_home = np.nanmean(np.where(team[None, :] == 0, pl[:, :, 0], np.nan), axis=1)
        cx_away = np.nanmean(np.where(team[None, :] == 1, pl[:, :, 0], np.nan), axis=1)

    return {
        "ball_seen": seen,
        "d_home": d_home,
        "d_away": d_away,
        "ball_out": ball_out,
        "ball_speed": v,
        "mean_speed": mean_speed,
        "missing_run_s": missing_run,
        "cx_home": cx_home,
        "cx_away": cx_away,
    }
