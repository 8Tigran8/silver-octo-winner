"""Точная копия базового решения организаторов (starter/baseline.py), переписанная у нас.

Ближайший к мячу игрок -> его команда; мяча нет -> никто; затем голосование большинством в окне ±w.
Нужна не ради счёта, а как проверка конвейера: наш out/baseline_smooth50.csv должен совпасть
с их ref/baseline_smooth50.csv байт в байт. Совпало — значит, чтение трекинга и выравнивание кадров верны.
"""

import numpy as np

from ..config import PITCH

HOME, AWAY, NOBODY = 0, 1, -1


def nearest_codes(d):
    L, W = d.get("pitch", PITCH)
    ball = d["ball"][:, :2] * np.array([L, W])
    pl = d["players"][:, :, :2] * np.array([L, W])
    dist = ((pl - ball[:, None, :]) ** 2).sum(axis=2)          # NaN где нет игрока/мяча
    dist = np.where(np.isnan(dist), np.inf, dist)
    best = np.argmin(dist, axis=1)                              # первый минимум, как у них (строгое <)
    has = np.isfinite(dist[np.arange(len(dist)), best])
    codes = np.where(has, d["slot_team"][best], NOBODY).astype(np.int8)
    return codes


def smooth(codes, window):
    if window <= 0:
        return np.where(codes == HOME, "H", np.where(codes == AWAY, "A", "D"))
    hc = np.concatenate(([0], np.cumsum(codes == HOME)))
    ac = np.concatenate(([0], np.cumsum(codes == AWAY)))
    idx = np.arange(len(codes))
    lo = np.maximum(0, idx - window)
    hi = np.minimum(len(codes), idx + window + 1)
    h, a = hc[hi] - hc[lo], ac[hi] - ac[lo]
    labels = np.full(len(codes), "D", dtype="<U1")
    seen = (h + a) > 0
    labels[seen] = np.where(h[seen] >= a[seen], "H", "A")
    return labels


def run(d, feats=None, window=50):
    return smooth(nearest_codes(d), window)
