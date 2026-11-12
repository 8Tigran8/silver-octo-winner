"""Сверка порчи внешних матчей с нашим трекингом — только сводные цифры, без ответов и видео.

Правила позволяют автоматические проверки на самих данных без ответов. Здесь сравниваем
характер дыр и обрывков (доля видимого мяча, длина дыр, длина кусков треков, объекты в кадре,
скорости), чтобы испорченные внешние матчи были похожи на наш. Ответы матча не читаются.

    .venv/bin/python tools/sverka_porchi.py [--params '{"ball_gap_mean_s": 1.0}']
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mtah import atd, degrade, external, features   # noqa: E402
from mtah.config import graded_mask, PITCH, FPS     # noqa: E402


def runs(mask):
    m = np.concatenate(([0], mask.astype(np.int8), [0]))
    e = np.diff(m)
    return np.flatnonzero(e == -1) - np.flatnonzero(e == 1)


def stats(ball, players, pitch, graded):
    L, W = pitch
    seen = np.isfinite(ball[:, 0])
    g = graded
    gaps = runs(~seen & g) / FPS
    on = np.isfinite(players[:, :, 0]) & g[:, None]
    frag = np.concatenate([runs(on[:, s]) for s in range(on.shape[1])]) / FPS
    n_obj = on.sum(axis=1)[g]
    sp = players[:, :, 2][on]
    b = ball[:, :2] * np.array([L, W])
    step = np.sqrt(((b[1:] - b[:-1]) ** 2).sum(axis=1)) * FPS
    step = step[np.isfinite(step) & g[1:]]
    out = (b[:, 0] < -0.5) | (b[:, 0] > L + 0.5) | (b[:, 1] < -0.5) | (b[:, 1] > W + 0.5)
    return {
        "мяч виден": round(float(seen[g].mean()), 3),
        "дыр у мяча на 1000 кадров": round(1000.0 * len(gaps) / g.sum(), 2),
        "дыра: средняя, медиана, 90% (с)": [round(float(np.mean(gaps)), 2), round(float(np.median(gaps)), 2),
                                           round(float(np.percentile(gaps, 90)), 2)],
        "кусок трека: средний, медиана (с)": [round(float(np.mean(frag)), 1), round(float(np.median(frag)), 1)],
        "объектов в кадре": round(float(n_obj.mean()), 1),
        "объектов 10%/50%/90%": np.percentile(n_obj, [10, 50, 90]).round(0).tolist(),
        "скорость игроков медиана/90% (м/с?)": [round(float(np.nanmedian(sp)), 2),
                                               round(float(np.nanpercentile(sp, 90)), 2)],
        "скорость мяча медиана/90% (м/с)": [round(float(np.median(step)), 2),
                                           round(float(np.percentile(step, 90)), 2)],
        "мяч за линией": round(float(out[g & seen].mean()), 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="{}")
    ap.add_argument("--only-ext", action="store_true")
    args = ap.parse_args()
    if not args.only_ext:
        d = atd.load()
        print("НАШ МАТЧ:", json.dumps(stats(d["ball"], d["players"], PITCH, graded_mask(len(d["ball"]))),
                                      ensure_ascii=False))
    params = json.loads(args.params)
    for n in external.names():
        z = np.load(os.path.join(external.CONV_DIR, n + ".npz"))
        clean = {k: z[k] for k in z.files}
        dd = degrade.degrade(clean, seed=0, params=params)
        print(n, json.dumps(stats(dd["ball"], dd["players"], dd["pitch"], dd["gt"] != 3), ensure_ascii=False))


if __name__ == "__main__":
    main()
