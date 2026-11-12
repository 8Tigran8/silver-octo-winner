"""Когда пропадает мяч: наш трекинг против внешних матчей (только сводные цифры, ответы нашего матча не читаются)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mtah import atd, degrade                       # noqa: E402
from mtah.config import graded_mask, PITCH          # noqa: E402


def st(ball, players, pitch, g, gt=None):
    L, W = pitch
    seen = np.isfinite(ball[:, 0])
    with np.errstate(all="ignore"):
        import warnings
        warnings.simplefilter("ignore")
        ms = np.nanmean(players[:, :, 2], axis=1)
    bins = [0, 0.7, 1.2, 1.8, 2.5, 99]
    out = []
    for a, b in zip(bins[:-1], bins[1:]):
        m = g & (ms >= a) & (ms < b)
        out.append((round(float(m.mean()), 2), round(float(1 - seen[m].mean()), 2)))
    e = np.diff(np.concatenate(([0], (~seen & g).astype(int), [0])))
    s0, s1 = np.flatnonzero(e == 1), np.flatnonzero(e == -1)
    lst = s0[s0 > 0] - 1
    bx = ball[lst, :2] * np.array([L, W])
    ok = np.isfinite(bx[:, 0])
    bx = bx[ok]
    near = (bx[:, 0] < 4) | (bx[:, 0] > L - 4) | (bx[:, 1] < 3) | (bx[:, 1] > W - 3)
    r = {"скорость: (доля кадров, доля без мяча)": out, "дыр начинается у линии": round(float(near.mean()), 2)}
    if gt is not None:
        dead = gt == 2
        r["P(мёртв|нет мяча)"] = round(float(dead[g & ~seen].mean()), 2)
        r["P(мёртв|мяч есть)"] = round(float(dead[g & seen].mean()), 2)
        lg = np.zeros(len(g), bool)
        for a, b in zip(s0, s1):
            if b - a > 250:
                lg[a:b] = True
        r["P(мёртв|дыра>10с)"] = round(float(dead[lg].mean()), 2)
    return r


d = atd.load()
print("НАШ", st(d["ball"], d["players"], PITCH, graded_mask(len(d["ball"]))))
for n in ["metrica_1", "metrica_2", "idsse_J03WMX"]:
    z = np.load(os.path.join("external", "conv", n + ".npz"))
    c = {k: z[k] for k in z.files}
    if n.startswith("metrica"):
        print(n, "чистый", st(c["ball"], c["players"], tuple(c["pitch"]), c["gt"] != 3, c["gt"]))
    dd = degrade.degrade(c, seed=0, params=__import__('json').loads(os.environ.get('DP','{}')))
    print(n, "испорч", st(dd["ball"], dd["players"], dd["pitch"], dd["gt"] != 3, dd["gt"]))
