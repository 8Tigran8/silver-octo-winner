"""Проверка, что наша метрика считает ровно как у организаторов.

Опубликовано в README организаторов (оценка на открытых 5 минутах):
    baseline_smooth125  macroF1 0.5812  spellF1 0.4242  владений 16  ср. длит. 18.5 c
    baseline_smooth50   macroF1 0.6207  spellF1 0.3256  26  10.8
    baseline_smooth25   macroF1 0.6200  spellF1 0.2868  36  7.6
    baseline_raw        macroF1 0.5972  spellF1 0.0378  131 2.0
    ответы: 17 владений, средняя длительность 14.2 c

Плюс синтетические случаи из EVALUATION.md (идеал = 1, сдвиг на 1 кадр = 1, константа = 0).

Запуск: .venv/bin/python -m tests.test_metrics
"""

import os
import sys

import numpy as np

from mtah import metrics as M
from mtah import submission as S
from mtah.config import REF_DIR, PUBLIC_GT, X, H, A, D

PUBLISHED = {
    "baseline_smooth125": (0.5812, 0.4242, 16, 18.5),
    "baseline_smooth50": (0.6207, 0.3256, 26, 10.8),
    "baseline_smooth25": (0.6200, 0.2868, 36, 7.6),
    "baseline_raw": (0.5972, 0.0378, 131, 2.0),
}


def check_published():
    gt = M.codes_from_labels(S.load_labels(PUBLIC_GT))
    ok = True
    for name, (mf, sf, n, dur) in PUBLISHED.items():
        pred = M.codes_from_labels(S.load_labels(os.path.join(REF_DIR, name + ".csv")))
        r = M.evaluate(gt, pred)
        # организаторы считают число владений на ВСЕХ оцениваемых кадрах ответа; у нас — только
        # внутри окна ответов, поэтому сравниваем лишь F1 (их «poss» в таблице тоже по окну).
        good = (round(r["macro_f1"], 4) == mf and round(r["spell_f1"], 4) == sf
                and r["n_pred_spells"] == n and round(r["pred_mean_dur_s"], 1) == dur)
        ok &= good
        print("{:<20} macroF1 {:.4f} (ждём {})  spellF1 {:.4f} (ждём {})  владений {} ({})  {}".format(
            name, r["macro_f1"], mf, r["spell_f1"], sf, r["n_pred_spells"], n,
            "OK" if good else "РАСХОЖДЕНИЕ"))
    r = M.spell_f1(gt, gt, detail=True)
    print("ответы: владений {}, средняя длительность {:.1f} c (ждём 17 и 14.2)".format(
        r["n_gt_spells"], r["gt_mean_dur_s"]))
    ok &= r["n_gt_spells"] == 17
    return ok


def check_synthetic():
    rng = np.random.default_rng(0)
    n = 20000
    gt = np.full(n, X, dtype=np.int8)
    pos, lab = 1000, H
    while pos < 19000:
        length = int(rng.integers(50, 600))
        gt[pos:pos + length] = lab
        pos += length
        lab = {H: rng.choice([A, D]), A: rng.choice([H, D]), D: rng.choice([H, A])}[lab]
    gt[19000:] = X
    ok = True
    ok &= M.spell_f1(gt, gt) == 1.0
    late = gt.copy()
    late[1:] = gt[:-1]
    late[gt == X] = X
    ok &= M.spell_f1(gt, late) == 1.0          # сдвиг на 1 кадр бесплатен
    const = np.where(gt == X, X, H).astype(np.int8)
    ok &= M.spell_f1(gt, const) == 0.0         # константа = 0
    print("синтетика: идеал=1, сдвиг на кадр=1, константа=0 ->", "OK" if ok else "ОШИБКА")
    return ok


if __name__ == "__main__":
    a = check_published()
    b = check_synthetic()
    print("ИТОГ:", "всё сходится" if a and b else "есть расхождения")
    sys.exit(0 if a and b else 1)
