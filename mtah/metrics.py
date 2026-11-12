"""Своя реализация метрик организаторов: macroF1 и spellF1 (только numpy).

Написана по описанию в EVALUATION.md и по чтению evaluation/possession_metrics.py организаторов
(их код мы не запускаем). Совпадение проверяет tests/test_metrics.py: на открытых 5 минутах
четыре их базовых решения должны дать ровно опубликованные числа.

Суть:
- macroF1: среднее F1 по классам H, A, D на кадрах, где в ответах не X. Наш X на таком кадре = промах.
- spellF1: владение = непрерывный кусок H или A внутри тайма. Перед подсчётом наши X заполняются
  предыдущей меткой того же тайма (в начале тайма — первой известной). Куски сопоставляются
  один к одному жадно по пересечению/объединению (IoU) — только одной команды, — F1 считается
  при порогах 0.3, 0.4, 0.5, 0.6, 0.7 и усредняется.
"""

import numpy as np

from .config import H, A, D, X, LABELS

SPELL_IOUS = (0.3, 0.4, 0.5, 0.6, 0.7)


def codes_from_labels(labels):
    """Массив строк 'H'/'A'/'D'/'X' -> коды 0..3."""
    labels = np.asarray([str(v).strip().upper() for v in labels])
    codes = np.full(len(labels), -1, dtype=np.int8)
    for k, name in enumerate(LABELS):
        codes[labels == name] = k
    if (codes < 0).any():
        raise ValueError("метки вне H/A/D/X: {}".format(sorted(set(labels[codes < 0]))[:5]))
    return codes


def macro_f1(gt, pred):
    graded = gt != X
    g, p = gt[graded], pred[graded]
    f1s = []
    for k in (H, A, D):
        tp = float(np.sum((g == k) & (p == k)))
        fp = float(np.sum((g != k) & (p == k)))
        fn = float(np.sum((g == k) & (p != k)))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s))


def valid_runs(gt):
    """[(start, stop)) куски, где ответы не X — у матча это два тайма."""
    graded = (gt != X).astype(np.int8)
    edges = np.diff(np.concatenate(([0], graded, [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def fill_abstentions(pred, runs):
    out = pred.copy()
    for start, stop in runs:
        w = out[start:stop]
        known = np.flatnonzero(w != X)
        if not len(known):
            continue
        carry = np.maximum.accumulate(np.where(w != X, np.arange(len(w)), -1))
        carry[carry < 0] = known[0]
        out[start:stop] = w[carry]
    return out


def segments(codes, runs):
    out = []
    for start, stop in runs:
        w = codes[start:stop]
        if not len(w):
            continue
        breaks = np.flatnonzero(w[1:] != w[:-1]) + 1
        bounds = np.concatenate(([0], breaks, [len(w)]))
        for i in range(len(bounds) - 1):
            out.append((start + int(bounds[i]), start + int(bounds[i + 1]), int(w[bounds[i]])))
    return out


def spells(codes, runs):
    """Владения: непрерывные куски H или A (остановки D не считаются)."""
    return [s for s in segments(codes, runs) if s[2] in (H, A)]


def _overlaps(gt_sp, pr_sp):
    """Все пары одной команды с пересечением, по убыванию IoU (как у организаторов)."""
    out = []
    pr_starts = np.array([s[0] for s in pr_sp]) if pr_sp else np.zeros(0, int)
    pr_stops = np.array([s[1] for s in pr_sp]) if pr_sp else np.zeros(0, int)
    for i, (gs, ge, gk) in enumerate(gt_sp):
        # кандидаты, которые пересекаются по времени
        cand = np.flatnonzero((pr_starts < ge) & (pr_stops > gs))
        for j in cand.tolist():
            ps, pe, pk = pr_sp[j]
            if pk != gk:
                continue
            inter = min(ge, pe) - max(gs, ps)
            union = max(ge, pe) - min(gs, ps)
            out.append((inter / float(union), i, j))
    out.sort(key=lambda t: (-t[0], t[1], t[2]))
    return out


def spell_f1(gt, pred, ious=SPELL_IOUS, detail=False):
    runs = valid_runs(gt)
    filled = fill_abstentions(pred, runs)
    gt_sp, pr_sp = spells(gt, runs), spells(filled, runs)
    ov = _overlaps(gt_sp, pr_sp)
    f1s, per = [], {}
    for thr in ious:
        used_g, used_p, tp = set(), set(), 0
        for iou, i, j in ov:
            if iou < thr:
                break
            if i in used_g or j in used_p:
                continue
            used_g.add(i)
            used_p.add(j)
            tp += 1
        fp, fn = len(pr_sp) - tp, len(gt_sp) - tp
        prec = tp / float(tp + fp) if tp + fp else 0.0
        rec = tp / float(tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per[thr] = (prec, rec, f1)
        f1s.append(f1)
    mean = float(np.mean(f1s))
    if not detail:
        return mean
    return {
        "spell_f1": mean,
        "per_iou": per,
        "n_gt_spells": len(gt_sp),
        "n_pred_spells": len(pr_sp),
        "pred_mean_dur_s": float(np.mean([b - a for a, b, _ in pr_sp])) / 25 if pr_sp else 0.0,
        "gt_mean_dur_s": float(np.mean([b - a for a, b, _ in gt_sp])) / 25 if gt_sp else 0.0,
    }


def evaluate(gt, pred):
    if len(gt) != len(pred):
        raise ValueError("длины не совпадают: {} и {}".format(len(gt), len(pred)))
    d = spell_f1(gt, pred, detail=True)
    d["macro_f1"] = macro_f1(gt, pred)
    return d
