"""v1: признаки (mtah/features2.py) -> две модели бустинга -> Витерби на 4 состояния с мин. длительностью.

Модели учатся ТОЛЬКО на внешних матчах (IDSSE/DFL, Metrica sample), испорченных под наш трекинг
(mtah/degrade.py). На нашем матче ничего не обучается и не подбирается (разъяснения организаторов 04.11).

- «мяч мёртв»: P(D | признаки);
- «кто владеет»: P(H | признаки, мяч в игре); учится на прямых и зеркальных (команды переставлены)
  данных, применяется как среднее прямого и зеркального ответа — без перекоса в пользу хозяев.
- Декодер: состояния H, A, D-после-H, D-после-A. Цены смен разные: H<->A, вход в остановку,
  выход той же командой, выход другой командой. У каждого состояния минимальная длина (цепочка
  подсостояний), так короткие ложные владения не появляются. Декодируем с шагом `stride` кадров:
  сдвиг на кадр-два метрика не штрафует, а считать в разы быстрее.
- Конец остановки подтягиваем к первому кадру, где мяч снова виден и движется (удар), если он рядом.
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .. import features2
from ..config import HALVES, N_FRAMES, H, A, D, X

MODEL = {"max_iter": 250, "learning_rate": 0.08, "max_leaf_nodes": 31, "min_samples_leaf": 200,
         "l2_regularization": 1.0, "max_bins": 128}
TRAIN_STRIDE = 4          # каждый 4-й кадр в обучение (соседние почти одинаковы)
SEED = 0

DECODER = {
    "stride": 5,
    "emit_scale": 1.0,
    "cost_ha": 6.0,        # смена H <-> A без остановки
    "cost_in": 6.0,        # H -> D(H), A -> D(A)
    "cost_same": 6.0,      # D(H) -> H, D(A) -> A
    "cost_other": 6.0,     # D(H) -> A, D(A) -> H
    "min_team_s": 1.0,     # минимальная длина владения, с
    "min_dead_s": 1.0,     # минимальная длина остановки, с
    "snap_s": 0.0,         # подтягивать конец остановки к удару не дальше, чем на столько секунд (0 = нет)
    "kick_speed": 4.0,     # м/с: мяч «пошёл»
}


def _clf():
    return HistGradientBoostingClassifier(random_state=SEED, early_stopping=False, **MODEL)


def training_rows(X, Xs, gt, stride=TRAIN_STRIDE, offset=0):
    """-> (X_dead, y_dead, X_team, y_team) с зеркальными копиями."""
    idx = np.arange(offset, len(gt), stride)
    idx = idx[gt[idx] != X_CODE]
    g = gt[idx]
    Xd = np.concatenate([X[idx], Xs[idx]])
    yd = np.concatenate([g == D, g == D]).astype(np.int8)
    live = idx[(g == H) | (g == A)]
    gl = gt[live]
    Xt = np.concatenate([X[live], Xs[live]])
    yt = np.concatenate([gl == H, gl == A]).astype(np.int8)    # в зеркале хозяева = бывшие гости
    return Xd, yd, Xt, yt


X_CODE = X


def fit(rows):
    """rows: список кортежей training_rows по матчам -> (модель мёртвого мяча, модель владения)."""
    Xd = np.concatenate([r[0] for r in rows])
    yd = np.concatenate([r[1] for r in rows])
    Xt = np.concatenate([r[2] for r in rows])
    yt = np.concatenate([r[3] for r in rows])
    md = _clf().fit(Xd, yd)
    mt = _clf().fit(Xt, yt)
    return md, mt


def predict(models, X, Xs):
    """-> pD, pH (P(H | мяч в игре)), симметрично по командам."""
    md, mt = models
    pD = 0.5 * (md.predict_proba(X)[:, 1] + md.predict_proba(Xs)[:, 1])
    pH = 0.5 * (mt.predict_proba(X)[:, 1] + (1.0 - mt.predict_proba(Xs)[:, 1]))
    return pD, pH


def _viterbi_chain(E, C, m):
    """E (T,4) лог-очки по базовым состояниям; C (4,4) цены переходов (inf = нельзя); m (4,) длины цепочек.

    Подсостояния: у базового k цепочка из m[k]; вход только в первое, выход только из последнего,
    последнее может оставаться на месте. -> путь базовых состояний (T,).
    """
    T = len(E)
    m = [max(1, int(v)) for v in m]
    first = np.cumsum([0] + m[:-1])
    last = first + np.array(m) - 1
    M = int(sum(m))
    base = np.repeat(np.arange(4), m)
    prev = np.arange(M) - 1                       # сдвиг по цепочке
    is_first = np.zeros(M, bool)
    is_first[first] = True
    NEG = -1e18
    V = np.full(M, NEG)
    V[first] = E[0]
    bp_entry = np.zeros((T, 4), np.int8)          # откуда вошли в первое подсостояние
    bp_stay = np.zeros((T, 4), bool)              # последнее подсостояние: осталось на месте
    Cn = np.where(np.isfinite(C), C, 1e18)
    single = np.array(m) == 1
    for t in range(1, T):
        lastv = V[last]
        cand = np.where(is_first, NEG, V[np.maximum(prev, 0)])
        stay = lastv
        moved = cand[last]
        stay_better = stay >= moved
        cand[last] = np.where(stay_better, stay, moved)
        tr = lastv[:, None] - Cn
        np.fill_diagonal(tr, NEG)
        ent = tr.max(axis=0)
        bp_entry[t] = tr.argmax(axis=0)
        # в первое подсостояние: вход извне (для цепочки длины 1 — ещё и «остаться»)
        f_old = cand[first]
        take_entry = ent > f_old
        cand[first] = np.where(take_entry, ent, f_old)
        bp_stay[t] = np.where(single, ~take_entry, stay_better)
        V = cand + E[t][base]
    # обратный ход
    path = np.zeros(T, np.int8)
    s = int(np.argmax(V))
    k = int(base[s])
    j = s - first[k]
    for t in range(T - 1, -1, -1):
        path[t] = k
        if t == 0:
            break
        if j == m[k] - 1 and bp_stay[t, k]:
            continue
        if j > 0:
            j -= 1
            continue
        k = int(bp_entry[t, k])
        j = m[k] - 1
    return path


def decode_segment(pD, pH, p):
    """Метки 0=H,1=A,2=D на кусок кадров."""
    st = int(p["stride"])
    n = len(pD)
    eps = 1e-4
    pD = np.clip(pD, eps, 1 - eps)
    pH = np.clip(pH, eps, 1 - eps)
    lh = np.log1p(-pD) + np.log(pH)
    la = np.log1p(-pD) + np.log1p(-pH)
    ld = np.log(pD)
    E = np.stack([lh, la, ld, ld], axis=1) * p["emit_scale"]
    nb = (n + st - 1) // st
    pad = nb * st - n
    if pad:
        E = np.concatenate([E, np.repeat(E[-1:], pad, axis=0)])
    Eb = E.reshape(nb, st, 4).sum(axis=1)
    inf = np.inf
    C = np.array([
        [0, p["cost_ha"], p["cost_in"], inf],
        [p["cost_ha"], 0, inf, p["cost_in"]],
        [p["cost_same"], p["cost_other"], 0, inf],
        [p["cost_other"], p["cost_same"], inf, 0],
    ], dtype=float)
    fps_b = 25.0 / st
    mt = max(1, int(round(p["min_team_s"] * fps_b)))
    mdd = max(1, int(round(p["min_dead_s"] * fps_b)))
    path_b = _viterbi_chain(Eb, C, [mt, mt, mdd, mdd])
    path = np.repeat(path_b, st)[:n]
    return np.where(path >= 2, D, path).astype(np.int8)


def snap_restarts(codes, kick, max_frames):
    """Конец остановки (D -> H/A) сдвигаем на ближайший кадр «мяч пошёл» в пределах max_frames."""
    if max_frames <= 0:
        return codes
    out = codes.copy()
    ch = np.flatnonzero((codes[:-1] == D) & (codes[1:] != D)) + 1
    kicks = np.flatnonzero(kick)
    for c in ch:
        j = np.searchsorted(kicks, c - max_frames)
        if j >= len(kicks) or abs(kicks[j] - c) > max_frames:
            continue
        # ближайший к c удар
        cands = kicks[j:np.searchsorted(kicks, c + max_frames + 1)]
        k = int(cands[np.argmin(np.abs(cands - c))])
        team = codes[c]
        if k > c:
            out[c:k] = D
        elif k < c:
            seg_start = k
            # не залезаем на прошлое владение: только внутри остановки
            out[seg_start:c] = np.where(codes[seg_start:c] == D, team, codes[seg_start:c])
    return out


def kick_frames(d, p):
    """Кадр, где мяч виден и после видимого или отсутствующего мяча начал двигаться быстрее kick_speed."""
    from ..config import PITCH, FPS
    L, W = d.get("pitch", PITCH)
    b = d["ball"][:, :2] * np.array([L, W])
    v = np.full(len(b), np.nan)
    v[1:] = np.sqrt(((b[1:] - b[:-1]) ** 2).sum(axis=1)) * FPS
    fast = np.nan_to_num(v) > p["kick_speed"]
    slow_before = np.zeros(len(b), bool)
    slow_before[1:] = ~fast[:-1]
    return fast & slow_before


def label(pD, pH, segments, n, p, kick=None):
    labels = np.full(n, X, dtype=np.int8)
    for a, b in segments:
        labels[a:b] = decode_segment(pD[a:b], pH[a:b], p)
    if kick is not None and p.get("snap_s", 0) > 0:
        labels = snap_restarts(labels, kick, int(round(p["snap_s"] * 25)))
    return labels


def run(d, models, params=None):
    """Наш матч: признаки -> вероятности (1-й и 2-й ярус) -> метки ('H'/'A'/'D'/'X') на N_FRAMES кадров.

    Второй тайм оценивается с 110808, а начинается с 113742 (стартовый код организаторов,
    starter/load_atd.py и load_smart_tagging.py: «2nd half kickoff»): до начала тайма мяч мёртв.
    """
    p = dict(DECODER, **(params or {}))
    segs = [(first, last + 1) for _, first, last in HALVES]
    X_, _ = features2.build(d, segs)
    Xs, _ = features2.build(d, segs, swap=True)
    pD, pH = predict(models["stage1"], X_, Xs)
    if models.get("stage2") is not None:
        from .. import stack
        pD, pH = stack.predict(models["stage2"], pD, pH)
    play = [(max(a, KICKOFF.get(i, a)), b) for i, (a, b) in enumerate(segs)]
    codes = label(pD, pH, play, N_FRAMES, p, kick_frames(d, p))
    for i, (a, b) in enumerate(segs):
        codes[a:max(a, KICKOFF.get(i, a))] = D
    return np.array(["H", "A", "D", "X"])[codes]


KICKOFF = {1: 113742}     # тайм 2 (индекс 1): кадр начала игры по стартовому коду организаторов
