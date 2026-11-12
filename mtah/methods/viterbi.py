"""v0: признаки -> очки за H / A / D на каждый кадр -> Витерби с ценой смены состояния.

Почему так: spellF1 считает целые владения, поэтому главное — не дёргаться. Витерби выбирает
последовательность меток, где каждая смена стоит штраф, и смена остаётся только если признаки
уверенно за неё «платят». Так одно решение заменяет и сглаживание, и склейку коротких кусков.

ПРАВИЛО (разъяснения организаторов 04.11): параметры ниже нельзя подгонять по 5 минутам ответов
и по видео матча. Значения подобраны только на внешних матчах (IDSSE/DFL, Metrica sample): tools/tune_v0.py.
"""

import numpy as np

from ..config import HALVES, N_FRAMES, H, A, D

DEFAULTS = {
    # Значения подобраны tools/tune_v0.py 10.11.2026 только на внешних матчах (6 матчей IDSSE/DFL,
    # испорченных mtah/degrade.py); проверка на J03WR9 и Metrica 1–2. Комментарии ниже — смысл параметра.
    # владение
    "r_control": 0.708,        # м: игрок ближе этого к мячу — вероятно, контролирует
    "team_gain": 0.078,        # сила голоса «чей ближе» на кадр
    "margin_scale": 0.556,     # м: насколько разница расстояний между командами считается уверенной
    # мёртвый мяч
    "w_out": 0.939,            # мяч за линией
    "w_missing": 0.15,        # мяча нет дольше missing_s секунд
    "missing_s": 1.73,
    "w_slow": 1.572,           # игроки почти стоят
    "slow_speed": 1.868,       # м/с: средняя скорость ниже — похоже на паузу
    "alive_bias": 0.432,       # мелкое предпочтение «живого» мяча, когда сигналов нет
    # цена смены
    "cost_ha": 0.656,          # H <-> A
    "cost_d": 11.329,          # вход и выход из остановки
}


def emissions(f, p):
    n = len(f["ball_seen"])
    sc = np.zeros((n, 3))
    dh, da = f["d_home"], f["d_away"]
    seen = f["ball_seen"] & ~f["ball_out"] & np.isfinite(dh) & np.isfinite(da)

    # кто ближе к мячу: разница расстояний в tanh, ослабляем, если никто не рядом
    margin = np.zeros(n)
    margin[seen] = np.tanh((da[seen] - dh[seen]) / p["margin_scale"])
    near = np.zeros(n)
    near[seen] = np.clip(p["r_control"] / np.maximum(np.minimum(dh[seen], da[seen]), 1e-3), 0, 1)
    vote = p["team_gain"] * margin * near
    sc[:, H] += vote
    sc[:, A] -= vote

    # остановка
    dead = p["w_out"] * f["ball_out"].astype(float)
    dead += p["w_missing"] * (f["missing_run_s"] > p["missing_s"])
    ms = np.nan_to_num(f["mean_speed"], nan=p["slow_speed"])
    dead += p["w_slow"] * np.clip((p["slow_speed"] - ms) / p["slow_speed"], 0, 1)
    sc[:, D] += dead - p["alive_bias"]
    return sc


def viterbi(scores, cost_ha, cost_d):
    """Максимум суммы очков минус штрафы за смены. Состояния 0=H, 1=A, 2=D.

    Цикл на простых числах Python: для трёх состояний это в разы быстрее, чем numpy на каждом шаге.
    """
    n = len(scores)
    sc = scores.tolist()
    back = bytearray(3 * n)
    a0, a1, a2 = sc[0]
    for t in range(1, n):
        # как np.argmax: при равенстве берём меньший номер состояния
        c0 = (a0, 0) if a0 >= a1 - cost_ha else (a1 - cost_ha, 1)
        if a2 - cost_d > c0[0]:
            c0 = (a2 - cost_d, 2)
        c1 = (a0 - cost_ha, 0) if a0 - cost_ha >= a1 else (a1, 1)
        if a2 - cost_d > c1[0]:
            c1 = (a2 - cost_d, 2)
        c2 = (a0 - cost_d, 0) if a0 >= a1 else (a1 - cost_d, 1)
        if a2 > c2[0]:
            c2 = (a2, 2)
        s0, s1, s2 = sc[t]
        a0, a1, a2 = c0[0] + s0, c1[0] + s1, c2[0] + s2
        back[3 * t], back[3 * t + 1], back[3 * t + 2] = c0[1], c1[1], c2[1]
    path = np.zeros(n, dtype=np.int8)
    last = max(range(3), key=lambda k: (a0, a1, a2)[k])
    path[-1] = last
    for t in range(n - 1, 0, -1):
        last = back[3 * t + last]
        path[t - 1] = last
    return path


def label_segments(sc, p, segments, n):
    """Метки на n кадров: Витерби внутри каждого куска [a, b), вне кусков X."""
    labels = np.full(n, "X", dtype="<U1")
    for a, b in segments:
        path = viterbi(sc[a:b], p["cost_ha"], p["cost_d"])
        labels[a:b] = np.array(["H", "A", "D"])[path]
    return labels


def run(d, f, params=None):
    p = dict(DEFAULTS, **(params or {}))
    sc = emissions(f, p)
    # вне таймов X: там не оценивают
    return label_segments(sc, p, [(first, last + 1) for _, first, last in HALVES], N_FRAMES)
