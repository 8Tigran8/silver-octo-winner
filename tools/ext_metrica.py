"""Metrica Sports sample-data, игры 1 и 2 (CSV) -> внешний матч в нашем формате (external/conv/*.npz).

Источник: github.com/metrica-sports/sample-data (условия в README: свободно, при публичном
использовании указать источник). Третью игру (формат EPTS со сменой состава по ходу матча) пока
не берём: разбор её метаданных дороже пользы, у нас уже 7 матчей DFL с настоящими метками.

Метки здесь выведены из событий Metrica, а не из исправленного трекинга, поэтому это второй сорт:
- мяч мёртв: от BALL OUT / FAULT RECEIVED / гола до начала следующего SET PIECE;
  до первого розыгрыша тайма и после последнего события тайма;
- мяч живой: владеет команда последнего «владеющего» события (PASS, SET PIECE, RECOVERY, SHOT,
  BALL LOST, CARRY). CHALLENGE (единоборство) владение не меняет — как в CHALLENGE.md организаторов:
  выигранная борьба без контроля не смена.

    python3 tools/ext_metrica.py
"""

import csv
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "external", "metrica")
DST = os.path.join(ROOT, "external", "conv")
FPS = 25
PITCH = (105.0, 68.0)
OWNING = {"PASS", "SET PIECE", "RECOVERY", "SHOT", "BALL LOST", "CARRY"}
STOPS = {"BALL OUT", "FAULT RECEIVED"}


def read_tracking(path):
    with open(path) as fh:
        rows = list(csv.reader(fh))
    head = rows[2]
    data = np.array([[np.nan if v in ("", "NaN") else float(v) for v in r] for r in rows[3:]])
    cols = [i for i, h in enumerate(head) if h.startswith("Player")]
    players = np.stack([data[:, [c, c + 1]] for c in cols], axis=1)      # (N, P, 2)
    bc = head.index("Ball")
    return data[:, 0].astype(int), data[:, 1].astype(int), players, data[:, [bc, bc + 1]]


def read_events(path):
    with open(path) as fh:
        out = []
        for r in csv.DictReader(fh):
            out.append({"team": 0 if r["Team"] == "Home" else 1, "type": r["Type"],
                        "sub": r["Subtype"], "period": int(r["Period"]),
                        "start": int(r["Start Frame"]), "end": int(r["End Frame"])})
    return out


def labels_from_events(events, frames, period):
    """Коды 0=H 1=A 2=D 3=X на каждый кадр трекинга (кадр Metrica = индекс + 1)."""
    n = len(frames)
    gt = np.full(n, 3, dtype=np.int8)
    for p in (1, 2):
        ev = sorted([e for e in events if e["period"] == p], key=lambda e: e["start"])
        on = np.flatnonzero(period == p)
        if not len(ev) or not len(on):
            continue
        lo, hi = frames[on[0]], frames[on[-1]]
        state = np.full(hi - lo + 1, 2, dtype=np.int8)       # по умолчанию мёртв
        owner, dead_from, alive = None, None, False
        cur = ev[0]["start"]
        for e in ev:
            t = e["start"]
            # закрасить промежуток [cur, t) текущим состоянием
            if alive and owner is not None:
                state[max(cur, lo) - lo:max(t, lo) - lo] = owner
            cur = t
            if e["type"] == "SET PIECE":
                alive, owner = True, e["team"]
            elif e["type"] in STOPS or (e["type"] == "SHOT" and "GOAL" in e["sub"]
                                        and "OFF" not in e["sub"] and "SAVED" not in e["sub"]):
                if alive and owner is not None and e["type"] == "BALL OUT":
                    end = max(e["end"], t)
                    state[max(cur, lo) - lo:max(end, lo) - lo] = owner
                    cur = end
                alive = False
            elif e["type"] in OWNING:
                if not alive and e["type"] in ("PASS", "RECOVERY") and owner is None:
                    alive = True
                if alive:
                    owner = e["team"]
        # после последнего события тайма — мёртв (конец тайма)
        idx = on
        gt[idx] = state[frames[idx] - lo]
    return gt


def convert(game):
    folder = os.path.join(SRC, game)
    fh = os.path.join(folder, "{}_RawTrackingData_Home_Team.csv".format(game))
    fa = os.path.join(folder, "{}_RawTrackingData_Away_Team.csv".format(game))
    period, frames, ph, ball = read_tracking(fh)
    _, frames_a, pa, _ = read_tracking(fa)
    assert np.array_equal(frames, frames_a)
    events = read_events(os.path.join(folder, "{}_RawEventsData.csv".format(game)))
    gt = labels_from_events(events, frames, period)

    # две половины подряд, между ними пустые кадры (как у DFL) — период 1 и 2
    keep = (period == 1) | (period == 2)
    pl = np.concatenate([ph, pa], axis=1)[keep]
    slot_team = np.array([0] * ph.shape[1] + [1] * pa.shape[1], dtype=np.int8)
    L, W = PITCH
    xy = pl * np.array([L, W])
    v = np.full(pl.shape[:2], np.nan)
    v[1:] = np.sqrt(((xy[1:] - xy[:-1]) ** 2).sum(axis=2)) * FPS
    players = np.concatenate([pl, v[:, :, None]], axis=2)
    b = np.concatenate([ball[keep], np.full((keep.sum(), 1), np.nan)], axis=1)
    g = gt[keep]
    # граница таймов: вставляем X-кадры
    cut = int(np.flatnonzero(period[keep] == 2)[0])
    gap = 10 * FPS
    players = np.concatenate([players[:cut], np.full((gap,) + players.shape[1:], np.nan), players[cut:]])
    b = np.concatenate([b[:cut], np.full((gap, 3), np.nan), b[cut:]])
    g = np.concatenate([g[:cut], np.full(gap, 3, dtype=np.int8), g[cut:]])

    os.makedirs(DST, exist_ok=True)
    out = os.path.join(DST, "metrica_{}.npz".format(game.split("_")[-1]))
    np.savez_compressed(out, ball=b, players=players, slot_team=slot_team, gt=g, pitch=np.array(PITCH))
    graded = g != 3
    share = np.bincount(g[graded], minlength=3)[:3] / graded.sum()
    seen = np.isfinite(b[graded, 0]).mean()
    return "{}: кадров {}, людей {}, H/A/D {:.2f}/{:.2f}/{:.2f}, мяч виден {:.0%}".format(
        game, len(g), len(slot_team), share[0], share[1], share[2], seen)


if __name__ == "__main__":
    for game in ("Sample_Game_1", "Sample_Game_2"):
        print(convert(game), flush=True)
