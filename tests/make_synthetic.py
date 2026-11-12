"""Синтетический «матч» в формате организаторов — чтобы прогнать конвейер до получения данных.

Ничего общего с настоящим матчем, кроме формата файлов и длины (199 319 кадров, те же таймы).
Пишет <папка>/tactical/BAR-ATH_tactical_atd.{xml,txt} и <папка>/truth.csv (ответы синтетики).

    .venv/bin/python -m tests.make_synthetic /tmp/mtah-synth
    MTAH_DATA_DIR=/tmp/mtah-synth .venv/bin/python run.py v0 --gt /tmp/mtah-synth/truth.csv
"""

import os
import sys

import numpy as np

from mtah.config import N_FRAMES, HALVES, PITCH, FPS

N_SLOTS = 46


def simulate(rng):
    labels = np.array(["X"] * N_FRAMES, dtype="<U1")
    for _, first, last in HALVES:
        t, state = first, "D"
        while t <= last:
            if state == "D":
                length = int(rng.integers(5 * FPS, 40 * FPS))
                nxt = rng.choice(["H", "A"])
            else:
                length = int(rng.exponential(12 * FPS)) + FPS
                nxt = rng.choice(["H", "A", "D"], p=[0.0, 0.8, 0.2] if state == "H" else [0.8, 0.0, 0.2])
            labels[t:min(t + length, last + 1)] = state
            t += length
            state = nxt
    return labels


def write(folder, seed=0):
    rng = np.random.default_rng(seed)
    labels = simulate(rng)
    L, W = PITCH
    os.makedirs(os.path.join(folder, "tactical"), exist_ok=True)
    with open(os.path.join(folder, "truth.csv"), "w") as fh:
        fh.write("frame,label\n")
        fh.writelines("{},{}\n".format(i, l) for i, l in enumerate(labels))

    meta = ['<?xml version="1.0"?>', "<Main><Metadata><Teams>",
            '<Team id="T1"><Name>Home FC</Name></Team>', '<Team id="T2"><Name>Away FC</Name></Team>',
            "</Teams><Players>"]
    slot_team = np.array([0] * 23 + [1] * 23)
    for s in range(N_SLOTS):
        meta.append('<Player id="P{}" teamId="{}"><Name>Track {}</Name></Player>'.format(
            s + 1, "T1" if slot_team[s] == 0 else "T2", s + 1))
    meta.append("</Players></Metadata></Main>")
    with open(os.path.join(folder, "tactical", "BAR-ATH_tactical_atd.xml"), "w") as fh:
        fh.write("\n".join(meta))

    active = np.r_[0:11, 23:34]                        # 22 игрока на поле
    pos = rng.uniform([0.1, 0.1], [0.9, 0.9], size=(N_SLOTS, 2))
    ball = np.array([0.5, 0.5])
    holder = 0
    lines = []
    for i in range(N_FRAMES):
        lab = labels[i]
        moving = lab in "HA"
        step = rng.normal(0, 0.0012 if moving else 0.0003, size=(N_SLOTS, 2))
        pos = np.clip(pos + step, 0.02, 0.98)
        if lab in "HA":
            team_slots = active[:11] if lab == "H" else active[11:]
            if i % 50 == 0 or holder not in team_slots:
                holder = int(rng.choice(team_slots))
            ball = pos[holder] + rng.normal(0, 0.004, 2)
        elif lab == "D":
            ball = np.array([1.02, ball[1]]) if i % 2000 < 1500 else ball
        missing = rng.random() < (0.35 if moving else 0.6)
        groups = []
        for s in range(N_SLOTS):
            if s in active and lab != "X":
                sp = abs(rng.normal(3.0 if moving else 0.8, 0.5))
                groups.append("{:.5f},{:.5f},{:.2f}".format(pos[s, 0], pos[s, 1], sp))
            else:
                groups.append("NaN,NaN,NaN")
        b = "NaN,NaN,NaN" if (missing or lab == "X") else "{:.5f},{:.5f},0".format(ball[0], ball[1])
        lines.append("{}:{}:{}\n".format(i + 1, ";".join(groups), b))
    with open(os.path.join(folder, "tactical", "BAR-ATH_tactical_atd.txt"), "w") as fh:
        fh.writelines(lines)
    return folder


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/mtah-synth"
    write(out)
    print("синтетика записана в", out)
