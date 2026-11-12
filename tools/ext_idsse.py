"""IDSSE / DFL (7 матчей Бундеслиги, CC BY 4.0) -> внешний матч в нашем формате (external/conv/*.npz).

Источник: Bassek et al., Scientific Data 2025, doi 10.1038/s41597-025-04505-y; зеркало
huggingface.co/datasets/pysport/idsse-data. Файлы кладём в external/idsse/ (в git не попадают).

Почему этот набор главный: в DFL у мяча на каждом кадре есть BallStatus (1 живой, 0 мёртвый) и
BallPossession (1 хозяева, 2 гости) — это ровно то, из чего организаторы строили ответы для нашего
матча («кадр D, где исправленный трекинг пишет, что мяч мёртв, иначе команда-владелец»).

Формат на выходе (общий для всех внешних наборов, см. mtah/external.py):
    ball (N,3)        x, y нормированы 0..1 по размеру поля, z в метрах (или NaN)
    players (N,S,3)   x, y нормированы 0..1, скорость м/с; S = все люди матча (чистые личности)
    slot_team (S,)    0 хозяева, 1 гости
    gt (N,)           коды 0=H 1=A 2=D 3=X; X между таймами
    pitch (2,)        длина и ширина поля, м

    python3 tools/ext_idsse.py               # все матчи из external/idsse
"""

import glob
import os
import re
import sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "external", "idsse")
DST = os.path.join(ROOT, "external", "conv")
FPS = 25
GAP = 10 * FPS          # пустые кадры (X) между таймами

RE_SET = re.compile(r'GameSection="(\w+)".*?TeamId="([^"]+)".*?PersonId="([^"]+)"')
RE_FR = re.compile(r'N="(\d+)".*? X="([-\d.]+)" Y="([-\d.]+)"')
RE_BALL = re.compile(r'Z="([-\d.]+)".*?BallPossession="(\d)" BallStatus="(\d)"')
RE_PITCH = re.compile(r'<PitchSize X="([\d.]+)" Y="([\d.]+)"')


def roles(info_path):
    txt = open(info_path, encoding="utf-8").read()
    out = {}
    for m in re.finditer(r'<Team [^>]*>', txt):
        tid = re.search(r'TeamId="([^"]+)"', m.group(0))
        role = re.search(r'Role="([^"]+)"', m.group(0))
        if tid and role:
            out[tid.group(1)] = role.group(1)
    return out


def parse_positions(path):
    """-> pitch, {(section, team, person): (frames, xy[, ball extra])}"""
    pitch, sets, cur, rows = None, {}, None, []

    def flush():
        if cur is not None and rows:
            sets[cur] = np.array(rows, dtype=np.float64)

    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if "<Frame " in line:
                m = RE_FR.search(line)
                row = [float(m.group(1)), float(m.group(2)), float(m.group(3))]
                if cur[1] == "BALL":
                    b = RE_BALL.search(line)
                    row += [float(b.group(1)), float(b.group(2)), float(b.group(3))]
                rows.append(row)
            elif "<FrameSet" in line:
                flush()
                m = RE_SET.search(line)
                cur, rows = (m.group(1), m.group(2), m.group(3)), []
            elif pitch is None and "<PitchSize" in line:
                m = RE_PITCH.search(line)
                pitch = (float(m.group(1)), float(m.group(2)))
    flush()
    return pitch, sets


def convert(pos_path):
    match_id = re.search(r"DFL-MAT-(\w+)\.xml", pos_path).group(1)
    info = glob.glob(os.path.join(SRC, "DFL_02_01_matchinformation_*{}.xml".format(match_id)))[0]
    role = roles(info)
    pitch, sets = parse_positions(pos_path)
    L, W = pitch

    # окно каждого тайма — по мячу
    sections = ["firstHalf", "secondHalf"]
    ball_sets = {s: v for (s, t, p), v in sets.items() if t == "BALL"}
    first = {s: int(ball_sets[s][0, 0]) for s in sections}
    length = {s: int(ball_sets[s][-1, 0]) - first[s] + 1 for s in sections}
    offset = {"firstHalf": 0, "secondHalf": length["firstHalf"] + GAP}
    n = offset["secondHalf"] + length["secondHalf"]

    people = sorted({(t, p) for (s, t, p) in sets if t not in ("BALL", "referee")})
    team_of = {t: (0 if role.get(t) == "home" else 1) for t, _ in people}
    if sorted(set(team_of.values())) != [0, 1]:
        raise ValueError("не нашла хозяев и гостей: {}".format(role))
    slot = {tp: i for i, tp in enumerate(people)}
    players = np.full((n, len(people), 3), np.nan)
    ball = np.full((n, 3), np.nan)
    gt = np.full(n, 3, dtype=np.int8)

    for (s, t, p), arr in sets.items():
        if t == "referee":
            continue
        idx = arr[:, 0].astype(np.int64) - first[s] + offset[s]
        keep = (idx >= offset[s]) & (idx < offset[s] + length[s])
        idx, arr = idx[keep], arr[keep]
        nx = (arr[:, 1] + L / 2) / L
        ny = (W / 2 - arr[:, 2]) / W
        if t == "BALL":
            ball[idx, 0], ball[idx, 1], ball[idx, 2] = nx, ny, arr[:, 3]
            poss, alive = arr[:, 4].astype(int), arr[:, 5].astype(int)
            gt[idx] = np.where(alive == 0, 2, np.where(poss == 1, 0, 1))
        else:
            k = slot[(t, p)]
            players[idx, k, 0], players[idx, k, 1] = nx, ny
    # скорость по позициям, м/с (в нашем матче третье поле игрока — скорость)
    xy = players[:, :, :2] * np.array([L, W])
    v = np.full(players.shape[:2], np.nan)
    v[1:] = np.sqrt(((xy[1:] - xy[:-1]) ** 2).sum(axis=2)) * FPS
    players[:, :, 2] = v

    slot_team = np.array([team_of[t] for t, _ in people], dtype=np.int8)
    os.makedirs(DST, exist_ok=True)
    out = os.path.join(DST, "idsse_{}.npz".format(match_id))
    np.savez_compressed(out, ball=ball, players=players, slot_team=slot_team, gt=gt,
                        pitch=np.array(pitch))
    graded = gt != 3
    share = np.bincount(gt[graded], minlength=3)[:3] / graded.sum()
    return "{}: кадров {}, людей {}, H/A/D {:.2f}/{:.2f}/{:.2f}".format(
        match_id, n, len(people), *share)


def main():
    paths = sorted(glob.glob(os.path.join(SRC, "DFL_04_03_positions_raw_observed_*.xml")))
    if len(sys.argv) > 1:
        paths = [p for p in paths if any(a in p for a in sys.argv[1:])]
    with Pool(min(4, len(paths))) as pool:
        for line in pool.imap_unordered(convert, paths):
            print(line, flush=True)


if __name__ == "__main__":
    main()
