"""Чтение автоматического трекинга (ATD, формат FIFA/EPTS) в массивы numpy с кэшем .npz.

Формат строки (DATA.md, starter/baseline.py организаторов):
    "<frame_id>:<игрок>;<игрок>;...;<игрок>:<мяч>"
    группа игрока "x,y,speed", мяч "x,y,..." ; пропуск = NaN ; frame_id с 1, кадр видео = frame_id - 1.
Метаданные XML: <Team id=..><Name>..</Name></Team>, <Player teamId=..> по порядку слотов.
Первая команда в файле — хозяева (H, Барселона).

ВАЖНО (правила, разъяснения организаторов от 04.11): данные матча нельзя отдавать облачным ИИ.
Этот модуль только читает файлы локально; наружу выводятся только сводные цифры, без строк.
"""

import hashlib
import os
import tempfile
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

from .config import asset, CACHE_DIR, PITCH

HOME, AWAY = 0, 1


def read_meta(meta_path):
    root = ET.parse(meta_path).getroot()
    teams = list(root.iter("Team"))
    if len(teams) != 2:
        raise ValueError("ожидалось 2 команды, найдено {}".format(len(teams)))
    team_ids = [t.get("id") for t in teams]
    if None in team_ids or len(set(team_ids)) != 2:
        raise ValueError("Invalid team identifiers")
    if any(p.get("teamId") not in team_ids for p in root.iter("Player")):
        raise ValueError("Player references an unknown team")
    home_id = teams[0].get("id")
    names = tuple(t.findtext("Name") for t in teams)
    slot_team = np.array([HOME if p.get("teamId") == home_id else AWAY
                          for p in root.iter("Player")], dtype=np.int8)
    if not len(slot_team):
        raise ValueError("в метаданных нет Player")
    return names, slot_team


def _f(s):
    try:
        value = float(s)
        return value if np.isfinite(value) else np.nan
    except ValueError:
        return np.nan


def parse_raw(raw_path, n_slots):
    """-> dict: ball (N,3) [x,y,третье поле], ball_nfields (N,), players (N,S,3) [x,y,speed], frame0.

    Потоковое чтение, ~1–2 мин на 199 тыс. строк. Значения в нормированных координатах 0..1.
    """
    with open(raw_path) as fh:
        n = sum(1 for _ in fh)
    ball = np.full((n, 3), np.nan, dtype=np.float64)
    ball_nf = np.zeros(n, dtype=np.int8)
    players = np.full((n, n_slots, 3), np.nan, dtype=np.float64)
    frame_ids = np.zeros(n, dtype=np.int64)
    with open(raw_path) as fh:
        for i, line in enumerate(fh):
            parts = line.rstrip("\n").split(":")
            if len(parts) < 3:
                continue
            frame_ids[i] = int(parts[0])
            groups = parts[1].split(";")
            for s, g in enumerate(groups[:n_slots]):
                if g and g[0] != "N":
                    v = g.split(",")
                    if len(v) < 2:
                        raise ValueError("Malformed player coordinates at line {}".format(i + 1))
                    players[i, s, 0] = _f(v[0])
                    players[i, s, 1] = _f(v[1])
                    if len(v) > 2:
                        players[i, s, 2] = _f(v[2])
            b = parts[2].split(",")
            ball_nf[i] = len(b)
            if b and b[0] and b[0][0] != "N":
                for k in range(min(3, len(b))):
                    ball[i, k] = _f(b[k])
    return {"ball": ball, "ball_nfields": ball_nf, "players": players, "frame_ids": frame_ids}


def _validate_arrays(d, n_slots):
    """Reject incomplete/corrupt cache contracts before inference or cache writes."""
    ids = d["frame_ids"]
    if not (ids.ndim == 1 and np.issubdtype(ids.dtype, np.integer) and len(ids)
            and np.array_equal(ids, np.arange(1, len(ids) + 1))):
        raise ValueError("frame_id не идут 1..N подряд — выравнивание не то, что в DATA.md")
    n = len(ids)
    for key, shape in (("ball", (n, 3)), ("players", (n, n_slots, 3))):
        arr = d[key]
        if arr.shape != shape or not np.issubdtype(arr.dtype, np.floating) or np.isinf(arr).any():
            raise ValueError("Invalid tracking array: " + key)
    nf = d["ball_nfields"]
    if nf.shape != (n,) or not np.issubdtype(nf.dtype, np.integer) or (nf < 1).any():
        raise ValueError("Invalid ball field counts")


def _write_cache(cache, d):
    """An interrupted process must not leave a partially written shared cache."""
    folder = os.path.dirname(cache)
    os.makedirs(folder, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".atd-", suffix=".npz", dir=folder)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez(stream, **d)
        os.replace(temporary, cache)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load(source="tactical", use_cache=True):
    """Трекинг, выровненный по кадрам видео (строка i = кадр i)."""
    meta_path, raw_path = asset("atd_meta", source), asset("atd_raw", source)
    for p in (meta_path, raw_path):
        if not os.path.isfile(p):
            raise IOError("нет файла данных: {}\nРаспакуй пакет организаторов в data/ "
                          "(или задай MTAH_DATA_DIR).".format(p))
    names, slot_team = read_meta(meta_path)
    # Cache identity includes source bytes and parser contract, not path/mtime.
    digest = hashlib.sha256(b"atd-parser-v2\0")
    for file in (meta_path, raw_path):
        with open(file, "rb") as stream:
            for block in iter(lambda: stream.read(1024*1024), b""):
                digest.update(block)
    key = digest.hexdigest()
    cache = os.path.join(CACHE_DIR, "atd_{}_{}.npz".format(source, key))
    d = None
    if use_cache and os.path.isfile(cache):
        try:
            with open(cache, "rb") as stream:
                with np.load(stream, allow_pickle=False) as archive:
                    d = dict(archive)
            _validate_arrays(d, len(slot_team))
        except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
            d = None  # Cache is disposable; the source remains authoritative.
    if d is None:
        d = parse_raw(raw_path, len(slot_team))
        _validate_arrays(d, len(slot_team))
        _write_cache(cache, d)
    d = dict(d)
    d["slot_team"] = slot_team
    d["team_names"] = names
    return d


def summary(d):
    """Сводка без сырых значений — её можно показывать."""
    n = len(d["frame_ids"])
    ball_seen = np.isfinite(d["ball"][:, :2]).all(axis=1)
    n_players = np.isfinite(d["players"][:, :, :2]).all(axis=2).sum(axis=1)
    nf = np.bincount(d["ball_nfields"], minlength=6)
    return {
        "кадров": n,
        "слотов": int(d["players"].shape[1]),
        "мяч виден, %": round(100.0 * ball_seen.mean(), 1),
        "игроков в кадре, среднее": round(float(n_players.mean()), 1),
        "полей у мяча (сколько строк с 1..5 полями)": nf[1:6].tolist(),
        "команды": d["team_names"],
    }


def to_metres(xy):
    return xy * np.array(PITCH)
