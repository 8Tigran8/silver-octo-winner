"""Read public SkillCorner tracking, keeping label fields out of model inputs.

Phases are automatic reference proxies, not corrected competition ground truth.
We score full metadata periods: phase intervals are H/A, intervening gaps D.
No player/ball extrapolation or possession field enters the predictor.
"""
import csv
import json
from pathlib import Path

import numpy as np


def upsample_observations(values, positions):
    """Linear 10->25 Hz interpolation only between adjacent finite observations."""
    left = np.floor(positions).astype(int)
    right = np.minimum(left + 1, len(values) - 1)
    fraction = positions - left
    shape = (len(fraction),) + (1,) * (values.ndim - 1)
    weight = fraction.reshape(shape)
    a, b = values[left], values[right]
    result = a * (1 - weight) + b * weight
    return np.where(weight == 0, a, result).astype(np.float32)


def phase_labels(meta, phases, n):
    gt = np.full(n, 3, np.int8)
    period = np.zeros(n, np.int8)
    for row in meta["match_periods"]:
        a, b = int(row["start_frame"]), int(row["end_frame"])
        if not 0 <= a < b <= n:
            raise ValueError("Invalid metadata period span")
        gt[a:b], period[a:b] = 2, int(row["period"])
    home, away = meta["home_team"]["id"], meta["away_team"]["id"]
    for row in phases:
        a, b = int(row["frame_start"]), int(row["frame_end"])
        team = int(row["team_in_possession_id"])
        if team not in (home, away) or a >= b:
            raise ValueError("Invalid phase team or interval")
        a, b = max(0,a), min(n,b)
        if a >= b:
            continue
        # Provider phase endpoints sometimes exceed metadata period endpoints.
        # Intersect with the independently supplied period; never borrow labels
        # from the next half or extend its scored span from a possession phase.
        inside = period[a:b] == int(row["period"])
        label = 0 if team == home else 1
        occupied = gt[a:b] < 2
        if np.any(inside & occupied & (gt[a:b] != label)):
            raise ValueError("Conflicting possession phases")
        gt[a:b] = np.where(inside, label, gt[a:b])
    return gt, period


def read_match(folder, match_id):
    folder = Path(folder)
    meta = json.loads((folder / f"{match_id}_match.json").read_text())
    pitch = (float(meta["pitch_length"]), float(meta["pitch_width"]))
    n = max(int(row["end_frame"]) for row in meta["match_periods"]) + 1
    players = meta["players"]
    ids = {int(p["id"]): i for i, p in enumerate(players)}
    home, away = meta["home_team"]["id"], meta["away_team"]["id"]
    if any(p["team_id"] not in (home, away) for p in players):
        raise ValueError("Player belongs to an unknown team")
    teams = np.array([0 if p["team_id"] == home else 1 for p in players], np.int8)
    ball = np.full((n, 3), np.nan, np.float32)
    positions = np.full((n, len(ids), 3), np.nan, np.float32)
    last = -1
    with (folder / f"{match_id}_tracking_extrapolated.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            frame = int(row["frame"])
            if frame <= last:
                raise ValueError("Tracking frame IDs are not increasing")
            last = frame
            if frame >= n:
                continue
            b = row["ball_data"]
            if b.get("is_detected"):
                ball[frame] = [b.get(k) if b.get(k) is not None else np.nan for k in ("x", "y", "z")]
            for player in row["player_data"]:
                if player.get("is_detected"):
                    index = ids[int(player["player_id"])]
                    positions[frame, index, :2] = [player["x"], player["y"]]
    # Source coordinates: metres about pitch centre. Preserve missing detections.
    for values in (ball, positions):
        values[..., :2] = values[..., :2] / np.asarray(pitch) + 0.5
    delta = np.diff(positions[..., :2], axis=0) * np.asarray(pitch)
    positions[1:, :, 2] = np.sqrt(np.square(delta).sum(axis=2)) * 10
    with (folder / f"{match_id}_phases_of_play.csv").open() as stream:
        gt10, period10 = phase_labels(meta, list(csv.DictReader(stream)), n)
    times = np.arange(int((n - 1) * 2.5) + 1) / 2.5
    gt = gt10[np.floor(times).astype(int)]
    period = period10[np.floor(times).astype(int)]
    return {"ball": upsample_observations(ball, times),
            "players": upsample_observations(positions, times),
            "slot_team": teams, "pitch": np.asarray(pitch), "gt": gt, "period": period}
