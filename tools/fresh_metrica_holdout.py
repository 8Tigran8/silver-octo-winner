"""Evaluate frozen nine-game weights on public Sample_Game_3; never train on it."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cloudpickle
import numpy as np
from threadpoolctl import threadpool_limits
from mtah import degrade, features2, stack
from mtah.methods import baseline, boost, boost_train
from quality_audit import SCENARIOS, official, argmax_labels, measure
_spec = importlib.util.spec_from_file_location("metrica_converter", ROOT / "tools/ext_metrica.py")
converter = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(converter)


def parse_tracking(metadata, tracking):
    root = ET.parse(metadata).getroot()
    fps = float(root.findtext(".//FrameRate"))
    if fps != 25:
        raise ValueError("Unexpected sample frame rate")
    field = root.find(".//FieldSize")
    pitch = (float(field.findtext("Width")), float(field.findtext("Height")))
    teams = list(root.iter("Team")); players_meta = list(root.iter("Player"))
    if len(teams) != 2:
        raise ValueError("Expected exactly two sample teams")
    team_map = {node.get("id"): i for i, node in enumerate(teams)}
    player_map = {node.get("id"): i for i, node in enumerate(players_meta)}
    slot_team = np.array([team_map[node.get("teamId")] for node in players_meta], dtype=np.int8)
    channels = {node.get("id"): (player_map[node.get("playerId")], {"x": 0, "y": 1}[node.get("channelId")])
                for node in root.iter("PlayerChannel")}
    specs = []
    for node in root.iter("DataFormatSpecification"):
        groups = list(node)[1]
        mapping = [[channels[ref.get("playerChannelId")] for ref in group] for group in groups]
        specs.append((int(node.get("startFrame")), int(node.get("endFrame")), mapping))
    specs.sort()
    with Path(tracking).open() as stream:
        n = sum(1 for _ in stream)
    players = np.full((n, len(player_map), 3), np.nan)
    ball = np.full((n, 3), np.nan)
    cursor = 0
    with Path(tracking).open() as stream:
        for index, line in enumerate(stream):
            parts = line.rstrip().split(":")
            if len(parts) != 3 or int(parts[0]) != index + 1:
                raise ValueError("Sample tracking frame IDs must be 1..N")
            frame = index + 1
            while cursor < len(specs) and frame > specs[cursor][1]:
                cursor += 1
            if cursor >= len(specs) or frame < specs[cursor][0]:
                raise ValueError("No metadata lineup mapping for sample frame")
            mapping = specs[cursor][2]
            groups = parts[1].split(";")
            if len(groups) != len(mapping):
                raise ValueError("Tracking groups do not match current lineup metadata")
            for group, refs in zip(groups, mapping):
                values = group.split(",")
                if len(values) != len(refs):
                    raise ValueError("Unexpected coordinate field count")
                for value, (player, channel) in zip(values, refs):
                    players[index, player, channel] = float(value) if value else np.nan
            values = parts[2].split(",")
            for channel, value in enumerate(values[:2]):
                ball[index, channel] = float(value) if value else np.nan
    xy = players[:, :, :2] * np.array(pitch)
    players[1:, :, 2] = np.sqrt(((xy[1:] - xy[:-1]) ** 2).sum(axis=2)) * fps
    return {"ball": ball, "players": players, "slot_team": slot_team, "pitch": pitch}, team_map, len(specs)


def subtype_text(value):
    if value is None:
        return ""
    if isinstance(value, list):
        return "-".join(subtype_text(v) for v in value)
    if isinstance(value, dict):
        return str(value.get("name", ""))
    return str(value)


def event_labels(path, n, team_map):
    records = json.loads(Path(path).read_text())["data"]
    events = [{"team": team_map[r["team"]["id"]], "type": r["type"]["name"].upper(),
               "sub": subtype_text(r.get("subtypes")).upper(), "period": int(r["period"]),
               "start": int(r["start"]["frame"]), "end": int(r["end"]["frame"])} for r in records]
    frames = np.arange(1, n + 1)
    period = np.zeros(n, dtype=int)
    # No period column exists in this export. Score only the annotated event span of each period.
    for p in (1, 2):
        selected = [e for e in events if e["period"] == p]
        lo, hi = min(e["start"] for e in selected), max(e["end"] for e in selected)
        if not (1 <= lo <= hi <= n):
            raise ValueError("Event spans are outside tracking timeline")
        period[(frames >= lo) & (frames <= hi)] = p
    return converter.labels_from_events(events, frames, period), len(events)


def main():
    started = time.perf_counter()
    folder = ROOT / "reports/fresh_holdout"
    protocol = json.loads((folder / "protocol.json").read_text())
    if "metrica_3" in protocol["model_training_games"]:
        raise ValueError("Held-out sample was part of model training")
    for item in protocol["files"]:
        if hashlib.sha256((ROOT / item["path"]).read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("Held-out source integrity mismatch")
    checkpoint = ROOT / protocol["frozen_model"]
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != protocol["model_sha256"]:
        raise ValueError("Frozen own checkpoint integrity mismatch")
    with checkpoint.open("rb") as stream:
        models = cloudpickle.load(stream)
    source = ROOT / "external/holdout_metrica3"
    clean, teams, mappings = parse_tracking(source / "Sample_Game_3_metadata.xml", source / "Sample_Game_3_tracking.txt")
    clean["gt"], event_count = event_labels(source / "Sample_Game_3_events.json", len(clean["ball"]), teams)
    np.savez(folder / "converted_holdout.npz", **clean)
    params = boost_train.decoder_params()
    records = []
    with threadpool_limits(1):
        for scenario in protocol["scenarios"]:
            cfg = SCENARIOS[scenario]
            d = clean if cfg["seed"] is None else degrade.degrade(clean, seed=cfg["seed"], params=cfg["params"])
            gt, seen = d["gt"], np.isfinite(d["ball"][:, 0])
            segs = official.valid_runs(gt)
            x, _ = features2.build(d, segs); xs, _ = features2.build(d, segs, swap=True)
            pd, ph = boost.predict(models["stage1"], x, xs)
            qd, qh = stack.predict(models["stage2"], pd, ph)
            candidates = {"baseline_smooth50": official.label_codes(baseline.run(d, window=50)),
                          "stage1_argmax": argmax_labels(pd, ph, gt),
                          "stage1_viterbi": boost.label(pd, ph, segs, len(gt), params),
                          "frozen_stack_argmax": argmax_labels(qd, qh, gt),
                          "frozen_stack_viterbi": boost.label(qd, qh, segs, len(gt), params)}
            for name, labels in candidates.items():
                records.append({"match": "metrica_3", "scenario": scenario, "method": name,
                                "ball_visible_fraction": float(seen[gt != 3].mean()),
                                **measure(gt, labels, seen, 1000 if name in {"baseline_smooth50", "frozen_stack_viterbi"} else 0)})
    result = {"status": "Completed frozen-weight additional-game evaluation", "model_sha256": protocol["model_sha256"],
              "training_games": protocol["model_training_games"], "test_game": "Metrica Sample_Game_3",
              "frames": len(clean["gt"]), "graded_frames": int((clean["gt"] != 3).sum()),
              "provider_events": event_count, "lineup_mappings": mappings,
              "labels": "Event-derived proxy; only event-covered period spans graded",
              "training_or_retuning_on_holdout": False, "seconds": round(time.perf_counter()-started, 3),
              "records": records}
    (folder / "results.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"status": result["status"], "seconds": result["seconds"],
                      "results": [{k:r[k] for k in ("scenario", "method", "spell_f1_mean", "macro_f1")} for r in records]},indent=2),flush=True)


if __name__ == "__main__":
    main()
