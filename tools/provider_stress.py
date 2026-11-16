"""Export fixed development games and observed-only provider stress copies.

Run: python tools/provider_stress.py

Rows retain the readers' 25 Hz grid and reference arrays. Each simulated sensor
tick selects the nearest source tick (ties forward), once only, within a period.
The effective rate is capped at the source rate. Real source times, not rounded
grid times, determine speed. Samples map to the nearest 25 Hz row (ties forward);
all other rows remain NaN. Existing grid gaps are also retained conservatively.
No interpolation, holding, clipping or temporal noise correlation is applied.
SkillCorner baselines retain reader interpolation for compatibility; stress
copies use only raw is_detected observations, never that interpolation.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
KEYS = ("players", "ball", "slot_team", "pitch", "period", "gt")
GRID_HZ = 25


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def spans(period):
    edges = np.r_[0, np.flatnonzero(np.diff(period)) + 1, len(period)]
    return [(int(a), int(b), int(period[a]))
            for a, b in zip(edges[:-1], edges[1:]) if period[a] > 0]


def load_idsse(source, external, output):
    reader = read_module("ext_idsse")
    position = external / source["raw_files"]["tracking"]["path"]
    reader.SRC = str(position.parent)
    # Use the unchanged reader for the exact comparison export, in a local temp.
    with tempfile.TemporaryDirectory(prefix="reader-", dir=output) as temporary:
        reader.DST = temporary
        reader.convert(str(position))
        with np.load(Path(temporary) / f"idsse_{source['game_id']}.npz") as archive:
            original = {key: archive[key] for key in archive.files}
    bounds = {}
    section = None
    # Only frame clocks and FrameSet period identifiers determine boundaries.
    with position.open() as stream:
        for line in stream:
            if "<FrameSet" in line:
                match = reader.RE_SET.search(line)
                section = match[1] if match[2] == "BALL" else None
            elif section and "<Frame " in line:
                frame = int(re.search(r'N="(\d+)"', line)[1])
                if section not in bounds:
                    bounds[section] = [frame, frame]
                bounds[section][1] = frame
    period = np.zeros(len(original["gt"]), np.int8)
    source_frame = np.full(len(period), -1, np.int64)
    offset = 0
    for number, section in enumerate(("firstHalf", "secondHalf"), 1):
        first, last = bounds[section]
        size = last - first + 1
        period[offset:offset + size] = number
        source_frame[offset:offset + size] = np.arange(first, last + 1)
        offset += size + (reader.GAP if number == 1 else 0)
    if offset != len(period):
        raise ValueError("DFL frame bounds disagree with reader grid")
    original["period"] = period
    native = {key: original[key] for key in ("players", "ball", "period")}
    native.update(timestamps=np.arange(len(period)) / 25, source_frame=source_frame)
    return original, native


def load_skillcorner(source, external, output):
    del output
    folder = (external / source["raw_files"]["metadata"]["path"]).parent
    original = read_module("skillcorner_reader").read_match(folder, source["game_id"])
    meta = json.loads((external / source["raw_files"]["metadata"]["path"]).read_text())
    n = max(int(row["end_frame"]) for row in meta["match_periods"]) + 1
    period = np.zeros(n, np.int8)
    for row in meta["match_periods"]:
        period[int(row["start_frame"]):int(row["end_frame"])] = int(row["period"])
    players = np.full((n, len(meta["players"]), 3), np.nan, np.float32)
    ball = np.full((n, 3), np.nan, np.float32)
    slots = {int(player["id"]): i for i, player in enumerate(meta["players"])}
    with (external / source["raw_files"]["tracking"]["path"]).open() as stream:
        for line in stream:
            row = json.loads(line)
            frame = int(row["frame"])
            if not 0 <= frame < n:
                continue
            detection = row["ball_data"]
            if detection.get("is_detected"):
                ball[frame] = [detection.get(k) if detection.get(k) is not None else np.nan
                               for k in ("x", "y", "z")]
            for player in row["player_data"]:
                if player.get("is_detected"):
                    players[frame, slots[int(player["player_id"])], :2] = [player["x"], player["y"]]
    for values in (players, ball):
        values[..., :2] = values[..., :2] / original["pitch"] + .5
    return original, dict(players=players, ball=ball, period=period,
                          timestamps=np.arange(n) / 10,
                          source_frame=np.arange(n, dtype=np.int64))


def corrupt_geometry(native, grid_players, grid_ball, grid_period, pitch,
                     source_hz, preset, seed):
    """Geometry-only interface: no reference, event or coverage arguments."""
    rate = min(float(preset["observation_hz"]), source_hz)
    noise = float(preset["noise_std_m"])
    burst_rate = float(preset["ball_bursts_per_minute"])
    low, high = float(preset["burst_min_s"]), float(preset["burst_max_s"])
    if not (0 < rate <= GRID_HZ and noise >= 0 and burst_rate >= 0 and 0 < low <= high):
        raise ValueError("Invalid observation, noise or burst parameters")
    times = native["timestamps"]
    if source_hz != int(source_hz) or source_hz <= 0 or times[0] != 0:
        raise ValueError("Source clock must start at zero with an integer sampling rate")
    if len(times) != len(native["period"]) or np.any(np.diff(times) <= 0):
        raise ValueError("Native clock must be strictly increasing")
    if not np.allclose(np.diff(times), 1 / source_hz, rtol=0, atol=1e-9):
        raise ValueError("Native clock disagrees with declared sampling rate")
    players = np.full_like(grid_players, np.nan)
    ball = np.full_like(grid_ball, np.nan)
    observation_mask = np.zeros(len(grid_period), bool)
    dropout_mask = np.zeros(len(grid_period), bool)
    indices, mappings, bursts = [], [], []
    before_bursts = 0
    dropped = 0
    # Prefix counts also detect gaps hidden between two selected coarse ticks.
    missing = ~np.isfinite(native["players"][..., :2]).all(axis=2)
    missing_prefix = np.vstack([np.zeros((1, missing.shape[1]), np.int64),
                                np.cumsum(missing, axis=0)])
    for segment, (start, stop, period_id) in enumerate(spans(native["period"])):
        noise_seed, burst_seed = np.random.SeedSequence([seed, period_id, segment]).spawn(2)
        noise_rng = np.random.default_rng(noise_seed)
        burst_rng = np.random.default_rng(burst_seed)
        count = int(np.floor((times[stop - 1] - times[start]) * rate + 1e-8)) + 1
        source_index = start + np.floor(np.arange(count) * source_hz / rate + .5).astype(np.int64)
        source_index = np.unique(source_index[source_index < stop])
        # Integer half-up rounding avoids binary ties such as 2.3 * 25 < 57.5.
        grid_index = ((2 * source_index * GRID_HZ + int(source_hz)) //
                      (2 * int(source_hz))).astype(np.int64)
        valid = (grid_index >= 0) & (grid_index < len(grid_period))
        source_index, grid_index = source_index[valid], grid_index[valid]
        valid = grid_period[grid_index] == period_id
        source_index, grid_index = source_index[valid], grid_index[valid]
        if len(grid_index) == 0:
            continue
        if len(np.unique(grid_index)) != len(grid_index):
            raise ValueError("More than one observation mapped to a grid row")
        xy = native["players"][source_index, :, :2].copy()
        b = native["ball"][source_index].copy()
        # Reject partial/missing xy and retain every pre-existing canonical gap.
        player_ok = (np.isfinite(xy).all(axis=2) &
                     np.isfinite(grid_players[grid_index, :, :2]).all(axis=2))
        ball_ok = (np.isfinite(b[:, :2]).all(axis=1) &
                   np.isfinite(grid_ball[grid_index, :2]).all(axis=1))
        xy[~player_ok] = np.nan
        b[~ball_ok] = np.nan
        xy += noise_rng.normal(0, noise, xy.shape) / pitch
        b[:, :2] += noise_rng.normal(0, noise, b[:, :2].shape) / pitch
        speed = np.full(xy.shape[:2], np.nan, players.dtype)
        if len(xy) > 1:
            dt = np.diff(times[source_index])
            velocity = np.linalg.norm(np.diff(xy, axis=0) * pitch, axis=2) / dt[:, None]
            continuous = (missing_prefix[source_index[1:] + 1] -
                          missing_prefix[source_index[:-1]]) == 0
            speed[1:] = np.where(continuous, velocity, np.nan)
        drop = np.zeros(len(source_index), bool)
        end = times[stop - 1] + 1 / source_hz
        if burst_rate:
            at = times[start] + burst_rng.exponential(60 / burst_rate)
            while at < end:
                until = min(at + burst_rng.uniform(low, high), end)
                bursts.append(dict(period=period_id, start_s=float(at), end_s=float(until)))
                drop |= (times[source_index] >= at) & (times[source_index] < until)
                at += burst_rng.exponential(60 / burst_rate)
        before_bursts += int(ball_ok.sum())
        dropped += int((drop & ball_ok).sum())
        b[drop] = np.nan
        players[grid_index, :, :2] = xy
        players[grid_index, :, 2] = speed
        ball[grid_index] = b
        observation_mask[grid_index] = True
        dropout_mask[grid_index] = drop
        indices.append(source_index)
        mappings.append(grid_index)
    source_index = np.concatenate(indices) if indices else np.empty(0, np.int64)
    grid_index = np.concatenate(mappings) if mappings else np.empty(0, np.int64)
    result = dict(players=players, ball=ball, observation_mask=observation_mask,
                  ball_dropout_mask=dropout_mask, source_index=source_index,
                  source_frame=native["source_frame"][source_index].copy(),
                  grid_index=grid_index, native_timestamps_s=times[source_index].copy())
    report = dict(requested_observation_hz=preset["observation_hz"],
                  effective_observation_hz=rate, source_hz=source_hz,
                  bursts=bursts, burst_count=len(bursts),
                  observed_ball_before_bursts=before_bursts,
                  dropped_ball_observations=dropped,
                  dropout_grid_rows=int(dropout_mask.sum()))
    return result, report


def make_copy(original, native, source_hz, preset, seed):
    result, report = corrupt_geometry(
        native, original["players"], original["ball"], original["period"],
        original["pitch"], source_hz, preset, seed)
    # Reference is attached after all decisions and random draws have finished.
    for key in ("slot_team", "pitch", "period", "gt"):
        result[key] = original[key].copy()
    return result, report


def verify(original, result, report, native, baseline=False):
    for key in KEYS:
        if result[key].shape != original[key].shape or result[key].dtype != original[key].dtype:
            raise ValueError(f"Schema mismatch: {key}")
    for key in ("gt", "period", "slot_team", "pitch"):
        np.testing.assert_array_equal(result[key], original[key])
    if baseline:
        for key in KEYS:
            np.testing.assert_array_equal(result[key], original[key])
    else:
        unobserved = ~result["observation_mask"]
        assert np.isnan(result["ball"][unobserved]).all()
        assert np.isnan(result["players"][unobserved]).all()
        assert np.isnan(result["ball"][result["ball_dropout_mask"]]).all()
        grid, source = result["grid_index"], result["source_index"]
        np.testing.assert_array_equal(np.flatnonzero(result["observation_mask"]), grid)
        np.testing.assert_array_equal(result["native_timestamps_s"], native["timestamps"][source])
        np.testing.assert_array_equal(result["source_frame"], native["source_frame"][source])
        np.testing.assert_array_equal(result["period"][grid], native["period"][source])
        assert not result["ball_dropout_mask"][unobserved].any()
        ball_ok = (np.isfinite(native["ball"][source, :2]).all(axis=1) &
                   np.isfinite(original["ball"][grid, :2]).all(axis=1))
        drop = result["ball_dropout_mask"][grid]
        assert report["dropped_ball_observations"] == int((ball_ok & drop).sum())
        assert report["observed_ball_before_bursts"] == int(ball_ok.sum())
        assert report["dropout_grid_rows"] == int(drop.sum())
        assert report["burst_count"] == len(report["bursts"])
        np.testing.assert_array_equal(np.isfinite(result["ball"][grid, :2]).all(axis=1),
                                      ball_ok & ~drop)
        for key in ("players", "ball"):
            missing = ~np.isfinite(native[key][source, ..., :2]).all(axis=-1)
            assert np.isnan(result[key][grid][missing]).all()
            missing = ~np.isfinite(original[key][..., :2]).all(axis=-1)
            assert np.isnan(result[key][missing]).all()
        for burst in report["bursts"]:
            assert any(p == burst["period"] and
                       native["timestamps"][a] <= burst["start_s"] < burst["end_s"] <=
                       native["timestamps"][b - 1] + 1 / report["source_hz"]
                       for a, b, p in spans(native["period"]))
    baseline_missing = ~np.isfinite(original["ball"][:, :2]).all(axis=1)
    output_missing = ~np.isfinite(result["ball"][:, :2]).all(axis=1)
    assert output_missing.sum() >= baseline_missing.sum()
    return dict(schema=True, unchanged_reference=True, unchanged_period=True,
                baseline_missing_ball_fraction=float(baseline_missing.mean()),
                missing_ball_rows=int((~np.isfinite(result["ball"][:, :2]).all(axis=1)).sum()),
                missing_ball_fraction=float((~np.isfinite(result["ball"][:, :2]).all(axis=1)).mean()))


def require_ignored(path):
    if subprocess.run(["git", "-C", str(ROOT), "check-ignore", "-q", str(path)],
                      capture_output=True).returncode:
        raise ValueError(f"Output must be inside an ignored directory: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "tools/provider_stress_presets.json")
    parser.add_argument("--external-root", type=Path, default=ROOT / "external")
    parser.add_argument("--output-root", type=Path, default=ROOT / "out/provider_stress")
    parser.add_argument("--source", nargs="+")
    parser.add_argument("--preset", nargs="+")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config["grid_hz"] != GRID_HZ:
        raise ValueError("Only the canonical 25 Hz grid is supported")
    plan_path = ROOT / "docs/experiments/temporal/plan.json"
    if digest(plan_path) != config["fold_plan_sha256"]:
        raise ValueError("Frozen fold plan hash mismatch")
    plan = json.loads(plan_path.read_text())
    development = json.loads((ROOT / "mtah/expanded_team_training.json").read_text())
    source_names = args.source or list(config["sources"])
    preset_names = args.preset or ["baseline", *config["presets"]]
    if set(source_names) - config["sources"].keys():
        raise ValueError("Unknown source")
    if set(preset_names) - ({"baseline"} | config["presets"].keys()):
        raise ValueError("Unknown preset")
    require_ignored(args.output_root / "manifest.json")
    args.output_root.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    provenance = dict(git_commit=commit, converter_sha256=digest(__file__),
                      config_sha256=digest(args.config),
                      python_version=platform.python_version(), numpy_version=np.__version__,
                      reader_sha256={name: digest(ROOT / "tools" / f"{name}.py")
                                     for name in ("ext_idsse", "skillcorner_reader")},
                      revision_note="git_commit is checkout HEAD; converter_sha256 identifies the uncommitted converter used for this export.",
                      working_tree_dirty=bool(subprocess.check_output(
                          ["git", "-C", str(ROOT), "status", "--porcelain"], text=True)))
    for name in source_names:
        source = config["sources"][name]
        family, game = source["fold_key"].split("/")
        if game not in development[family] or plan["folds"][source["fold_key"]] != source["fold"]:
            raise ValueError("Source must belong to the pinned development fold")
        for item in source["raw_files"].values():
            if digest(args.external_root / item["path"]) != item["sha256"]:
                raise ValueError(f"Raw input hash mismatch: {item['path']}")
        if shutil.disk_usage(args.output_root).free < 20 * 1024**3:
            raise ValueError("At least 20 GiB free disk space is required")
        loader = load_idsse if source["provider"] == "idsse" else load_skillcorner
        original, native = loader(source, args.external_root, args.output_root)
        source_periods = []
        for start, stop, period_id in spans(native["period"]):
            grid_rows = np.flatnonzero(original["period"] == period_id)
            source_periods.append(dict(
                period=period_id, native_start_index=start, native_stop_index=stop,
                source_first_frame=int(native["source_frame"][start]),
                source_last_frame=int(native["source_frame"][stop - 1]),
                start_s=float(native["timestamps"][start]),
                end_s=float(native["timestamps"][stop - 1] + 1 / source["source_hz"]),
                grid_start=int(grid_rows[0]), grid_stop=int(grid_rows[-1]) + 1))
        for preset_name in preset_names:
            baseline = preset_name == "baseline"
            if baseline:
                result = {k: original[k].copy() for k in KEYS}
                report = dict(requested_observation_hz=source["source_hz"],
                              effective_observation_hz=source["source_hz"],
                              source_hz=source["source_hz"], bursts=[], burst_count=0,
                              dropped_ball_observations=0, dropout_grid_rows=0)
            else:
                result, report = make_copy(original, native, source["source_hz"],
                                           config["presets"][preset_name], config["seed"])
            checks = verify(original, result, report, native, baseline)
            folder = args.output_root / name
            folder.mkdir(exist_ok=True)
            target = folder / f"{preset_name}.npz"
            manifest_path = target.with_suffix(".json")
            require_ignored(target)
            require_ignored(manifest_path)
            np.savez_compressed(target, **result)
            with np.load(target) as archive:
                for key in result:
                    np.testing.assert_array_equal(archive[key], result[key])
            manifest = dict(source=source, preset=preset_name,
                            parameters=None if baseline else config["presets"][preset_name],
                            seed=config["seed"], fold=source["fold"],
                            fold_policy="All copies remain in the original game's fold; use its excluded-fold model.",
                            provenance=provenance, grid_hz=GRID_HZ,
                            timing=config["baseline_timing"] if baseline else config["stress_timing"],
                            source_clock=source["clock_description"],
                            source_periods=source_periods,
                            shapes={k: list(v.shape) for k, v in result.items()},
                            dtypes={k: str(v.dtype) for k, v in result.items()},
                            statistics=report, checks=checks, npz_sha256=digest(target))
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            print(f"{name}/{preset_name}: {checks['missing_ball_fraction']:.6f} missing ball, "
                  f"{report['dropped_ball_observations']} observations dropped", flush=True)


if __name__ == "__main__":
    main()
