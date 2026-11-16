"""Local aggregate-only roughness diagnosis; never reads reference labels.

Second differences confound true acceleration, camera motion and track changes.
They do not identify true measurement error or justify a model switch threshold.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import provider_stress as stress


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def roughness(values, period, pitch):
    xy = values[..., :2] * pitch
    delta = xy[2:] - 2 * xy[1:-1] + xy[:-2]
    valid = (np.isfinite(xy[2:]).all(axis=-1) & np.isfinite(xy[1:-1]).all(axis=-1)
             & np.isfinite(xy[:-2]).all(axis=-1))
    within = (period[2:] > 0) & (period[2:] == period[1:-1]) & (period[2:] == period[:-2])
    valid &= within.reshape((-1,) + (1,) * (valid.ndim - 1))
    rows = delta[valid]
    if len(rows) == 0:
        raise ValueError('No finite consecutive triples')
    center = np.median(rows, axis=0)
    mad = np.median(abs(rows - center), axis=0)
    return {'finite_triples': len(rows),
            'iid_equivalent_roughness_m_xy': (mad / (.67448975 * np.sqrt(6))).tolist(),
            'absolute_second_difference_norm_m_quantiles': dict(zip(
                ['q50', 'q90', 'q95', 'q99'],
                np.quantile(np.linalg.norm(rows, axis=-1), [.5, .9, .95, .99]).tolist()))}


def summarize(data):
    return {kind: roughness(data[kind], data['period'], data['pitch'])
            for kind in ['players', 'ball']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, default=ROOT/'reports/tracking_roughness/results.json')
    a = p.parse_args()
    if a.output.exists():
        p.error('Do not overwrite completed diagnosis')
    os.environ['MTAH_DATA_DIR'] = str(a.data_dir.resolve())
    from mtah import atd, config
    target = atd.load()
    target['pitch'] = np.asarray(config.PITCH)
    target['period'] = np.zeros(len(target['ball']), np.int8)
    for number, first, last in config.HALVES:
        target['period'][first:last+1] = number
    result = {'purpose': 'Unsupervised geometry-only roughness; no quality score or model switch',
              'plan_sha256': sha(ROOT/'docs/experiments/provider-stress/roughness_plan.json'),
              'runner_sha256': sha(__file__), 'target': summarize(target),
              'true_noise_identified': False, 'model_changed': False, 'target_labels_read': False}
    del target
    cfg = json.loads((ROOT/'tools/provider_stress_presets.json').read_text())
    source = cfg['sources']['idsse_J03WMX']
    for item in source['raw_files'].values():
        if sha(ROOT/'external'/item['path']) != item['sha256']:
            raise ValueError('DFL source changed')
    # Open only geometry from the existing DFL conversion, never its gt member.
    with np.load(ROOT/'external/conv/idsse_J03WMX.npz', allow_pickle=False) as z:
        data = {k: z[k] for k in ['players', 'ball', 'pitch']}
    periods = json.loads((ROOT/'reports/temporal/sequences/original__idsse_J03WMX/matched_seed0/meta.json').read_text())['full_segments']
    data['period'] = np.zeros(len(data['ball']), np.int8)
    for number,(first,last) in enumerate(periods,1):data['period'][first:last] = number
    result['DFL_clean'] = summarize(data)
    native = {k:data[k] for k in ['players','ball','period']}
    native.update(timestamps=np.arange(len(data['ball']))/25,
                  source_frame=np.arange(len(data['ball']),dtype=np.int64))
    for noise in [.5,2.]:
        preset=dict(observation_hz=25,noise_std_m=noise,ball_bursts_per_minute=0,burst_min_s=2,burst_max_s=6)
        observed,_=stress.corrupt_geometry(native,data['players'],data['ball'],data['period'],data['pitch'],25,preset,cfg['seed'])
        observed.update(period=data['period'],pitch=data['pitch'])
        result['DFL_noise_'+str(noise)] = summarize(observed)
    result['input_hashes'] = {ext:sha(a.data_dir/'tactical'/('BAR-ATH_tactical_atd.'+ext)) for ext in ['xml','txt']}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k in ['target','DFL_clean','DFL_noise_0.5','DFL_noise_2.0']},indent=2))


if __name__=='__main__':
    main()
