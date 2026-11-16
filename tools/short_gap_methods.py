"""Development-only recovery policies; no target, reference or fixture identity inputs."""
import warnings
import numpy as np
from mtah import features2


def activity_features(data, segments):
    """Team/slot/reflection invariant player activity, period-local at 25 Hz.

    The binary gap expert sees no ball coordinates, missing-ball duration,
    current-model outputs, timestamps, identities or reference-derived features.
    """
    if 'gt' in data:
        raise ValueError('Labels are not features')
    out = None
    for a, b in segments:
        xy = data['players'][a:b, :, :2].astype(float) * np.asarray(data['pitch'])
        good = np.isfinite(xy).all(axis=2)
        xy[~good] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            cx = np.nanmean(xy, axis=1)
            delta = xy - cx[:, None, :]
            values = {'visible': good.sum(axis=1).astype(float),
                      'spread_x': np.nanstd(xy[:, :, 0], axis=1),
                      'spread_y': np.nanstd(xy[:, :, 1], axis=1),
                      'radius': np.nanmean(np.linalg.norm(delta, axis=2), axis=1)}
            for lag in [10, 25]:
                v = np.full_like(xy, np.nan)
                v[lag:] = (xy[lag:] - xy[:-lag]) * (25 / lag)
                speed = np.linalg.norm(v, axis=2)
                speed[speed > 12] = np.nan
                count = np.isfinite(speed).sum(axis=1)
                values[f'valid_speed_{lag}'] = count.astype(float)
                values[f'mean_speed_{lag}'] = np.nanmean(speed, axis=1)
                values[f'q75_speed_{lag}'] = np.nanquantile(speed, .75, axis=1)
                for threshold in [1, 3, 6]:
                    values[f'fraction_{threshold}_{lag}'] = np.where(count > 0,
                        (speed > threshold).sum(axis=1) / np.maximum(count, 1), np.nan)
            for key, value in list(values.items()):
                for window in [13, 51, 151]:
                    values[f'{key}_w{window}'] = features2._smooth(value, window)
        names = sorted(values)
        x = np.column_stack([values[k] for k in names]).astype(np.float32)
        if out is None:
            out = np.full((len(data['ball']), len(names)), np.nan, np.float32)
        out[a:b] = x
    return out, names


def bridge_dead(pd, ball, segments):
    """Reduce D evidence only in bounded gaps with live, interior endpoints."""
    result = pd.copy()
    for start, end in segments:
        seen = np.isfinite(ball[start:end, :2]).all(axis=1)
        for aa, bb in zip(*features2._runs(~seen)):
            a, b = start + aa, start + bb
            if a == start or b == end or b-a > 250:
                continue
            endpoints = ball[[a-1, b], :2]
            if not ((endpoints >= .03) & (endpoints <= .97)).all() or max(pd[a-1], pd[b]) > .2:
                continue
            p = np.clip(pd[a:b], 1e-6, 1-1e-6)
            result[a:b] = 1 / (1 + np.exp(-(np.log(p)-np.log1p(-p)-1.0)))
    return result


def policy(name, cache, decoder, expert=None):
    pd, ph = cache['pd'].copy(), cache['ph']
    params = dict(decoder)
    if name.startswith('short'):
        params['min_team_s'] = .4
        if name != 'short04':
            params['cost_ha'] = 4.5
    if 'neural' in name:
        pd[~cache['seen']] = cache['nd'][~cache['seen']]
    if 'bridge' in name:
        pd = bridge_dead(pd, cache['ball'], cache['segments'])
    if 'expert' in name:
        weight = .25 if name.endswith('25') else .5
        if expert is None:
            raise ValueError('Missing separately trained gap expert')
        mask = ~cache['seen']
        pd[mask] = (1-weight)*pd[mask] + weight*expert[mask]
    return pd, ph, params
