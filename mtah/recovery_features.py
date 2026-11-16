"""Player activity features validated in the registered recovery study."""
import warnings
import numpy as np
from . import features2
from .config import PITCH


def activity_features(data, segments):
    """Team/slot/reflection invariant player activity, period-local at 25 Hz.

    The binary gap expert sees no ball coordinates, missing-ball duration,
    current-model outputs, timestamps, identities or reference-derived features.
    """
    if 'gt' in data:
        raise ValueError('Labels are not features')
    out = None
    for a, b in segments:
        xy = data['players'][a:b, :, :2].astype(float) * np.asarray(data.get('pitch', PITCH))
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
