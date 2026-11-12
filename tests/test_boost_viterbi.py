"""Витерби с цепочками (mtah/methods/boost.py) против перебора на коротких рядах."""
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mtah.methods.boost import _viterbi_chain   # noqa: E402


def brute(E, C, m):
    T = len(E)
    best, arg = -1e18, None
    for path in itertools.product(range(4), repeat=T):
        ok, s = True, E[0, path[0]]
        runs = []
        cur = 1
        for t in range(1, T):
            a, b = path[t - 1], path[t]
            if a != b:
                if not np.isfinite(C[a, b]) or cur < m[a]:
                    ok = False
                    break
                s -= C[a, b]
                cur = 1
            else:
                cur += 1
            s += E[t, b]
        if ok and s > best:
            best, arg = s, path
    return np.array(arg)


def score(E, C, path):
    s = E[0, path[0]]
    for t in range(1, len(path)):
        if path[t] != path[t - 1]:
            s -= C[path[t - 1], path[t]]
        s += E[t, path[t]]
    return s


rng = np.random.default_rng(1)
inf = np.inf
for trial in range(60):
    T = int(rng.integers(3, 8))
    E = rng.normal(0, 2, (T, 4))
    C = np.array([[0, 1, 2, inf], [1, 0, inf, 2], [1.5, 0.5, 0, inf], [0.5, 1.5, inf, 0]]) * rng.uniform(0.2, 3)
    m = [int(v) for v in rng.integers(1, 4, 4)]
    p = _viterbi_chain(E, C, m)
    b = brute(E, C, m)
    # последний кусок может быть короче минимума (обрыв ряда) — перебор это тоже разрешает
    assert abs(score(E, C, p) - score(E, C, b)) < 1e-9, (trial, p, b, m)
print("viterbi_chain: совпадает с перебором на 60 случаях")
