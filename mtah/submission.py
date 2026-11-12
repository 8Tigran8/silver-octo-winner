"""Запись, чтение и проверка файла сдачи. Проверки — те же, что в validate_submission.py организаторов
(SUBMISSION.md): заголовок frame,label; ровно 199 319 строк; кадры 0..199318 по порядку; метки H/A/D/X;
пустых ячеек нет.
"""

import os

import numpy as np

from .config import N_FRAMES, LABELS


def write(path, labels):
    labels = np.asarray(labels)
    assert len(labels) == N_FRAMES, "нужно {} меток, а их {}".format(N_FRAMES, len(labels))
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", newline="\n") as fh:
        fh.write("frame,label\n")
        fh.writelines("{},{}\n".format(i, l) for i, l in enumerate(labels))
    return path


def read(path):
    """-> (frames: int array, labels: str array). Без pandas, построчно."""
    with open(path, encoding="utf-8-sig") as fh:
        header = fh.readline()
        rows = [line.rstrip("\r\n").split(",") for line in fh if line.strip() != ""]
    return header, rows


def validate(path, n_expected=N_FRAMES):
    """None, если файл годен; иначе строка с причиной."""
    header, rows = read(path)
    cols = [c.strip().lower() for c in header.strip().split(",")]
    if cols != ["frame", "label"]:
        return "заголовок {!r}, нужен frame,label".format(header.strip())
    if len(rows) != n_expected:
        return "строк {}, нужно {}".format(len(rows), n_expected)
    if any(len(r) != 2 for r in rows):
        return "есть строки не из двух полей"
    try:
        frames = np.array([int(r[0]) for r in rows])
    except ValueError:
        return "столбец frame не целые числа"
    if not np.array_equal(frames, np.arange(n_expected)):
        return "кадры не 0..{} по порядку".format(n_expected - 1)
    labels = [r[1].strip().upper() for r in rows]
    if any(l == "" for l in labels):
        return "пустые метки; для отказа пиши X"
    bad = sorted(set(labels) - set(LABELS))
    if bad:
        return "метки вне H/A/D/X: {}".format(bad[:5])
    return None


def load_labels(path):
    err = validate(path)
    if err:
        raise ValueError("{}: {}".format(path, err))
    _, rows = read(path)
    return np.array([r[1].strip().upper() for r in rows])
