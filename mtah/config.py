"""Константы матча BAR-ATH и пути. Всё опубликовано организаторами (DATA.md, games.py).

Ничего здесь не выведено из просмотра видео или из ответов: только границы, которые организаторы
сами дают всем участникам.
"""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("MTAH_DATA_DIR") or os.path.join(ROOT, "data")
OUT_DIR = os.path.join(ROOT, "out")
REF_DIR = os.path.join(ROOT, "ref")
CACHE_DIR = os.path.join(OUT_DIR, "cache")

GAME = "BAR-ATH"
N_FRAMES = 199319          # строк в сабмите = кадров тактического видео
FPS = 25
PITCH = (105.3, 68.0)      # метры; координаты в трекинге нормированы 0..1

# Оцениваемые кадры (видео-кадры, включительно). DATA.md, «Which frames are graded».
HALVES = ((1, 11185, 90651), (2, 110808, 190197))

LABELS = ("H", "A", "D", "X")
H, A, D, X = 0, 1, 2, 3

PUBLIC_GT = os.path.join(REF_DIR, "BAR-ATH_possession_gt_public.csv")


def asset(name, source="tactical"):
    """Путь к файлу из пакета данных организаторов (раскладка data/<source>/BAR-ATH_<source>_*)."""
    suffix = {
        "video": ".mp4",
        "atd_meta": "_atd.xml",
        "atd_raw": "_atd.txt",
        "homography": "_homography.csv",
        "smart_tagging": "_smart_tagging.xml",
    }[name]
    return os.path.join(DATA_DIR, source, "{}_{}{}".format(GAME, source, suffix))


def graded_mask(n=N_FRAMES):
    """Булева маска оцениваемых кадров."""
    import numpy as np
    mask = np.zeros(n, dtype=bool)
    for _, first, last in HALVES:
        mask[first:last + 1] = True
    return mask
