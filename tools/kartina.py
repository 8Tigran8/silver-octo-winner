"""Ролик «матч точками»: поле сверху, игроки и мяч из тактического трекинга, сверху — кто владеет
мячом по открытым ответам организаторов и по нашему решению, снизу — тег автоматической разметки.

Только для того, чтобы человек глазами понял, что лежит в данных. Ни на что в решении не влияет.

ПРАВИЛО организаторов: ролик из настоящего матча нельзя показывать облачному ИИ. Проверять отрисовку
глазами можно только на фальшивом матче:

    .venv/bin/python -m tests.make_synthetic /tmp/mtah-synth
    MTAH_DATA_DIR=/tmp/mtah-synth .venv/bin/python tools/kartina.py \
        --gt /tmp/mtah-synth/truth.csv --ours /tmp/mtah-synth/truth.csv --png-every 300 --out /tmp/k.mp4

Настоящий матч (по умолчанию: ref/…gt_public.csv и out/v0.csv):

    .venv/bin/python tools/kartina.py

Рисует на numpy, надписи накладывает ffmpeg (libass); других библиотек не нужно.
"""

import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mtah import atd, tagging  # noqa: E402
from mtah.config import OUT_DIR, PITCH, PUBLIC_GT, FPS, N_FRAMES  # noqa: E402

W, H = 1280, 720
SCALE = 7.8                                   # пикселей на метр
PW, PH = int(round(PITCH[0] * SCALE)), int(round(PITCH[1] * SCALE))
X0, Y0 = (W - PW) // 2, 118                   # левый верхний угол поля

GRASS = ((38, 112, 56), (44, 124, 62))
LINE = (235, 240, 235)
BAR_FILL, BAR_RING = (24, 70, 160), (165, 0, 68)      # сине-гранатовые
ATH_FILL, ATH_RING = (220, 30, 36), (255, 255, 255)   # красно-белые
BALL_FILL, BALL_RING = (255, 255, 255), (10, 10, 10)
BG = (18, 22, 28)

SAY = {"H": "у Барселоны", "A": "у Атлетика", "D": "мяч мёртв", "X": "не знаю"}
COL = {"H": "&H00F0A050&", "A": "&H004040FF&", "D": "&H00B0B0B0&", "X": "&H00808080&"}  # ASS: &HAABBGGRR

CODES = {
    "PLAYERS IN THE BOX": "игроки в штрафной",
    "BALL IN FINAL THIRD": "мяч в последней трети",
    "BALL IN THE BOX": "мяч в штрафной",
    "PROGRESSION": "продвижение вперёд",
    "DEFENDING IN MIDDLE THIRD": "оборона в центре поля",
    "SET PIECES": "стандарт",
    "CREATING CHANCES": "создание момента",
    "DEFENDING IN DEFENSIVE THIRD": "оборона у своих ворот",
    "LONG BALL": "длинная передача",
    "ATTACKING TRANSITION": "переход в атаку",
    "DEFENSIVE TRANSITION": "переход в оборону",
    "BUILD UP": "розыгрыш сзади",
    "DEFENDING IN ATTACKING THIRD": "прессинг у чужих ворот",
    "GOALS": "гол",
}
TYPES = {
    "throw-in": "аут", "corner kick": "угловой", "goal kick": "удар от ворот", "free kick": "штрафной",
    "kick off (after goal)": "начало с центра после гола", "kick off (start)": "начало игры",
}
TEAM_RU = {"H": "Барселона", "A": "Атлетик"}


# ---------- рисование на numpy ----------

def _px(xm, ym):
    return X0 + xm * SCALE, Y0 + ym * SCALE


def _rect(img, x1, y1, x2, y2, col):
    x1, x2 = sorted((int(round(x1)), int(round(x2))))
    y1, y2 = sorted((int(round(y1)), int(round(y2))))
    img[max(y1, 0):min(y2, H), max(x1, 0):min(x2, W)] = col


def _box(img, xm1, ym1, xm2, ym2, t=2):
    """Прямоугольник линиями (координаты в метрах)."""
    a, b = _px(xm1, ym1)
    c, d = _px(xm2, ym2)
    _rect(img, a - t / 2, b - t / 2, c + t / 2, b + t / 2, LINE)
    _rect(img, a - t / 2, d - t / 2, c + t / 2, d + t / 2, LINE)
    _rect(img, a - t / 2, b - t / 2, a + t / 2, d + t / 2, LINE)
    _rect(img, c - t / 2, b - t / 2, c + t / 2, d + t / 2, LINE)


def _ring(img, cxm, cym, rm, t=2, keep=None):
    cx, cy = _px(cxm, cym)
    r = rm * SCALE
    yy, xx = np.mgrid[0:H, 0:W]
    dist = np.hypot(xx - cx, yy - cy)
    m = np.abs(dist - r) <= t / 2
    if keep is not None:
        m &= keep(xx, yy)
    img[m] = LINE


def _dot(img, cxm, cym, rm_px, col):
    cx, cy = _px(cxm, cym)
    r = int(np.ceil(rm_px)) + 1
    x1, x2 = int(cx) - r, int(cx) + r + 1
    y1, y2 = int(cy) - r, int(cy) + r + 1
    if x2 <= 0 or y2 <= 0 or x1 >= W or y1 >= H:
        return
    yy, xx = np.mgrid[y1:y2, x1:x2]
    m = np.hypot(xx - cx, yy - cy) <= rm_px
    sub = img[max(y1, 0):min(y2, H), max(x1, 0):min(x2, W)]
    m = m[max(-y1, 0):max(-y1, 0) + sub.shape[0], max(-x1, 0):max(-x1, 0) + sub.shape[1]]
    sub[m] = col


def pitch_image():
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[:] = BG
    L, Wd = PITCH
    # трава полосами (с полями за линиями)
    pad = 1.5
    n = 12
    for k in range(n):
        xa = -pad + (L + 2 * pad) * k / n
        xb = -pad + (L + 2 * pad) * (k + 1) / n
        a, b = _px(xa, -pad)
        c, d = _px(xb, Wd + pad)
        _rect(img, a, b, c + 1, d, GRASS[k % 2])
    _box(img, 0, 0, L, Wd)
    a, _ = _px(L / 2, 0)
    _rect(img, a - 1, Y0, a + 1, Y0 + PH, LINE)
    _ring(img, L / 2, Wd / 2, 9.15)
    _dot(img, L / 2, Wd / 2, 3, LINE)
    for side in (0, 1):
        x_goal = 0 if side == 0 else L
        sgn = 1 if side == 0 else -1
        _box(img, x_goal, Wd / 2 - 20.16, x_goal + sgn * 16.5, Wd / 2 + 20.16)
        _box(img, x_goal, Wd / 2 - 9.16, x_goal + sgn * 5.5, Wd / 2 + 9.16)
        _box(img, x_goal, Wd / 2 - 3.66, x_goal - sgn * 1.6, Wd / 2 + 3.66, t=3)
        _dot(img, x_goal + sgn * 11, Wd / 2, 3, LINE)
        edge = _px(x_goal + sgn * 16.5, 0)[0]
        keep = (lambda xx, yy, e=edge: xx > e) if side == 0 else (lambda xx, yy, e=edge: xx < e)
        _ring(img, x_goal + sgn * 11, Wd / 2, 9.15, keep=keep)
    return img


# ---------- данные ----------

def read_labels(path):
    lab = np.full(N_FRAMES, "X", dtype="<U1")
    with open(path, encoding="utf-8-sig") as fh:
        fh.readline()
        for line in fh:
            p = line.strip().split(",")
            if len(p) == 2 and p[0].isdigit():
                i = int(p[0])
                if i < N_FRAMES:
                    lab[i] = p[1].strip().upper() or "X"
    return lab


def tag_text(tags, f):
    on = [t for t in tags if t["start"] <= f <= t["end"]]
    if not on:
        return ""
    on.sort(key=lambda t: t["start"], reverse=True)       # сначала самое свежее
    parts = []
    for t in on[:2]:
        s = CODES.get(t["code"], t["code"].lower())
        if t["code"] == "SET PIECES" and t["type"] in TYPES:
            s = TYPES[t["type"]]
        if t["team"]:
            s += " · " + TEAM_RU[t["team"]]
        parts.append(_esc(s))
    return "   |   ".join(parts)


# ---------- надписи (ASS) ----------

def _ts(k):
    cs = k * 100 // FPS
    return "{}:{:02d}:{:02d}.{:02d}".format(cs // 360000, cs // 6000 % 60, cs // 100 % 60, cs % 100)


def _esc(s):
    return s.replace("{", "(").replace("}", ")")


ASS_HEAD = """[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Top,DejaVu Sans,40,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,2,0,7,0,0,0,1
Style: Small,DejaVu Sans,26,&H00DDDDDD,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,0,7,0,0,0,1
Style: Foot,DejaVu Sans,30,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,2,0,8,0,0,0,1
Style: Warn,DejaVu Sans,26,&H0040E0FF,&H00FFFFFF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,2,0,7,0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _events(texts, style, pos):
    """texts[k] на кадр k -> склеенные события ASS с одной позицией."""
    out, k0 = [], 0
    for k in range(1, len(texts) + 1):
        if k == len(texts) or texts[k] != texts[k0]:
            if texts[k0]:
                out.append("Dialogue: 0,{},{},{},,0,0,0,,{{\\pos({},{})}}{}".format(
                    _ts(k0), _ts(k), style, pos[0], pos[1], texts[k0]))
            k0 = k
    return out


def build_ass(frames, gt, ours, ball_seen, tags, start):
    def lab(name, l):
        return "{}: {{\\c{}}}{}".format(name, COL[l], SAY[l])

    top1 = [lab("Организаторы", gt[f]) for f in frames]
    top2 = []
    for f in frames:
        mark = ("{\\c&H0050E050&}  ✓ совпадает" if ours[f] == gt[f] else "{\\c&H004060FF&}  ✗ не совпадает") \
            if gt[f] != "X" else ""
        top2.append(lab("Наше решение", ours[f]) + mark)
    clock = []
    for f in frames:
        s = max(f - start, 0) // FPS
        clock.append("Время игры\\N{{\\fs44\\b1}}{}:{:02d}".format(s // 60, s % 60))
    warn = ["" if b else "мяч не виден" for b in ball_seen]
    foot = []
    for f in frames:
        t = tag_text(tags, f)
        foot.append("Разметка: " + t if t else "")
    legend = ["{\\c&H00A04618&}●{\\c&H00FFFFFF&} Барселона\\N{\\c&H00241EDC&}●{\\c&H00FFFFFF&} Атлетик\\N"
              "{\\c&H00FFFFFF&}●{\\c&H00DDDDDD&} мяч\\N\\Nускорено ×5"] * len(frames)
    ev = []
    ev += _events(top1, "Top", (40, 10))
    ev += _events(top2, "Top", (40, 60))
    ev += _events(clock, "Small", (14, Y0 + 4))
    ev += _events(warn, "Warn", (14, Y0 + 110))
    ev += _events(legend, "Small", (X0 + PW + 16, Y0 + 4))
    ev += _events(foot, "Foot", (W // 2, Y0 + PH + 14))
    return ASS_HEAD.format(W=W, H=H) + "\n".join(ev) + "\n"


# ---------- главный ход ----------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--start", type=int, default=11185, help="первый кадр (начало 1-го тайма)")
    ap.add_argument("--frames", type=int, default=7500, help="сколько кадров матча (7500 = 5 минут)")
    ap.add_argument("--step", type=int, default=5, help="брать каждый N-й кадр (ускорение)")
    ap.add_argument("--gt", default=PUBLIC_GT, help="открытые ответы организаторов")
    ap.add_argument("--ours", default=os.path.join(OUT_DIR, "v0.csv"), help="наш файл сдачи")
    ap.add_argument("--tags", default=None, help="smart_tagging.xml (по умолчанию из пакета данных)")
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "kartina", "match-5min-tochki.mp4"))
    ap.add_argument("--png-every", type=int, default=0,
                    help="сохранить каждый N-й выходной кадр в png рядом с роликом (только для синтетики!)")
    ap.add_argument("--crf", type=int, default=24)
    a = ap.parse_args()

    d = atd.load()
    gt, ours = read_labels(a.gt), read_labels(a.ours)
    tags = tagging.load(a.tags)
    frames = np.arange(a.start, min(a.start + a.frames, len(d["frame_ids"])), a.step)
    ball = d["ball"][:, :2] * np.array(PITCH)
    players = d["players"][:, :, :2] * np.array(PITCH)
    team = d["slot_team"]
    ball_seen = ~np.isnan(ball[frames, 0])

    base = pitch_image()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="kartina-")
    ass_path = os.path.join(tmp, "labels.ass")
    with open(ass_path, "w", encoding="utf-8") as fh:
        fh.write(build_ass(frames, gt, ours, ball_seen, tags, a.start))

    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", "{}x{}".format(W, H), "-r", str(FPS), "-i", "-",
           "-vf", "ass={}".format(ass_path), "-c:v", "libx264", "-preset", "medium",
           "-crf", str(a.crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", a.out]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    png_dir = os.path.splitext(a.out)[0] + "_png"
    for k, f in enumerate(frames):
        img = base.copy()
        for s in np.flatnonzero(~np.isnan(players[f, :, 0])):
            x, y = players[f, s]
            ring, fill = (BAR_RING, BAR_FILL) if team[s] == atd.HOME else (ATH_RING, ATH_FILL)
            _dot(img, x, y, 10, ring)
            _dot(img, x, y, 7, fill)
        if ball_seen[k]:
            bx, by = ball[f]
            _dot(img, bx, by, 8, BALL_RING)
            _dot(img, bx, by, 6, BALL_FILL)
        ff.stdin.write(img.tobytes())
    ff.stdin.close()
    if ff.wait() != 0:
        sys.exit("ffmpeg упал")

    if a.png_every:
        os.makedirs(png_dir, exist_ok=True)
        for k in range(0, len(frames), a.png_every):
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", "{:.2f}".format(k / FPS + 0.02),
                            "-i", a.out, "-frames:v", "1", os.path.join(png_dir, "k{:05d}.png".format(k))],
                           check=True)

    # только сводные цифры
    gr = gt[frames] != "X"
    print("кадров в ролике:", len(frames), "| секунд:", round(len(frames) / FPS, 1),
          "| мяч виден, %:", round(100 * ball_seen.mean(), 1),
          "| совпадение наше/организаторы, %:", round(100 * (gt[frames][gr] == ours[frames][gr]).mean(), 1)
          if gr.any() else "—",
          "| тегов разметки:", len(tags),
          "| размер, МБ:", round(os.path.getsize(a.out) / 2 ** 20, 1))


if __name__ == "__main__":
    main()
