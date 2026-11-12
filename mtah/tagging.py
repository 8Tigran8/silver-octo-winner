"""Smart Tagging (автоматическая тактическая разметка Metrica, SportsCode XML) -> интервалы в кадрах.

Время <start>/<end> — секунды от начала тактического видео; кадр = round(сек * 25) (DATA.md).
Команды в теге Team: 'Barcelona' -> H, 'Atletic Club' -> A, 'N/A' (все ауты) -> None.
Это автоматическая разметка, а не ответы: использовать как подсказку, а не как метку.
"""

import os
import xml.etree.ElementTree as ET

from .config import asset, FPS

TEAM_MAP = {"barcelona": "H", "atletic club": "A", "athletic club": "A"}


def load(path=None):
    path = path or asset("smart_tagging")
    if not os.path.isfile(path):
        return []
    out = []
    for inst in ET.parse(path).iter("instance"):
        tags = {(l.findtext("group") or "").strip(): (l.findtext("text") or "").strip()
                for l in inst.findall("label")}
        team = TEAM_MAP.get(tags.get("Team", "").lower())
        out.append({
            "code": (inst.findtext("code") or "").strip().upper(),
            "start": int(round(float(inst.findtext("start")) * FPS)),
            "end": int(round(float(inst.findtext("end")) * FPS)),
            "team": team,
            "type": tags.get("Type", "").lower(),
            "half": tags.get("Half", ""),
        })
    return out
