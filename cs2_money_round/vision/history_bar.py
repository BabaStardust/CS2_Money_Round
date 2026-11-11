"""Rundenhistorie aus der Symbolleiste des Scoreboards lesen – Bildanalyse statt Sprachmodell.

Aufbau der Leiste (kalibriert an ``data/screenshots/kalibrierung``):
  - Je Runde ein kurzer Strich auf einer Linie. Farbe = Sieger dieser Runde: Gold = T, Blau = CT,
    Weiss = laufende Runde, Grau = noch nicht gespielt.
  - Über oder unter dem Strich ein Symbol für den Grund: Totenkopf (alle tot), Stern (Bombe
    explodiert), Zange (entschärft), Uhr (Zeit abgelaufen).

Vorgehen: Linie mit den meisten farbigen Pixeln suchen → Striche als Farbläufe zählen → je
gespieltem Strich das Symbol ausschneiden und mit gelernten Vorlagen vergleichen (nächster
Nachbar). Die Vorlagen erzeugt ``tools/learn_history_icons.py`` aus Screenshots mit bekannter
Wahrheit aus den Demos; sie liegen in ``history_icons.json``.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from .schema import RoundResult

TEMPLATES = Path(__file__).with_name("history_icons.json")
REASON_NAMES = {"e": "elimination", "b": "bomb", "d": "defuse", "t": "time"}
BAR = (0.17, 0.47, 0.70, 0.62)  # Leiste in Anteilen des Panels (x0, y0, x1, y1)
ICON_SIZE = 16
MAX_ICON_DISTANCE = 0.35  # mittlere Abweichung je Pixel; darüber gilt der Grund als unbekannt


# Feste Lage der 24 Rundenstriche in Anteilen der Leiste ``BAR`` (gemessen an den Kalibrier-Screenshots)
SLOT_START = (0.049, 0.564)  # Mitte des ersten Strichs der 1. bzw. 2. Halbzeit
SLOT_PITCH = 0.03927
SLOT_WIDTH = 0.035
LINE_Y = 0.508


def bar_image(img: Image.Image, panel: tuple[int, int, int, int]) -> Image.Image:
    px0, py0, px1, py1 = panel
    w, h = px1 - px0, py1 - py0
    return img.convert("RGB").crop((int(px0 + BAR[0] * w), int(py0 + BAR[1] * h), int(px0 + BAR[2] * w), int(py0 + BAR[3] * h)))


def slot_color(rgb: np.ndarray) -> str:
    """Mittlere Farbe eines Strichs → 'G' (T), 'B' (CT), 'W' (laufende Runde), 'g' (grau/leer)."""
    r, g, b = (float(v) for v in rgb)
    if min(r, g, b) > 150 and max(r, g, b) - min(r, g, b) < 25:  # weiss ≈ 170–220, grau ≈ 70
        return "W"
    if r - b > 25 and r > 100:
        return "G"
    if b - r > 15 and b > 100:
        return "B"
    return "g"


def _line_y(rgb: np.ndarray) -> int:
    """Zeile der Striche: um ``LINE_Y`` (± 6 % der Höhe) die Zeile mit den meisten farbigen Pixeln.

    Die Linie ist nur wenige Pixel hoch; ein um 1–2 px verschobenes Panel (Zuschnitt) träfe sie sonst nicht.
    """
    h = rgb.shape[0]
    lo, hi = max(0, int((LINE_Y - 0.06) * h)), min(h, int((LINE_Y + 0.06) * h) + 1)
    r, g, b = rgb[lo:hi, :, 0], rgb[lo:hi, :, 1], rgb[lo:hi, :, 2]
    sat = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
    colored = ((sat > 25) & (np.maximum(r, b) > 100)) | (np.minimum(np.minimum(r, g), b) > 150)
    return lo + int(np.argmax(colored.sum(axis=1)))


def _slot_colors(bar: Image.Image) -> tuple[list[tuple[int, int, str]], int]:
    """Alle 24 Striche als (x0, x1, Farbe) und die y-Lage der Linie in ``bar``."""
    rgb = np.asarray(bar, dtype=np.float32)
    y = _line_y(rgb)
    dy = max(1, bar.height // 120)
    half = SLOT_WIDTH * bar.width / 2
    out = []
    for i in range(24):
        cx = (SLOT_START[i // 12] + (i % 12) * SLOT_PITCH) * bar.width
        x0, x1 = max(0, int(cx - 0.8 * half)), int(cx + 0.8 * half) + 1
        patch = rgb[max(0, y - dy): y + dy + 1, x0:x1].reshape(-1, 3)
        if not len(patch):
            out.append((int(cx - half), int(cx + half) + 1, "g"))
            continue
        # die farbigsten bzw. hellsten Pixel zählen (Rundennummern und Hintergrund überdecken Teile des
        # Strichs; bei kleiner Auflösung mischt sich der dünne Strich mit dem dunklen Hintergrund)
        third = max(1, len(patch) // 3)
        sat = patch.max(axis=1) - patch.min(axis=1)
        brightest = slot_color(patch[np.argsort(patch.mean(axis=1))[-third:]].mean(axis=0))
        if brightest == "W":
            color = "W"
        else:
            color = slot_color(patch[np.argsort(sat)[-third:]].mean(axis=0)) if sat.max() > 20 else slot_color(patch.mean(axis=0))
        out.append((max(0, int(cx - half)), int(cx + half) + 1, color))
    return out, y


def slots(bar: Image.Image) -> tuple[list[tuple[int, int, str]], int]:
    """Gespielte Runden als (x0, x1, Farbe) und die y-Lage der Linie.

    Gespielt ist alles vor dem weissen Strich (laufende Runde). Fehlt er, zählen die
    Striche ab Runde 1, solange sie gold oder blau sind. Ein ausgegrauter Strich vor dem
    weissen (kommt in CS2 vor) zählt als gespielt mit Farbe 'g'.
    """
    all_slots, y = _slot_colors(bar)
    colors = [c for _, _, c in all_slots]
    if "W" in colors:
        return all_slots[: colors.index("W")], y
    played = []
    for s in all_slots:
        if s[2] not in ("G", "B"):
            break
        played.append(s)
    return played, y


def _icon_box(bar: Image.Image, x0: int, x1: int, y: int) -> tuple[np.ndarray, str]:
    """Graustufen-Ausschnitt des Symbols über oder unter dem Strich (dort, wo mehr Kontrast ist)."""
    gray = np.asarray(bar.convert("L"), dtype=np.float32)
    background = float(np.median(gray))
    size = x1 - x0
    boxes = {
        "oben": (max(0, y - int(1.25 * size)), max(1, y - int(0.15 * size))),
        "unten": (y + int(0.15 * size) + 1, min(gray.shape[0], y + int(1.25 * size) + 1)),
    }
    best = max(boxes, key=lambda k: (gray[boxes[k][0]: boxes[k][1], x0:x1] > background + 18).sum())
    a, b = boxes[best]
    return gray[a:b, x0:x1] - background, best


def icon_vector(bar: Image.Image, x0: int, x1: int, y: int) -> tuple[np.ndarray, str]:
    """Symbol als Binärbild ``ICON_SIZE``² und seine Lage ('oben'/'unten').

    Die Schwelle passt sich dem Symbol an, damit auch ausgegraute Symbole dieselbe Form ergeben.
    """
    patch, where = _icon_box(bar, x0, x1, y)
    if patch.size == 0 or patch.max() <= 10:
        return np.zeros(ICON_SIZE * ICON_SIZE, dtype=np.float32), where
    mask = patch > 0.5 * patch.max()
    # auf die Umrandung des Symbols zuschneiden und quadratisch ergänzen: unabhängig von 1–2 px Versatz
    ys, xs = np.nonzero(mask)
    mask = mask[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]
    side = max(mask.shape)
    square = np.zeros((side, side), dtype=bool)
    oy, ox = (side - mask.shape[0]) // 2, (side - mask.shape[1]) // 2
    square[oy: oy + mask.shape[0], ox: ox + mask.shape[1]] = mask
    norm = Image.fromarray((square * 255).astype(np.uint8)).resize((ICON_SIZE, ICON_SIZE), Image.Resampling.BILINEAR)
    return (np.asarray(norm, dtype=np.float32) / 255).ravel(), where


@lru_cache(maxsize=1)
def _templates() -> dict[str, np.ndarray]:
    if not TEMPLATES.exists():
        return {}
    raw = json.loads(TEMPLATES.read_text())
    return {k: np.array(v, dtype=np.float32) for k, v in raw["vectors"].items() if v}


def classify(vec: np.ndarray) -> str | None:
    """Nächste Vorlage → 'e'/'b'/'d'/'t'; None, wenn nichts nah genug ist."""
    best, dist = None, np.inf
    for reason, vecs in _templates().items():
        d = float(np.abs(vecs - vec).mean(axis=1).min())
        if d < dist:
            best, dist = reason, d
    return best if dist <= MAX_ICON_DISTANCE else None


def _side_then(side_now: str, round_now: int, past_round: int) -> str:
    """Seite eines Teams in einer früheren Runde (Seitenwechsel nach Runde 12)."""
    flipped = (round_now - 1) // 12 != (past_round - 1) // 12
    return ("T" if side_now == "CT" else "CT") if flipped else side_now


def read_history(img: Image.Image, panel: tuple[int, int, int, int]) -> tuple[list[RoundResult], list[str]]:
    """Gespielte Runden mit Sieger (Seite zu jener Zeit) und Grund, älteste zuerst, plus Hinweise.

    Sieger: Symbol über dem Strich = Team, das jetzt oben steht (CT), darunter = Team unten (T).
    Die Strichfarbe (Gold = T, Blau = CT) prüft das gegen; bei Widerspruch gilt die Farbe.
    """
    bar = bar_image(img, panel)
    played, y = slots(bar)
    round_now = len(played) + 1
    out, notes = [], []
    for i, (x0, x1, color) in enumerate(played, start=1):
        vec, where = icon_vector(bar, x0, x1, y)
        by_position = _side_then("CT" if where == "oben" else "T", round_now, i)
        by_color = {"G": "T", "B": "CT"}.get(color)
        if by_color and by_color != by_position:
            notes.append(f"Runde {i}: Symbol-Lage und Strichfarbe widersprechen sich, Farbe gilt")
        code = classify(vec)
        out.append(RoundResult(round_number=i, winner=by_color or by_position,  # type: ignore[arg-type]
                               reason=REASON_NAMES.get(code) if code else None))  # type: ignore[arg-type]
    return out, notes
