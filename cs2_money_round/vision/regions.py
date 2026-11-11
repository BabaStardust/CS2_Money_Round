"""Feste Bereiche des CS2-Scoreboards für Vision-Modelle, die Ausschnitte statt ganzer Bilder brauchen.

Ein Zuschnitt verschiebt und skaliert das Scoreboard nur; innerhalb des Scoreboards bleibt alles
am selben relativen Ort. Darum werden alle Bereiche als Anteil des Scoreboard-Panels angegeben:
erst das Panel finden (``locate_panel``), dann die Bereiche daraus schneiden (``crop``).

Kalibriert an den Lern-Screenshots in ``data/screenshots/kalibrierung`` (5120 × 1440, Demo-Ansicht,
Dust II und Anubis): Panel 1424 × 800 px bei (1848, 360). Das Panel skaliert mit der Bildhöhe.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

# Seitenverhältnis des Panels (Breite / Höhe) und Fallback-Lage bei unbeschnittenen Screenshots:
# das Panel skaliert mit der Bildhöhe und sitzt horizontal mittig.
PANEL_ASPECT = 1424 / 800
FALLBACK_HEIGHT = 800 / 1440
FALLBACK_CENTER_Y = 760 / 1440


@dataclass(frozen=True)
class Box:
    """Rechteck in Anteilen des Panels (0..1)."""

    x0: float
    y0: float
    x1: float
    y1: float


REGIONS = {
    # nur die Zahlenspalten Money, Kills, Deaths, Assists (mit Kopfzeile, beide Teams): ohne Namen
    # und Rating liest Granite alle Zeilen; mit Rating hielt es das Rating für das Geld
    "numbers": Box(0.56, 0.17, 0.80, 0.85),
    # Spielernamen beider Teams
    "names": Box(0.12, 0.17, 0.56, 0.85),
    # einzelne Zahlenspalten (für den zweiten Leseversuch, je Team mit TEAM_ROWS kombiniert)
    "col_money": Box(0.570, 0.17, 0.642, 0.85),
    "col_kills": Box(0.651, 0.17, 0.680, 0.85),
    "col_deaths": Box(0.699, 0.17, 0.735, 0.85),
    "col_assists": Box(0.752, 0.17, 0.788, 0.85),
    # die zwei grossen Rundenzahlen links (oben CT, unten T)
    "scores": Box(0.016, 0.195, 0.115, 0.856),
    # Symbolleiste der Rundenhistorie zwischen den Teams
    "history": Box(0.12, 0.49, 0.92, 0.60),
}


# Zeilenmitten der Spielertabelle (Anteil der Panelhöhe): 5 CT oben, 5 T unten
ROW_STEP = 15 / 348
ROW_Y = [(235 - 157) / 348 + i * ROW_STEP for i in range(5)] + [(385 - 157) / 348 + i * ROW_STEP for i in range(5)]
ROW_X = (0.14, 0.91)
# Die hervorgehobene Zeile (eigener Spieler) trägt die Teamfarbe: auf CT blau (Blau − Rot ≈ +30),
# auf T gelb (Rot − Blau ≈ +33, Grün − Blau ≈ +28). Rot getönte Zeilen anderer Spieler
# (Grün ≈ Blau) und neutrale Zeilen (Abweichung < 15) zählen nicht.
HIGHLIGHT_MARGIN = 20


def _is_highlight(r: float, g: float, b: float) -> bool:
    blue = b - r > HIGHLIGHT_MARGIN
    yellow = r - b > HIGHLIGHT_MARGIN and g - b > HIGHLIGHT_MARGIN
    return blue or yellow


def highlighted_rows(img: Image.Image, panel: tuple[int, int, int, int]) -> list[int]:
    """Indizes (0–4 CT, 5–9 T) der in Teamfarbe hervorgehobenen Zeilen."""
    px0, py0, px1, py1 = panel
    pw, ph = px1 - px0, py1 - py0
    rgb = np.asarray(img.convert("RGB"), dtype=np.int16)
    half = max(1, int(ROW_STEP * ph * 0.25))
    x0, x1 = int(px0 + ROW_X[0] * pw), int(px0 + ROW_X[1] * pw)
    hits = []
    for i, ry in enumerate(ROW_Y):
        y = int(py0 + ry * ph)
        seg = rgb[max(0, y - half): y + half, x0:x1].reshape(-1, 3)
        if len(seg) and _is_highlight(*np.median(seg, axis=0)):
            hits.append(i)
    return hits


def _edges(flags: np.ndarray) -> tuple[int, int]:
    """Erster und letzter dunkler Eintrag + 1; Lücken innen (z. B. die Historien-Leiste) bleiben."""
    idx = np.flatnonzero(flags)
    return (int(idx[0]), int(idx[-1]) + 1) if len(idx) else (0, 0)


def _trim(raw: np.ndarray, ys: slice, xs: slice) -> tuple[slice, slice]:
    """Ränder abschneiden, die nur durch die Glättung dazukamen (z. B. die HUD-Leiste unter dem Panel)."""
    for _ in range(2):
        if ys.stop <= ys.start or xs.stop <= xs.start:
            break
        r0, r1 = _edges(raw[ys, xs].mean(axis=1) > 0.5)
        ys = slice(ys.start + r0, ys.start + r1)
        if ys.stop <= ys.start:
            break
        c0, c1 = _edges(raw[ys, xs].mean(axis=0) > 0.5)
        xs = slice(xs.start + c0, xs.start + c1)
    return ys, xs


def _dark_panel_bbox(img: Image.Image, max_lum: int = 90) -> tuple[int, int, int, int] | None:
    """Grösste dunkle, entsättigte Fläche, die die Bildmitte überdeckt (das Scoreboard-Panel)."""
    from scipy import ndimage

    scale = 4
    small = img.convert("RGB").resize((max(1, img.width // scale), max(1, img.height // scale)))
    rgb = np.asarray(small, dtype=np.int16)
    lum = rgb.mean(axis=2)
    sat = rgb.max(axis=2) - rgb.min(axis=2)
    raw = (lum < max_lum) & (sat < 40)
    mask = ndimage.binary_closing(raw, structure=np.ones((5, 5)), iterations=2)
    labels, n = ndimage.label(mask)  # type: ignore[misc]
    if n == 0:
        return None
    cy, cx = small.height // 2, small.width // 2
    best = None
    for i, sl in enumerate(ndimage.find_objects(labels), start=1):
        ys, xs = _trim(raw, *sl)
        h, w = ys.stop - ys.start, xs.stop - xs.start
        if not (xs.start <= cx < xs.stop and ys.start <= cy + small.height // 4 and ys.stop > cy - small.height // 4):
            continue
        if not (0.75 * PANEL_ASPECT <= w / max(h, 1) <= 1.25 * PANEL_ASPECT):
            continue
        area = raw[ys, xs].sum()
        if area < 0.6 * w * h:  # Panel ist grösstenteils dunkel
            continue
        if best is None or w * h > best[2] * best[3]:
            best = (xs.start, ys.start, w, h)
    if best is None:
        return None
    x, y, w, h = best
    return x * scale, y * scale, (x + w) * scale, (y + h) * scale


def locate_panel(img: Image.Image) -> tuple[tuple[int, int, int, int], str]:
    """Scoreboard-Panel in Pixeln und wie es gefunden wurde ('erkannt' oder 'Standardlage')."""
    # Das Panel ist halbtransparent (Median-Helligkeit ≈ 68): vor hellem Hintergrund trennt 90 sauber,
    # vor mittelhellem verschmilzt es damit – dann trennt die strengere Schwelle 70.
    found = _dark_panel_bbox(img, 90) or _dark_panel_bbox(img, 70)
    if found:
        return found, "erkannt"
    h = FALLBACK_HEIGHT * img.height
    w = PANEL_ASPECT * h
    cx, cy = img.width / 2, FALLBACK_CENTER_Y * img.height
    return (int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2)), "Standardlage"


# Zeilenbereich je Team (Anteil der Panelhöhe): fünf Spielerzeilen
TEAM_ROWS = {"CT": (0.195, 0.425), "T": (0.625, 0.855)}


def column_box(column: str, side: str) -> Box:
    """Eine Zahlenspalte (money/kills/deaths/assists) nur für die fünf Zeilen eines Teams."""
    b = REGIONS[f"col_{column}"]
    y0, y1 = TEAM_ROWS[side]
    return Box(b.x0, y0, b.x1, y1)


def crop(img: Image.Image, panel: tuple[int, int, int, int], region: str | Box, min_width: int = 1100) -> Image.Image:
    """Bereich aus dem Panel schneiden und so vergrössern, dass kleine Schrift lesbar wird."""
    px0, py0, px1, py1 = panel
    b = REGIONS[region] if isinstance(region, str) else region
    pw, ph = px1 - px0, py1 - py0
    box = (int(px0 + b.x0 * pw), int(py0 + b.y0 * ph), int(px0 + b.x1 * pw), int(py0 + b.y1 * ph))
    part = img.crop(box)
    factor = max(1, round(min_width / max(part.width, 1)))
    return part.resize((part.width * factor, part.height * factor), Image.Resampling.LANCZOS) if factor > 1 else part
