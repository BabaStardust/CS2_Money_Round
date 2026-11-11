"""Demo-Screenshots für das Repo aufbereiten: auf das Scoreboard zuschneiden, Namen verpixeln.

Die Originale (``data/screenshots/demo_original``, nicht im Repo) zeigen Spielernamen aus eigenen
Matches. Für ``examples/screenshots`` wird nur das Scoreboard-Panel behalten und die Spalte mit
Avatar, Clan-Tag und Name verpixelt. Die Vorhersage braucht keine Namen; der Zuschnitt zeigt
zugleich, dass die Pipeline auch Ausschnitte liest.

    uv run python tools/make_example_screenshots.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cs2_money_round.vision.regions import locate_panel  # noqa: E402

SRC = ROOT / "data" / "screenshots" / "demo_original"
OUT = ROOT / "examples" / "screenshots"
MARGIN = 0.03
# Spalte Rang/Avatar/Clan-Tag/Name in Anteilen des Panels, je Team (y0, y1)
NAME_X = (0.10, 0.50)
NAME_ROWS = [(0.195, 0.425), (0.625, 0.855)]


def pixelate(img: Image.Image, box: tuple[int, int, int, int], block: int) -> None:
    part = img.crop(box)
    small = part.resize((max(1, part.width // block), max(1, part.height // block)), Image.Resampling.BOX)
    img.paste(small.resize(part.size, Image.Resampling.NEAREST), box)


def main() -> None:
    truth = json.loads((ROOT / "data" / "screenshots" / "truth.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    public = {}
    for name, t in truth.items():
        if t["zweck"] != "demo":
            continue
        img = Image.open(SRC / name).convert("RGB")
        (x0, y0, x1, y1), _ = locate_panel(img)
        w, h = x1 - x0, y1 - y0
        for ya, yb in NAME_ROWS:
            box = (int(x0 + NAME_X[0] * w), int(y0 + ya * h), int(x0 + NAME_X[1] * w), int(y0 + yb * h))
            pixelate(img, box, block=max(8, w // 90))
        m = int(MARGIN * w)
        img.crop((x0 - m, y0 - m, x1 + m, y1 + m)).save(OUT / name, optimize=True)
        hist = t["historie_gegner"]
        verdict = "möglich" if t["full_buy_moeglich"] else "nicht möglich"
        (OUT / name).with_suffix(".txt").write_text(
            f"{t['map'].replace('dust2', 'Dust II').replace('anubis', 'Anubis')} · Runde {t['runde']} · "
            f"Score {t['score_ct_t'][0]}:{t['score_ct_t'][1]} (CT:T) · Gegner {t['gegner']} · laut Demo: Full Buy {verdict}\n")
        public[name] = {k: t[k] for k in ("map", "runde", "gegner", "score_ct_t", "historie_gegner", "kills_gegner",
                                          "deaths_gegner", "full_buy_moeglich", "teamgeld_gegner", "min_geld_gegner")}
        print(name, "→", (OUT / name).stat().st_size // 1024, "KB ·", hist, "·", verdict)
    (OUT / "truth.json").write_text(json.dumps(public, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
