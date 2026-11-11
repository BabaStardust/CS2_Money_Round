"""Symbolvorlagen der Rundenhistorie aus Screenshots mit bekannter Wahrheit lernen.

Eingabe: ``data/screenshots/truth.json`` (Wahrheit aus den Demos, je Bild Rundenhistorie aus
Sicht des Gegners) und die Bilder mit ``zweck = kalibrierung``. Die Demo-Bilder (``zweck = demo``)
bleiben aussen vor und dienen als unabhängiger Test.

Ausgabe: ``cs2_money_round/vision/history_icons.json`` – je Grund (e/b/d/t) die Binärbilder der
Symbole. Enthält keine Spielernamen, nur Symbolformen.

    uv run python tools/learn_history_icons.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cs2_money_round.vision.history_bar import TEMPLATES, bar_image, icon_vector, slots  # noqa: E402
from cs2_money_round.vision.regions import locate_panel  # noqa: E402

SHOTS = ROOT / "data" / "screenshots"
SCALES = (1, 0.75, 0.5)


def main() -> None:
    truth = json.loads((SHOTS / "truth.json").read_text())
    vectors: dict[str, list[list[float]]] = {"e": [], "b": [], "d": [], "t": []}
    for name, t in truth.items():
        if t["zweck"] != "kalibrierung":
            continue
        full = Image.open(SHOTS / "kalibrierung" / name).convert("RGB")
        reasons = t["historie_gegner"][1::2]
        # mehrere Auflösungen: 1440p wie aufgenommen, ~1080p und ~720p (kleinere Symbole sehen anders aus)
        for scale in SCALES:
            size = (int(full.width * scale), int(full.height * scale))
            img = full if scale == 1 else full.resize(size, Image.Resampling.LANCZOS)
            panel, _ = locate_panel(img)
            bar = bar_image(img, panel)
            played, y = slots(bar)
            if len(played) != len(reasons):
                print(f"   übersprungen: {name} ×{scale}: {len(played)} Striche gelesen, laut Demo {len(reasons)}")
                continue
            for (x0, x1, _), reason in zip(played, reasons):
                vec, _ = icon_vector(bar, x0, x1, y)
                vectors[reason].append([round(float(v), 3) for v in vec])
    TEMPLATES.write_text(json.dumps({
        "quelle": "data/screenshots/kalibrierung (Wahrheit aus Demos), erzeugt mit tools/learn_history_icons.py",
        "vectors": vectors,
    }) + "\n")
    print(f"{TEMPLATES.relative_to(ROOT)}: " + ", ".join(f"{k}={len(v)}" for k, v in vectors.items()))
    missing = [k for k, v in vectors.items() if not v]
    if missing:
        print(f"Ohne Beispiel (wird als unbekannt gelesen): {', '.join(missing)}")
    print("Verteilung:", Counter({k: len(v) for k, v in vectors.items()}))


if __name__ == "__main__":
    main()
