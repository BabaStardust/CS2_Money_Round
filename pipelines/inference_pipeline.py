"""Inference Pipeline: Screenshot der Kaufphase → Features → Modell → P(Full Buy möglich) für das Gegnerteam.

Schritte (gemäss Aufgabenstellung):
  1. Live-Daten: Screenshot per Vision-Modell lesen (oder gespeicherte Lesung / manuelle Eingabe)
     → RT-Features (Spielstand, letzte Runde) und aggregierte Features aus dem Match (Rundenhistorie, K/D)
  2. Feature View holen, neueste Features laden → historische Kaufquote aus buy_rate_history
  3. Modell aus der Model Registry laden
  4. Prediction: Wahrscheinlichkeit, dass das Gegnerteam diese Runde einen Full Buy kaufen kann

Aufruf:
    uv run python pipelines/inference_pipeline.py data/screenshots/runde.png
    uv run python pipelines/inference_pipeline.py --reading data/inference/runde.reading.json
    uv run python pipelines/inference_pipeline.py --reading … --team T --history WeLtWdLe
    # ganz ohne Screenshot (User-Input):
    uv run python pipelines/inference_pipeline.py --team T --history WeWeLbLe --kills 21 --deaths 23

Das Gegnerteam ist standardmässig das Team ohne hervorgehobene Zeile (eigener Spieler).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cs2_money_round.inference import (  # noqa: E402,F401  (Re-Exports für Tests)
    InferenceError, demo_money_check, history_from_reading, load_model, observation, predict,
    side_in_round, target_side,
)
from cs2_money_round.specs import MODEL_NAME  # noqa: E402
from cs2_money_round.store import store_from_env  # noqa: E402
from cs2_money_round.vision.schema import ScoreboardReading  # noqa: E402

READING_DIR = ROOT / "data" / "inference"


# ---------------------------------------------------------------- 1. Live-Daten
def read_live(screenshot: Path | None, reading_path: Path | None) -> ScoreboardReading | None:
    if reading_path:
        return ScoreboardReading.model_validate_json(reading_path.read_text())
    if not screenshot:
        return None
    from cs2_money_round.vision import read_screenshot

    print(f"   Vision-Modell liest {screenshot.name} …")
    reading = read_screenshot(screenshot)
    READING_DIR.mkdir(parents=True, exist_ok=True)
    out = READING_DIR / f"{screenshot.stem}.reading.json"
    out.write_text(reading.model_dump_json(indent=2))
    print(f"   Lesung gespeichert: {out.relative_to(ROOT)} (wiederverwendbar mit --reading)")
    return reading


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("screenshot", nargs="?", type=Path, help="Screenshot während der Kaufphase")
    ap.add_argument("--reading", type=Path, help="gespeicherte Vision-Lesung (JSON) statt Screenshot")
    ap.add_argument("--team", choices=["CT", "T"], help="Seite des Gegnerteams (Standard: Team ohne Hervorhebung)")
    ap.add_argument("--history", help="Rundenhistorie aus Sicht des Gegnerteams, z. B. WeLtWd (überschreibt das Bild)")
    ap.add_argument("--kills", type=int, help="Summe Kills des Gegnerteams (überschreibt das Bild)")
    ap.add_argument("--deaths", type=int, help="Summe Deaths des Gegnerteams (überschreibt das Bild)")
    args = ap.parse_args()
    load_dotenv(ROOT / ".env")

    print("1. Live-Daten (Screenshot / Eingabe)")
    try:
        reading = read_live(args.screenshot, args.reading)
        side = target_side(reading, args.team)
        if side is None:
            sys.exit("Gegnerteam unklar (keine eindeutig hervorgehobene Zeile) – --team CT|T angeben.")
        obs, notes = observation(side, args.history, args.kills, args.deaths, reading)
        print(f"   Gegnerteam {side}, Runde {obs['round_number']}, Stand {obs['obs_score_team']}:{obs['obs_score_opp']}, "
              f"Historie {obs['obs_round_history'] or '–'}")
        store = store_from_env()
        print(f"2. Feature View holen, neueste Features laden [{store.kind}]")
        print("3. Modell aus der Model Registry laden")
        result = predict(obs, store, notes=notes)
    except InferenceError as e:
        sys.exit(str(e))
    for note in result.notes:
        print(f"   ⚠ {note}")
    if result.skipped:
        print(f"   ➜ {result.skipped}")
        return
    r = result.hist_rate
    print(f"   hist_full_buy_rate({side}, Stufe {r['loss_bonus_level']}) = {r['rate']:.0%} "
          f"aus {r['n_rounds']} Runden, Stand {r['updated_at']:%d.%m.%Y}")
    print(f"   {MODEL_NAME} v{result.model_version} (ROC-AUC Test {result.model_auc})")
    print("4. Prediction")
    for group, values in (("RT ", result.rt), ("AGG", result.agg)):
        print(f"   {group} " + ", ".join(f"{n}={v:.3g}" for n, v in values.items()))
    if reading and (check := demo_money_check(reading, side)):
        print(f"   Kontrolle Demo-Ansicht: {check}")
    verdict = "Full Buy möglich" if result.possible else "kein Full Buy möglich"
    print(f"   ➜ Gegner {side}: P(Full Buy möglich) = {result.probability:.0%}  →  {verdict} "
          f"(Konfidenz {result.confidence:.0%})")


if __name__ == "__main__":
    main()
