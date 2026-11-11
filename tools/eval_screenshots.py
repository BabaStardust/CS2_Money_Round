"""Screenshots mit bekannter Wahrheit Ende-zu-Ende prüfen: Vision → Werte → Prognose.

Vergleicht je Bild, was die Inferenz aus dem Screenshot macht, mit der Wahrheit aus der Demo:
Gegnerseite, Score, Rundenhistorie, Kills/Deaths des Gegners und am Schluss die Prognose
gegen das Label «Full Buy möglich».

    uv run --extra granite python tools/eval_screenshots.py                     # Beispiele im Repo, Granite
    uv run python tools/eval_screenshots.py --provider claude_cli
    uv run --extra granite python tools/eval_screenshots.py --set kalibrierung  # lokale Lernbilder

Wahrheit: ``examples/screenshots/truth.json`` bzw. ``data/screenshots/truth.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cs2_money_round.inference import InferenceError, load_model, observation, predict, target_side  # noqa: E402
from cs2_money_round.store import store_from_env  # noqa: E402
from cs2_money_round.vision import make_reader, read_image  # noqa: E402

SETS = {
    "beispiele": (ROOT / "examples" / "screenshots", ROOT / "examples" / "screenshots" / "truth.json"),
    "kalibrierung": (ROOT / "data" / "screenshots" / "kalibrierung", ROOT / "data" / "screenshots" / "truth.json"),
    "demo_original": (ROOT / "data" / "screenshots" / "demo_original", ROOT / "data" / "screenshots" / "truth.json"),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--provider", default="granite")
    ap.add_argument("--set", choices=SETS, default="beispiele")
    ap.add_argument("--ohne-bildanalyse", action="store_true", help="nur das Vision-Modell, ohne Symbolleiste/Zeilenfarbe")
    args = ap.parse_args()
    load_dotenv(ROOT / ".env")

    folder, truth_file = SETS[args.set]
    truth = json.loads(truth_file.read_text())
    reader = make_reader(args.provider)
    store = store_from_env()
    model = load_model(store)
    checks = {"Seite": 0, "Score": 0, "Historie": 0, "Kills": 0, "Deaths": 0, "Prognose": 0}
    n = 0
    for name, t in truth.items():
        path = folder / name
        if not path.exists():
            continue
        n += 1
        started = time.monotonic()
        reading = read_image(path.read_bytes(), "image/png", reader, use_image_analysis=not args.ohne_bildanalyse)
        seconds = time.monotonic() - started
        side = target_side(reading, None)
        score = [reading.ct.score, reading.t.score]
        res = {"Seite": side == t["gegner"], "Score": score == t["score_ct_t"]}
        line = f"{name:32} {seconds:5.0f}s  Seite {side or '?':2}  Score {score}"
        try:
            obs, notes = observation(side or t["gegner"], None, None, None, reading)
            res["Historie"] = obs["obs_round_history"] == t["historie_gegner"]
            res["Kills"] = obs["obs_team_kills_total"] == t["kills_gegner"]
            res["Deaths"] = obs["obs_team_deaths_total"] == t["deaths_gegner"]
            p = predict(obs, store, model, notes)
            res["Prognose"] = p.possible == t["full_buy_moeglich"]
            line += f"  K/D {obs['obs_team_kills_total']}/{obs['obs_team_deaths_total']}  P={p.probability:.0%}"
        except InferenceError as e:
            line += f"  ✗ {e}"
        for k, v in res.items():
            checks[k] += bool(v)
        line += "  " + " ".join(f"{k}{'✓' if v else '✗'}" for k, v in res.items())
        line += f"  (Demo: {'möglich' if t['full_buy_moeglich'] else 'nicht möglich'})"
        print(line, flush=True)
    print(f"\n{args.provider}{' ohne Bildanalyse' if args.ohne_bildanalyse else ''}, {n} Bilder ({args.set}): "
          + ", ".join(f"{k} {v}/{n}" for k, v in checks.items()))
    print("Prognose = Modellentscheid (≥ 50 %) stimmt mit dem Label aus der Demo überein; die Lesung kann richtig "
          "und die Prognose trotzdem falsch sein (Modellgenauigkeit ≈ 73 %).")


if __name__ == "__main__":
    main()
