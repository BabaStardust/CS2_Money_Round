"""Wahrheit der Screenshots aus den eingelesenen Demos (neu) berechnen.

``data/screenshots/truth.json`` ordnet jedem Screenshot Match, Runde, Gegnerseite und Zweck zu
(``demo`` = wird ausgeliefert, ``kalibrierung`` = Lernbild). Dieses Werkzeug füllt die Werte aus
dem Demo-Cache (``data/interim/rounds``) neu: Score, Rundenhistorie aus Sicht des Gegners,
Kills/Deaths des Gegners und das Label. Danach ``tools/make_example_screenshots.py`` für die
ausgelieferte Kopie in ``examples/screenshots/truth.json``.

    uv run python tools/screenshot_truth.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TRUTH = ROOT / "data" / "screenshots" / "truth.json"
CACHE = ROOT / "data" / "interim" / "rounds"


def main() -> None:
    truth = json.loads(TRUTH.read_text())
    frames: dict[str, pd.DataFrame] = {}
    for name, t in truth.items():
        df = frames.setdefault(t["match_id"], pd.read_parquet(CACHE / f"{t['match_id']}.parquet"))
        rnd = df[df["round_number"] == t["runde"]]
        x = rnd[rnd["team_side"] == t["gegner"]].iloc[0]
        ct = rnd[rnd["team_side"] == "CT"].iloc[0]
        new = {
            "score_ct_t": [int(ct.obs_score_team), int(ct.obs_score_opp)],
            "historie_gegner": x.obs_round_history,
            "kills_gegner": int(x.obs_team_kills_total),
            "deaths_gegner": int(x.obs_team_deaths_total),
            "full_buy_moeglich": bool(x.full_buy_possible),
            "teamgeld_gegner": int(x.gt_team_cash),
            "min_geld_gegner": int(x.gt_min_cash),
        }
        changed = {k: (t.get(k), v) for k, v in new.items() if t.get(k) != v}
        t.update(new)
        print(f"{name:32} " + (", ".join(f"{k}: {a} → {b}" for k, (a, b) in changed.items()) or "unverändert"))
    TRUTH.write_text(json.dumps(truth, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
