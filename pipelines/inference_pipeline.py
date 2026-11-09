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
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cs2_money_round.features import (  # noqa: E402
    AGG_FEATURES, HALF_ROUNDS, RT_FEATURES, compute_features, parse_history,
)
from cs2_money_round.model import Rules  # noqa: E402
from cs2_money_round.specs import MODEL_NAME, RATES_FV  # noqa: E402
from cs2_money_round.store import store_from_env  # noqa: E402
from cs2_money_round.vision.schema import ScoreboardReading  # noqa: E402

READING_DIR = ROOT / "data" / "inference"
REASON_CODE = {"elimination": "e", "bomb": "b", "defuse": "d", "time": "t", None: "?"}
CODE_REASON = {"e": "elimination", "b": "bomb", "d": "defuse", "t": "time", "?": None}


def half_of(round_number: int) -> int:
    """Halbzeit-Index: 0/1 regulär (je 12 Runden), danach Overtime-Hälften à 3 Runden."""
    if round_number <= 2 * HALF_ROUNDS:
        return (round_number - 1) // HALF_ROUNDS
    return 2 + (round_number - 2 * HALF_ROUNDS - 1) // 3


def side_in_round(side_now: str, round_now: int, past_round: int) -> str:
    """Seite, auf der das Team ``side_now`` in einer früheren Runde stand (Seitenwechsel je Halbzeit)."""
    flipped = (half_of(round_now) - half_of(past_round)) % 2 == 1
    return ("T" if side_now == "CT" else "CT") if flipped else side_now


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


def target_side(reading: ScoreboardReading | None, team: str | None) -> str:
    if team:
        return team
    if reading:
        own = [s for s, t in (("CT", reading.ct), ("T", reading.t)) if any(p.highlighted for p in t.players)]
        if len(own) == 1:
            return "T" if own[0] == "CT" else "CT"
    sys.exit("Gegnerteam unklar (keine eindeutig hervorgehobene Zeile) – --team CT|T angeben.")


def history_from_reading(reading: ScoreboardReading, side: str, round_number: int) -> str | None:
    """Rundenhistorie des Bildes → 'WeLt…' aus Sicht des Teams; None, wenn unvollständig."""
    played = round_number - 1
    rounds = sorted(reading.round_history, key=lambda r: r.round_number)
    if len(rounds) != played or [r.round_number for r in rounds] != list(range(1, played + 1)):
        return None
    return "".join(
        ("W" if r.winner == side_in_round(side, round_number, r.round_number) else "L") + REASON_CODE[r.reason]
        for r in rounds
    )


def observation(args, reading: ScoreboardReading | None, side: str) -> dict:
    """Live-Werte → obs_*-Zeile im selben Format wie der Demo-Import."""
    team = opp = None
    if reading:
        team, opp = (reading.ct, reading.t) if side == "CT" else (reading.t, reading.ct)
    history = args.history
    if history is None and reading and team and opp and team.score is not None and opp.score is not None:
        history = history_from_reading(reading, side, team.score + opp.score + 1)
        if history is None:
            sys.exit("Rundenhistorie auf dem Bild unvollständig – mit --history nachreichen "
                     "(je Runde W/L + Grund e/b/d/t aus Sicht des Gegnerteams, z. B. WeLtWd).")
    if history is None:
        sys.exit("Ohne Screenshot ist --history nötig (z. B. --history WeLtWd).")
    parsed = parse_history(history)
    played = len(parsed)
    wins = sum(w for w, _ in parsed)
    if team and opp and team.score is not None and opp.score is not None and (team.score, opp.score) != (wins, played - wins):
        print(f"   ⚠ Score laut Bild {team.score}:{opp.score} (Gegner {side}), laut Historie {wins}:{played - wins} – "
              "Historie gilt. Ist --history aus Sicht des Gegnerteams?")

    kills, deaths = args.kills, args.deaths
    if team and (kills is None or deaths is None):
        k = [p.kills for p in team.players]
        d = [p.deaths for p in team.players]
        if len(team.players) != 5 or None in k or None in d:
            sys.exit("Kills/Deaths des Gegnerteams nicht vollständig lesbar – --kills und --deaths angeben.")
        kills = kills if kills is not None else sum(k)  # type: ignore[arg-type]
        deaths = deaths if deaths is not None else sum(d)  # type: ignore[arg-type]
    if kills is None or deaths is None:
        sys.exit("--kills und --deaths (Summe des Gegnerteams) angeben.")

    last = parsed[-1] if parsed else None
    return {
        "round_number": played + 1,
        "team_side": side,
        "obs_round_history": history,
        "obs_score_team": sum(w for w, _ in parsed),
        "obs_score_opp": sum(not w for w, _ in parsed),
        "obs_prev_round_won": last[0] if last else False,
        "obs_prev_round_reason": CODE_REASON[last[1]] if last else None,
        "obs_team_kills_total": kills,
        "obs_team_deaths_total": deaths,
        "displayed_loss_bonus": team.loss_bonus_level if team else None,
    }


def demo_money_check(reading: ScoreboardReading, side: str, rules: Rules = Rules()) -> str | None:
    """Nur Demo-Ansicht: sichtbares Geld der Gegner gegen den Neukaufpreis (ohne gespeicherte Waffen)."""
    team = reading.ct if side == "CT" else reading.t
    money = [p.money for p in team.players]
    if not reading.is_demo_view or None in money:
        return None
    need = rules.rifle_price(side) + rules.kevlar_price + rules.helmet_upgrade_price  # type: ignore[arg-type]
    ok = sum(m >= need for m in money)  # type: ignore[operator]
    return f"{ok}/5 Spieler haben ≥ ${need:,} für Rifle + Kevlar/Helm (gespeicherte Ausrüstung nicht sichtbar)"


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
    reading = read_live(args.screenshot, args.reading)
    side = target_side(reading, args.team)
    obs = observation(args, reading, side)
    rnd = obs["round_number"]
    print(f"   Gegnerteam {side}, Runde {rnd}, Stand {obs['obs_score_team']}:{obs['obs_score_opp']}, "
          f"Historie {obs['obs_round_history'] or '–'}")
    if rnd in (1, HALF_ROUNDS + 1):
        print("   ➜ Pistolenrunde: Full Buy unmöglich (Startgeld). Keine Modellvorhersage nötig.")
        return
    if rnd > 2 * HALF_ROUNDS:
        print("   ➜ Overtime ist nicht im Trainingsumfang – keine Vorhersage.")
        return
    x = compute_features(pd.DataFrame([obs])).iloc[0].to_dict()
    if obs["displayed_loss_bonus"] is not None and obs["displayed_loss_bonus"] != x["loss_bonus_level"]:
        print(f"   ⚠ Loss Bonus laut Bild {obs['displayed_loss_bonus']}, aus Historie berechnet {x['loss_bonus_level']}")

    store = store_from_env()
    print(f"2. Feature View holen, neueste Features laden [{store.kind}]")
    entry = {"team_side": side, "loss_bonus_level": int(x["loss_bonus_level"])}
    rate = store.feature_vector(RATES_FV, entry)
    if rate is None:
        sys.exit(f"Kein Eintrag in {RATES_FV.group.name} für {entry} – zuerst die Feature Pipeline laufen lassen.")
    x["hist_full_buy_rate"] = rate["hist_full_buy_rate"]
    print(f"   hist_full_buy_rate({side}, Stufe {entry['loss_bonus_level']}) = {rate['hist_full_buy_rate']:.0%} "
          f"aus {int(rate['n_rounds'])} Runden, Stand {pd.Timestamp(rate['updated_at']):%d.%m.%Y}")

    print("3. Modell aus der Model Registry laden")
    handle = store.latest_model(MODEL_NAME)
    # joblib/pickle: das Modell stammt aus der eigenen Registry (vertrauenswürdige Quelle)
    model = joblib.load(handle.path / "model.pkl")
    features = json.loads((handle.path / "metadata.json").read_text())["features"]
    print(f"   {MODEL_NAME} v{handle.version} (ROC-AUC Test {handle.metrics.get('roc_auc')})")

    print("4. Prediction")
    X = pd.DataFrame([x])[features]
    p = float(model.predict_proba(X)[0, 1])
    for group, names in (("RT ", RT_FEATURES), ("AGG", AGG_FEATURES)):
        print(f"   {group} " + ", ".join(f"{n}={X.iloc[0][n]:.3g}" for n in names if n in features))
    if reading and (check := demo_money_check(reading, side)):
        print(f"   Kontrolle Demo-Ansicht: {check}")
    verdict = "Full Buy möglich" if p >= 0.5 else "kein Full Buy möglich"
    print(f"   ➜ Gegner {side}: P(Full Buy möglich) = {p:.0%}  →  {verdict} (Konfidenz {max(p, 1 - p):.0%})")


if __name__ == "__main__":
    main()
