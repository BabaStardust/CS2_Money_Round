"""Gemeinsame Inferenz-Logik für die Inference Pipeline (CLI) und die Web-App.

Ablauf (gemäss Aufgabenstellung):
  1. Live-Daten: Screenshot-Lesung oder manuelle Eingabe → ``observation`` (obs_*-Zeile wie im Demo-Import)
  2. Feature View holen → historische Kaufquote aus ``buy_rate_history`` (``predict``)
  3. Modell aus der Model Registry laden (``load_model``)
  4. Prediction (``predict``)

Fehler in der Eingabe lösen :class:`InferenceError` aus; die Aufrufer zeigen die Meldung an.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import joblib
import pandas as pd

from .features import AGG_FEATURES, HALF_ROUNDS, RT_FEATURES, compute_features, parse_history
from .model import Rules
from .specs import MODEL_NAME, RATES_FV
from .vision.schema import ScoreboardReading

REASON_CODE = {"elimination": "e", "bomb": "b", "defuse": "d", "time": "t", None: "?"}
CODE_REASON = {"e": "elimination", "b": "bomb", "d": "defuse", "t": "time", "?": None}


class InferenceError(ValueError):
    """Eingabe reicht für eine Vorhersage nicht aus (Meldung ist für den User formuliert)."""


def half_of(round_number: int) -> int:
    """Halbzeit-Index: 0/1 regulär (je 12 Runden), danach Overtime-Hälften à 3 Runden."""
    if round_number <= 2 * HALF_ROUNDS:
        return (round_number - 1) // HALF_ROUNDS
    return 2 + (round_number - 2 * HALF_ROUNDS - 1) // 3


def side_in_round(side_now: str, round_now: int, past_round: int) -> str:
    """Seite, auf der das Team ``side_now`` in einer früheren Runde stand (Seitenwechsel je Halbzeit)."""
    flipped = (half_of(round_now) - half_of(past_round)) % 2 == 1
    return ("T" if side_now == "CT" else "CT") if flipped else side_now


def target_side(reading: ScoreboardReading | None, team: str | None) -> str | None:
    """Gegnerteam: explizit gewählt, sonst das Team ohne hervorgehobene Zeile (eigener Spieler)."""
    if team:
        return team
    if reading:
        own = [s for s, t in (("CT", reading.ct), ("T", reading.t)) if any(p.highlighted for p in t.players)]
        if len(own) == 1:
            return "T" if own[0] == "CT" else "CT"
    return None


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


def team_totals(reading: ScoreboardReading, side: str) -> tuple[int | None, int | None]:
    """Summe Kills/Deaths des Teams; None, wenn nicht alle fünf Zeilen lesbar sind."""
    team = reading.ct if side == "CT" else reading.t
    k = [p.kills for p in team.players]
    d = [p.deaths for p in team.players]
    if len(team.players) != 5:
        return None, None
    return (None if None in k else sum(k)), (None if None in d else sum(d))  # type: ignore[arg-type]


def validate_history(history: str) -> None:
    if len(history) % 2 or any(
        history[i] not in "WL" or history[i + 1] not in CODE_REASON for i in range(0, len(history), 2)
    ):
        raise InferenceError(
            "Rundenhistorie ungültig: je Runde W/L und Grund e/b/d/t/?, z. B. WeLtWd (aus Sicht des Gegnerteams)."
        )


def observation(
    side: str,
    history: str | None,
    kills: int | None,
    deaths: int | None,
    reading: ScoreboardReading | None = None,
) -> tuple[dict, list[str]]:
    """Live-Werte → obs_*-Zeile im selben Format wie der Demo-Import, dazu Hinweise für den User.

    Explizite Werte (``history``, ``kills``, ``deaths``) haben Vorrang vor der Bild-Lesung.
    """
    notes: list[str] = []
    team = opp = None
    if reading:
        team, opp = (reading.ct, reading.t) if side == "CT" else (reading.t, reading.ct)
    if history is None and reading and team and opp and team.score is not None and opp.score is not None:
        history = history_from_reading(reading, side, team.score + opp.score + 1)
        if history is None:
            raise InferenceError(
                "Rundenhistorie auf dem Bild unvollständig – bitte nachreichen "
                "(je Runde W/L + Grund e/b/d/t aus Sicht des Gegnerteams, z. B. WeLtWd)."
            )
    if history is None:
        raise InferenceError("Ohne lesbaren Screenshot ist die Rundenhistorie nötig (z. B. WeLtWd).")
    validate_history(history)
    parsed = parse_history(history)
    played = len(parsed)
    wins = sum(w for w, _ in parsed)
    if team and opp and team.score is not None and opp.score is not None and (team.score, opp.score) != (wins, played - wins):
        notes.append(f"Score laut Bild {team.score}:{opp.score} (Gegner {side}), laut Historie {wins}:{played - wins} – "
                     "die Historie gilt. Ist sie aus Sicht des Gegnerteams?")

    if reading and (kills is None or deaths is None):
        k, d = team_totals(reading, side)
        kills = kills if kills is not None else k
        deaths = deaths if deaths is not None else d
    if kills is None or deaths is None:
        raise InferenceError("Kills und Deaths des Gegnerteams (Summe der fünf Spieler) angeben.")
    if kills < 0 or deaths < 0:
        raise InferenceError("Kills und Deaths dürfen nicht negativ sein.")

    last = parsed[-1] if parsed else None
    obs = {
        "round_number": played + 1,
        "team_side": side,
        "obs_round_history": history,
        "obs_score_team": wins,
        "obs_score_opp": played - wins,
        "obs_prev_round_won": last[0] if last else False,
        "obs_prev_round_reason": CODE_REASON[last[1]] if last else None,
        "obs_team_kills_total": kills,
        "obs_team_deaths_total": deaths,
        "displayed_loss_bonus": team.loss_bonus_level if team else None,
    }
    return obs, notes


def demo_money_check(reading: ScoreboardReading, side: str, rules: Rules = Rules()) -> str | None:
    """Nur Demo-Ansicht: sichtbares Geld der Gegner gegen den Neukaufpreis (ohne gespeicherte Waffen)."""
    team = reading.ct if side == "CT" else reading.t
    money = [p.money for p in team.players]
    if not reading.is_demo_view or not money or None in money:
        return None
    need = rules.rifle_price(side) + rules.kevlar_price + rules.helmet_upgrade_price  # type: ignore[arg-type]
    ok = sum(m >= need for m in money)  # type: ignore[operator]
    return f"{ok}/5 Spieler haben ≥ ${need:,} für Rifle + Kevlar/Helm (gespeicherte Ausrüstung nicht sichtbar)"


@dataclass
class LoadedModel:
    model: object
    features: list[str]
    version: int
    metrics: dict


def load_model(store) -> LoadedModel:
    """Neueste Modellversion aus der Registry (Hopsworks oder lokal)."""
    handle = store.latest_model(MODEL_NAME)
    # joblib/pickle: das Modell stammt aus der eigenen Registry (vertrauenswürdige Quelle)
    model = joblib.load(handle.path / "model.pkl")
    features = json.loads((handle.path / "metadata.json").read_text())["features"]
    return LoadedModel(model, features, handle.version, handle.metrics)


@dataclass
class Prediction:
    side: str
    round_number: int
    score: tuple[int, int]
    history: str
    skipped: str | None = None
    probability: float | None = None
    rt: dict = field(default_factory=dict)
    agg: dict = field(default_factory=dict)
    hist_rate: dict | None = None
    model_version: int | None = None
    model_auc: float | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def possible(self) -> bool | None:
        return None if self.probability is None else self.probability >= 0.5

    @property
    def confidence(self) -> float | None:
        return None if self.probability is None else max(self.probability, 1 - self.probability)


def predict(obs: dict, store, model: LoadedModel | None = None, notes: list[str] | None = None) -> Prediction:
    """Schritte 2–4: Kaufquote aus dem Featurestore, Modell, Wahrscheinlichkeit."""
    rnd = obs["round_number"]
    out = Prediction(
        side=obs["team_side"], round_number=rnd, score=(obs["obs_score_team"], obs["obs_score_opp"]),
        history=obs["obs_round_history"], notes=list(notes or []),
    )
    if rnd in (1, HALF_ROUNDS + 1):
        out.skipped = "Pistolenrunde: Full Buy unmöglich (Startgeld). Keine Modellvorhersage nötig."
        return out
    if rnd > 2 * HALF_ROUNDS:
        out.skipped = "Overtime ist nicht im Trainingsumfang – keine Vorhersage."
        return out

    x = compute_features(pd.DataFrame([obs])).iloc[0].to_dict()
    if obs.get("displayed_loss_bonus") is not None and obs["displayed_loss_bonus"] != x["loss_bonus_level"]:
        out.notes.append(f"Loss Bonus laut Bild {obs['displayed_loss_bonus']}, aus Historie berechnet {x['loss_bonus_level']}.")

    entry = {"team_side": obs["team_side"], "loss_bonus_level": int(x["loss_bonus_level"])}
    rate = store.feature_vector(RATES_FV, entry)
    if rate is None:
        raise InferenceError(f"Kein Eintrag in {RATES_FV.group.name} für {entry} – zuerst die Feature Pipeline laufen lassen.")
    x["hist_full_buy_rate"] = float(rate["hist_full_buy_rate"])
    out.hist_rate = {
        "rate": float(rate["hist_full_buy_rate"]), "n_rounds": int(rate["n_rounds"]),
        "updated_at": pd.Timestamp(rate["updated_at"]), **entry,
    }

    model = model or load_model(store)
    X = pd.DataFrame([x])[model.features]
    out.probability = float(model.model.predict_proba(X)[0, 1])  # type: ignore[attr-defined]
    out.rt = {n: float(X.iloc[0][n]) for n in RT_FEATURES if n in model.features}
    out.agg = {n: float(X.iloc[0][n]) for n in AGG_FEATURES if n in model.features}
    out.model_version = model.version
    out.model_auc = model.metrics.get("roc_auc")
    return out
