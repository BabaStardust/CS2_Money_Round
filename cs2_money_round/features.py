"""Feature Engineering für die Full-Buy-Vorhersage des Gegnerteams.

Eine Zeile = ein Team zu Beginn der Kaufphase einer Runde. Alle Features sind aus
dem Live-Scoreboard während der Kaufphase ableitbar (Spielstand, Rundenhistorie,
kumulierte Kills/Deaths). Versteckte Werte (Geld, Ausrüstung) gehen nur ins Label.

Dieselbe Funktion :func:`compute_features` rechnet die Features für das Training
(aus Demos) und für die Inferenz (aus einem Screenshot), damit beide identisch sind.
"""

from __future__ import annotations

import pandas as pd

from .model import Rules

HALF_ROUNDS = 12

# RT = erst zum Inferenz-Zeitpunkt bekannt (aktuelles Scoreboard).
RT_FEATURES = [
    "round_in_half",
    "is_ct",
    "score_team",
    "score_opp",
    "prev_round_won",
    "prev_reason_bomb",
    "prev_reason_defuse",
    "prev_reason_time",
]
# AGG = über mehrere Runden aggregiert (Rundenhistorie, kumulierte Stats) bzw.
# über frühere Matches (hist_full_buy_rate, kommt aus der Feature Group buy_rate_history).
AGG_FEATURES = [
    "loss_streak",
    "loss_bonus_level",
    "wins_last_3",
    "deaths_per_round",
    "kill_death_diff",
    "hist_full_buy_rate",
]
FEATURES = RT_FEATURES + AGG_FEATURES
LABEL = "full_buy_possible"

# Schlüssel der historischen Kaufquote und Glättung (Beta-Prior mit PRIOR_WEIGHT Pseudo-Runden).
RATE_KEYS = ["team_side", "loss_bonus_level"]
PRIOR_RATE = 0.5
PRIOR_WEIGHT = 10


def parse_history(history: str) -> list[tuple[bool, str]]:
    """'WeLtWd' → [(True,'e'), (False,'t'), (True,'d')]; Grund: e/b/d/t/?"""
    return [(history[i] == "W", history[i + 1]) for i in range(0, len(history), 2)]


def history_features(history: str, round_number: int, rules: Rules = Rules()) -> dict:
    """Aggregierte Features aus der Rundenhistorie der aktuellen Halbzeit.

    Loss-Bonus nach dem Startprofil in ``Rules``: Halbzeitstart auf
    ``initial_loss_level``, Niederlage +1 (max. ``max_loss_level``), Sieg −1 (min. 0).
    """
    rounds = parse_history(history)
    half_start = 0 if round_number <= HALF_ROUNDS else HALF_ROUNDS
    half = rounds[half_start:]
    level = rules.initial_loss_level
    streak = 0
    for won, _ in half:
        if won:
            level = max(0, level - 1)
            streak = 0
        else:
            level = min(rules.max_loss_level, level + 1)
            streak += 1
    return {
        "loss_streak": streak,
        "loss_bonus_level": level,
        "wins_last_3": sum(won for won, _ in half[-3:]),
    }


def compute_features(obs: pd.DataFrame) -> pd.DataFrame:
    """``obs_*``-Spalten (Demo-Import oder Screenshot) → Modell-Features (ohne hist_full_buy_rate).

    Erwartet: round_number, team_side, obs_round_history, obs_score_team, obs_score_opp,
    obs_prev_round_won, obs_prev_round_reason, obs_team_deaths_total, obs_team_kills_total.
    """
    df = obs.copy()
    hist = pd.DataFrame(
        [history_features(h, r) for h, r in zip(df["obs_round_history"], df["round_number"])], index=df.index
    )
    df = df.join(hist)
    played = (df["round_number"] - 1).clip(lower=1)
    df["round_in_half"] = ((df["round_number"] - 1) % HALF_ROUNDS) + 1
    df["is_ct"] = (df["team_side"] == "CT").astype(int)
    df["score_team"] = df["obs_score_team"].astype(int)
    df["score_opp"] = df["obs_score_opp"].astype(int)
    df["prev_round_won"] = df["obs_prev_round_won"].astype(int)
    for reason in ("bomb", "defuse", "time"):
        df[f"prev_reason_{reason}"] = (df["obs_prev_round_reason"] == reason).astype(int)
    df["deaths_per_round"] = df["obs_team_deaths_total"] / played
    df["kill_death_diff"] = (df["obs_team_kills_total"] - df["obs_team_deaths_total"]).astype(int)
    return df


def in_scope(rounds: pd.DataFrame) -> pd.DataFrame:
    """Reguläre Runden ohne Pistolenrunden (1, 13) und Overtime, nur mit Label."""
    keep = (rounds["round_number"] <= 2 * HALF_ROUNDS) & ~rounds["round_number"].isin([1, HALF_ROUNDS + 1])
    return rounds[keep & rounds[LABEL].notna()]


def historical_rates(df: pd.DataFrame) -> pd.Series:
    """Point-in-time-Kaufquote je (Seite, Loss-Bonus-Stufe) aus allen Matches VOR dem jeweiligen Match.

    Pro Match nur Matches mit kleinerer ``match_time`` → kein Leakage aus der Zukunft
    und keines aus dem eigenen Match. Geglättet mit einem Beta-Prior.
    """
    out = pd.Series(index=df.index, dtype=float)
    match_times = df.groupby("match_id")["match_time"].first().sort_values()
    for match_id, t in match_times.items():
        past = df[df["match_time"] < t]
        stats = past.groupby(RATE_KEYS)[LABEL].agg(["sum", "count"])
        rows = df.index[df["match_id"] == match_id]
        for idx in rows:
            key = tuple(df.loc[idx, RATE_KEYS])
            s, n = stats.loc[key] if key in stats.index else (0, 0)
            out[idx] = (s + PRIOR_RATE * PRIOR_WEIGHT) / (n + PRIOR_WEIGHT)
    return out


def current_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Aktuelle Kaufquote je (Seite, Loss-Bonus-Stufe) über alle Matches → Feature Group für die Inferenz."""
    grid = pd.MultiIndex.from_product([["CT", "T"], range(Rules().max_loss_level + 1)], names=RATE_KEYS)
    stats = df.groupby(RATE_KEYS)[LABEL].agg(["sum", "count"]).reindex(grid, fill_value=0)
    rates = stats.assign(
        hist_full_buy_rate=(stats["sum"] + PRIOR_RATE * PRIOR_WEIGHT) / (stats["count"] + PRIOR_WEIGHT),
        n_rounds=stats["count"].astype(int),
    )
    out = rates[["hist_full_buy_rate", "n_rounds"]].reset_index()
    out["loss_bonus_level"] = out["loss_bonus_level"].astype(int)
    out["updated_at"] = df["match_time"].max()
    return out


def build_features(rounds: pd.DataFrame) -> pd.DataFrame:
    """Rohtabelle aus ``demo_import`` (mit ``match_time``) → Feature-Tabelle für die Feature Group."""
    df = compute_features(in_scope(rounds))
    df["hist_full_buy_rate"] = historical_rates(df)
    df[LABEL] = df[LABEL].astype(int)
    keys = ["match_id", "round_number", "team_side", "map_name", "match_time"]
    extra = ["gt_team_cash", "gt_min_cash", "gt_max_equipped_players", "post_rifle_armor_players"]
    return df[keys + FEATURES + [LABEL] + extra].reset_index(drop=True)
