"""Namen und Versionen der Feature Groups, Feature Views und des Modells (eine Quelle für alle Pipelines)."""

from .features import FEATURES, LABEL, RATE_KEYS
from .store import FeatureGroupSpec, FeatureViewSpec

ROUNDS_FG = FeatureGroupSpec(
    name="round_buy_state",
    # v1: 23 eigene Matches, gt_* als double; v2: + HLTV-Pro-Maps, Int64-Typen;
    # v3: Kills/Deaths nach Zeitpunkt gezählt (Tode nach dem Rundenende zählen wie auf dem Scoreboard)
    version=3,
    description="Eine Zeile pro Team und Runde zu Beginn der Kaufphase (CS2-Demos): "
                "RT-Features vom Scoreboard, aggregierte Features aus Rundenhistorie und früheren Matches, "
                "Label full_buy_possible aus dem versteckten Geld/Equipment.",
    primary_key=["match_id", "round_number", "team_side"],
    event_time="match_time",
)

RATES_FG = FeatureGroupSpec(
    name="buy_rate_history",
    version=1,
    description="Historische Quote 'Full Buy möglich' je Seite und Loss-Bonus-Stufe über alle Matches (für die Inferenz).",
    primary_key=RATE_KEYS,
    event_time="updated_at",
    online=True,
)

TRAINING_FV = FeatureViewSpec(
    name="full_buy_fv",
    version=3,
    description="Features (RT + aggregiert) und Label full_buy_possible",
    group=ROUNDS_FG,
    columns=FEATURES,
    labels=[LABEL],
)

RATES_FV = FeatureViewSpec(
    name="buy_rate_fv",
    version=1,
    description="Neueste historische Kaufquote per (team_side, loss_bonus_level)",
    group=RATES_FG,
    columns=["hist_full_buy_rate", "n_rounds", "updated_at"],
    labels=[],
)

MODEL_NAME = "full_buy_predictor"
