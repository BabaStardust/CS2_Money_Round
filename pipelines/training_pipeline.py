"""Training Pipeline: Feature Group → Feature View → Training Dataset → Modell → Model Registry.

Schritte (gemäss Aufgabenstellung):
  1. Feature Group round_buy_state laden
  2. Feature View erstellen (Features + Label full_buy_possible)
  3. Training Dataset aus der Feature View erstellen – zeitlicher Split nach ganzen Matches:
     die ältesten ~80 % der Matches sind Train, die neuesten ~20 % Test
  4. Modell trainieren und evaluieren (logistische Regression, Vergleich mit Mehrheitsklasse)
  5. Modell lokal speichern (joblib) und in die Model Registry hochladen

Aufruf:
    uv run python pipelines/training_pipeline.py                        # FEATURE_STORE aus .env
    FEATURE_STORE=local uv run python pipelines/training_pipeline.py    # lokaler Featurestore
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cs2_money_round.features import AGG_FEATURES, FEATURES, LABEL, RT_FEATURES  # noqa: E402
from cs2_money_round.specs import MODEL_NAME, ROUNDS_FG, TRAINING_FV  # noqa: E402
from cs2_money_round.store import store_from_env  # noqa: E402

MODEL_DIR = ROOT / "models" / MODEL_NAME
TEST_FRACTION = 0.2


def make_model() -> Pipeline:
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def split_time(match_times: pd.Series) -> pd.Timestamp:
    """Erste Match-Zeit des Testteils: die neuesten ~TEST_FRACTION der Matches gehen ins Testset."""
    ordered = match_times.drop_duplicates().sort_values().reset_index(drop=True)
    n_test = max(1, round(len(ordered) * TEST_FRACTION))
    return ordered.iloc[len(ordered) - n_test]


def main() -> None:
    load_dotenv(ROOT / ".env")
    store = store_from_env()

    print(f"1. Feature Group laden [{store.kind}]")
    keys = store.read_group(ROUNDS_FG, ["match_id", "match_time"])
    print(f"   {ROUNDS_FG.name} v{ROUNDS_FG.version}: {len(keys)} Zeilen, {keys['match_id'].nunique()} Matches")

    print("2./3. Feature View + Training Dataset (zeitlicher Split nach Match)")
    split = split_time(keys.groupby("match_id")["match_time"].first())
    start, end = keys["match_time"].min(), keys["match_time"].max() + pd.Timedelta(seconds=1)
    X_train, X_test, y_train, y_test = store.train_test_split(TRAINING_FV, split, start, end)
    X_train, X_test = X_train[FEATURES], X_test[FEATURES]
    y_train, y_test = y_train[LABEL].astype(int), y_test[LABEL].astype(int)
    n_train_m = keys[keys["match_time"] < split]["match_id"].nunique()
    n_test_m = keys[keys["match_time"] >= split]["match_id"].nunique()
    print(f"   Split bei {split:%d.%m.%Y %H:%M} UTC → Train {len(X_train)} Zeilen ({n_train_m} Matches), "
          f"Test {len(X_test)} Zeilen ({n_test_m} Matches)")

    print("4. Modell trainieren und evaluieren")
    model = make_model().fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    majority = int(y_train.mean() >= 0.5)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, proba >= 0.5)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "log_loss": round(float(log_loss(y_test, proba)), 4),
        "brier": round(float(brier_score_loss(y_test, proba)), 4),
        "baseline_accuracy": round(float((y_test == majority).mean()), 4),
        "baseline_brier": round(float(brier_score_loss(y_test, np.full(len(y_test), y_train.mean()))), 4),
        "label_rate_train": round(float(y_train.mean()), 4),
        "label_rate_test": round(float(y_test.mean()), 4),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "split_time": f"{split:%Y-%m-%dT%H:%M:%SZ}",
    }
    coefs = dict(zip(FEATURES, model[-1].coef_[0].round(4).tolist()))
    for k, v in metrics.items():
        print(f"   {k:<18} {v}")
    print("   Koeffizienten (standardisiert, + = eher Full Buy möglich):")
    for k, v in sorted(coefs.items(), key=lambda kv: -abs(kv[1])):
        kind = "RT " if k in RT_FEATURES else "AGG" if k in AGG_FEATURES else "   "
        print(f"     {kind} {k:<20} {v:+.4f}")

    print("5. Modell lokal speichern und registrieren")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    # joblib/pickle: das Modell stammt aus der eigenen Registry (vertrauenswürdige Quelle)
    joblib.dump(model, MODEL_DIR / "model.pkl")
    (MODEL_DIR / "metadata.json").write_text(json.dumps(
        {"features": FEATURES, "rt_features": RT_FEATURES, "agg_features": AGG_FEATURES, "label": LABEL,
         "metrics": metrics, "coefficients": coefs,
         "feature_view": {"name": TRAINING_FV.name, "version": TRAINING_FV.version}}, indent=2) + "\n")
    version = store.save_model(
        MODEL_NAME, MODEL_DIR, metrics,
        description="Logistische Regression: kann das Team diese Runde einen Full Buy (Rifle + Kevlar/Helm) kaufen?",
        input_example=X_train.head(1), view=TRAINING_FV,
    )
    print(f"   [{store.kind}] {MODEL_NAME} v{version} gespeichert")


if __name__ == "__main__":
    main()
