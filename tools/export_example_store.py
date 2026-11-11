"""Schnappschuss für das Repo: neuestes Modell und ``buy_rate_history`` → ``examples/store``.

Damit laufen Inference Pipeline und Web-App auch ohne Hopsworks-Zugang (``FEATURE_STORE=example``).
Exportiert wird nur, was die Inferenz braucht und keine Spielerdaten enthält:

- Modell ``full_buy_predictor`` (Pipeline aus StandardScaler + LogisticRegression, Kennzahlen)
- Feature Group ``buy_rate_history``, neuester Stand (10 Zeilen: Seite, Loss-Bonus-Stufe, Quote, Anzahl Runden)

Nicht exportiert wird ``round_buy_state`` (Trainingsdaten mit Match-IDs).

    uv run python tools/export_example_store.py          # aus FEATURE_STORE (Standard: hopsworks)
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cs2_money_round.specs import MODEL_NAME, RATES_FG  # noqa: E402
from cs2_money_round.store import EXAMPLE_ROOT, ExampleStore, store_from_env  # noqa: E402


def main() -> None:
    load_dotenv(ROOT / ".env")
    source = store_from_env()
    if source.kind == "example":
        sys.exit("Quelle ist schon der Beispiel-Store – FEATURE_STORE=hopsworks oder local setzen.")
    handle = source.latest_model(MODEL_NAME)
    rates = source.read_group(RATES_FG, [*RATES_FG.primary_key, "hist_full_buy_rate", "n_rounds", RATES_FG.event_time])
    # der Offline-Store hält jeden Stand (Event Time); die Inferenz braucht nur den neuesten je Schlüssel
    rates = rates.sort_values(RATES_FG.event_time).drop_duplicates(RATES_FG.primary_key, keep="last")

    target = ExampleStore()
    if EXAMPLE_ROOT.exists():
        shutil.rmtree(EXAMPLE_ROOT)
    model_dir = target.registry / MODEL_NAME / f"v{handle.version}"
    model_dir.mkdir(parents=True)
    for name in ("model.pkl", "metadata.json"):
        shutil.copy2(handle.path / name, model_dir / name)
    meta = json.loads((handle.path / "metadata.json").read_text())
    (model_dir / "registry.json").write_text(json.dumps({
        "name": MODEL_NAME, "version": handle.version, "metrics": handle.metrics,
        "feature_view": meta.get("feature_view"), "quelle": source.kind,
    }, indent=2) + "\n")

    target.fs_dir.mkdir(parents=True)
    path = target._path(RATES_FG)
    rates.sort_values(["team_side", "loss_bonus_level"]).to_parquet(path, index=False)
    (target.fs_dir / f"{RATES_FG.name}_v{RATES_FG.version}.json").write_text(json.dumps(
        {"description": RATES_FG.description, "primary_key": RATES_FG.primary_key,
         "event_time": RATES_FG.event_time, "online": RATES_FG.online, "quelle": source.kind}, indent=2) + "\n")
    print(f"{MODEL_NAME} v{handle.version} und {RATES_FG.name} v{RATES_FG.version} ({len(rates)} Zeilen) "
          f"aus {source.kind} → {EXAMPLE_ROOT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
