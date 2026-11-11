# AGENTS.md – CS2_Money_Round

CAS MLOps Projektarbeit (Abgabe 15.11.2026 23:59, Testat): FTI-Pipelines mit Hopsworks.
Vorhersage: Kann das gegnerische Team diese Runde einen Full Buy kaufen (Rifle + Kevlar/Helm,
T $3'700 / CT $3'900)? Eingabe zur Inferenz ist ein Screenshot während der Kaufphase.

## Struktur

- `pipelines/` – Feature, Training, Inference (Einstiegspunkte, siehe README)
- `cs2_money_round/demo_import.py` – Demo → Rohtabelle (`obs_*` sichtbar, `gt_*` nur Label, `post_*` nur Analyse)
- `cs2_money_round/features.py` – einzige Feature-Funktion für Training und Inferenz
- `cs2_money_round/store.py` – Hopsworks, lokaler Ersatz oder Beispiel-Schnappschuss
  (`FEATURE_STORE=hopsworks|local|example`); `examples/store` neu mit `tools/export_example_store.py`
  nach jedem Training, das ins Repo soll
- `cs2_money_round/specs.py` – Namen/Versionen von Feature Groups, Views, Modell
- `cs2_money_round/inference.py` – gemeinsame Inferenz-Logik für CLI und Web-App
- `cs2_money_round/vision/` – austauschbare Vision-Reader (`VISION_PROVIDER`, inkl. Granite lokal);
  `regions.py` (Panel + feste Bereiche), `history_bar.py` (Rundenhistorie per Bildanalyse,
  Vorlagen aus `tools/learn_history_icons.py`)
- `app/` – Web-App (FastAPI + statisches HTML im FHNW-Look), `uv run uvicorn app.server:app`
- `examples/screenshots/` – ausgelieferte Demo-Screenshots, zugeschnitten und Namen verpixelt
  (`tools/make_example_screenshots.py`); Lernbilder bleiben in `data/screenshots/kalibrierung`
- `cs2_money_round/buy.py`, `model.py`, `simulation.py` – Economy-Regeln, Label über `evaluate`

## Regeln

- Python 3.12 via uv (`uv sync`); `demoparser2==0.42.0` bleibt gepinnt (0.42.1 hat kein macOS-arm64-Wheel).
- Nie `gt_*`/`post_*` oder Geld als Feature verwenden: live ist das Gegnergeld unsichtbar.
- Teams über Spieler-IDs zuordnen, nie über die Seite (Halbzeitwechsel).
- `data/`, `models/`, `.env` nie committen (Spielernamen, Secrets). Screenshots ins Repo nur
  über `tools/make_example_screenshots.py` (Namen verpixelt).
- Neue Lern-Screenshots: Wahrheit in `data/screenshots/truth.json`, dann
  `tools/learn_history_icons.py` und `tools/eval_screenshots.py` laufen lassen.
- Keine CS2-Daten ins Lichess-Hopsworks-Projekt schreiben ohne OK des Owners.
- Feature-Änderung → `specs.py`-Version erhöhen, sonst mischt der Upsert alte und neue Zeilen.

## Tests

`uv run python -m unittest discover -s tests -v`. Querschnitts-Ring: `tests/test_pipeline.py`
(Halbzeit-Regression, Point-in-time-Rate, LocalStore) und `tests/test_vision.py`
(Panel, Rundenhistorie und Hervorhebung an den Beispiel-Screenshots).

Retros: `~/.agents/retro/cs2_money_round/`
