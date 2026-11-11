# AGENTS.md – CS2_Money_Round

Einstieg für Coding-Agenten: Was das Projekt tut, wie man es startet, wo was liegt und was
man nicht brechen darf. Ausführliche Begründungen, Messwerte und Reflexion stehen im
`README.md`. Diese Datei fasst nur zusammen und verweist darauf.

## Worum es geht

CAS MLOps Projektarbeit (FHNW, Testat). FTI-Pipelines (Feature, Training, Inference) mit
Hopsworks als Featurestore und Model Registry.

- **Frage.** Kann das *gegnerische* Team diese Runde einen Full Buy kaufen? Gemeint sind
  Rifle plus Kevlar mit Helm für alle fünf Spieler (T $3'700, CT $3'900).
- **Eingabe zur Inferenz.** Ein Screenshot vom Scoreboard (Tab) während der Kaufphase. Ein
  Vision-Modell liest ihn aus, eine Bildanalyse ergänzt die Lesung. Alternativ werden die
  Werte von Hand eingegeben.
- **Ausgabe.** P(Full Buy möglich) als Logistische Regression. Die Wahl des Modells ist im
  README begründet.
- **Daten.** 74 CS2-Demos: 23 eigene Premier-Matches und 51 HLTV-Pro-Maps. Daraus 2'715
  Team-Runden im Scope, mit einem zeitlichen Split nach ganzen Maps (Quellen in
  `docs/datenquellen.md`).
- **Stand.** Modell `full_buy_predictor` v5 erreicht Accuracy 0,749, ROC-AUC 0,828 und
  Brier 0,164; die Baseline liegt bei Accuracy 0,614. Das Modell läuft auf Feature Group
  `round_buy_state` v3 und `buy_rate_history` v1 sowie auf Feature View `full_buy_fv` v3 und
  `buy_rate_fv` v1.
- **Geprüft.** Alles läuft Ende-zu-Ende gegen das Hopsworks-Projekt `cs2_ml`.

## Sofort loslegen (ohne Hopsworks, ohne Demos)

Voraussetzungen: Python 3.12 (Hopsworks verlangt < 3.14) und `uv`. Alle Versionen sind in
`uv.lock` festgehalten.

```bash
uv sync
uv run python -m unittest discover -s tests -v      # alle Tests, ohne Netz und ohne Keys
FEATURE_STORE=example uv run python pipelines/inference_pipeline.py --team T --history WeWeLbLe --kills 21 --deaths 23
uv run uvicorn app.server:app --port 8765           # Web-App → http://127.0.0.1:8765
```

- **Ohne Zugangsdaten.** Der Standard ist der schreibgeschützte Schnappschuss `examples/store`
  mit Modell v5 und Kaufquoten. Damit läuft die Inferenz ohne `.env`.
- **Für Screenshots braucht es ein Vision-Modell**, eines von diesen drei:
  - die `claude`-CLI im PATH,
  - `ANTHROPIC_API_KEY` in `.env`,
  - Granite lokal: `uv sync --extra granite`, dann
    `uv run --extra granite uvicorn app.server:app --port 8765`. Beim ersten Lauf werden rund
    6 GB geladen; auf dem Mac läuft es über MPS.
- **Beispielbilder.** Die Web-App bietet die vier Bilder aus `examples/screenshots/` direkt
  zur Auswahl an.
- **Port belegt?** Einen anderen Port nehmen und keine fremden Prozesse beenden.

## Vollständig mit Hopsworks

```bash
cp .env.example .env        # HOPSWORKS_API_KEY, HOPSWORKS_PROJECT, VISION_PROVIDER eintragen
uv run python tools/setup_hopsworks.py
# Demos (.dem / .dem.bz2) nach data/raw/demos/, optional upload_times.csv (datei,last_modified)
uv run python pipelines/feature_pipeline.py --dry-run   # nur rechnen, Vorschau als CSV
uv run python pipelines/feature_pipeline.py             # Feature Groups befüllen
uv run python pipelines/training_pipeline.py            # trainieren, evaluieren, registrieren
uv run python pipelines/inference_pipeline.py pfad/zum/screenshot.png
```

- **Quelle umschalten.** `FEATURE_STORE=hopsworks|local|example` gilt für alle Pipelines.
  - `local` legt Parquet unter `data/featurestore/` ab und Modelle unter `models/registry/`.
  - `example` ist nur für die Inferenz da.
- **Web-App.** Sie hat einen eigenen Schalter oben auf der Seite.
  - Standard ist `example`. `APP_FEATURE_STORE` überschreibt den Standard.
  - Hopsworks erscheint nur, wenn `HOPSWORKS_API_KEY` gesetzt ist.
  - `local` erscheint nur, wenn `data/featurestore` und `models/registry` existieren.
- **Nach einem neuen Training**, das ins Repo soll, den Schnappschuss neu erzeugen:
  `uv run python tools/export_example_store.py`.
- **Wiederholbar.** Die Feature Pipeline cacht je Demo unter `data/interim/rounds/` und macht
  Upserts. Ein erneuter Lauf ist gefahrlos. Bei sporadischen Hopsworks-Fehlern (HDFS) genügt
  es, ihn zu wiederholen.

## Wo was liegt

| Pfad | Inhalt |
|---|---|
| `pipelines/feature_pipeline.py` | Demos parsen → Features → FG `round_buy_state` + Online-FG `buy_rate_history` |
| `pipelines/training_pipeline.py` | Feature View, zeitlicher Split, Training gegen Baseline, Registry |
| `pipelines/inference_pipeline.py` | CLI-Inferenz (Screenshot, gespeicherte Lesung oder `--team/--history/--kills/--deaths`) |
| `cs2_money_round/demo_import.py` | Demo → Rohtabelle. Spalten `obs_*` sind auf dem Scoreboard sichtbar, `gt_*` nur fürs Label, `post_*` nur für Analysen |
| `cs2_money_round/features.py` | **einzige** Feature-Funktion für Training und Inferenz (`RT_FEATURES`, `AGG_FEATURES`, `LABEL`) |
| `cs2_money_round/buy.py`, `model.py`, `simulation.py` | Economy-Regeln; das Label kommt aus `buy.evaluate` |
| `cs2_money_round/specs.py` | Namen und Versionen aller Feature Groups, Feature Views und des Modells |
| `cs2_money_round/store.py` | `HopsworksStore`, `LocalStore`, `ExampleStore` (read-only), `make_store`, `available_stores` |
| `cs2_money_round/inference.py` | gemeinsame Inferenz für CLI und Web-App: `observation`, `load_model`, `predict` |
| `cs2_money_round/vision/` | `schema.py` (Pydantic-Lesung), `readers.py` (Provider, `enhance`), `granite.py` (Ausschnitte + Plausibilitätsprüfung), `regions.py` (Panel, feste Bereiche, eigene Zeile), `history_bar.py` + `history_icons.json` (Rundenhistorie per Bildanalyse) |
| `app/server.py`, `app/static/` | FastAPI + statisches HTML/CSS/JS im FHNW-Look (mobil zuerst) |
| `examples/screenshots/` | 4 Demo-Screenshots, zugeschnitten und mit verpixelten Namen, plus `truth.json` |
| `examples/store/` | Schnappschuss von Modell und Kaufquoten aus Hopsworks |
| `tools/` | `eval_screenshots.py` (Lesequalität messen), `learn_history_icons.py`, `make_example_screenshots.py`, `screenshot_truth.py`, `export_example_store.py`, `setup_hopsworks.py` |
| `docs/` | `datenquellen.md` (Demo-Quellen), `economy-prototype.md` (Economy-CLI `cs2-money-round evaluate/estimate/simulate`) |
| `data/`, `models/` | lokal, **nicht im Repo**: Demos, Caches, Lern-Screenshots, lokale Registry, gespeicherte Lesungen (`data/inference/`) |

## Ablauf der Inferenz (Kurzform)

1. **Panel finden.** `vision/regions.py` sucht die dunkle, halbtransparente Fläche mit
   Seitenverhältnis 1,78. Alle weiteren Bereiche liegen relativ zu ihr; ein Zuschnitt des
   Bilds ist damit egal.
2. **Gegnerseite bestimmen.** Die eigene Zeile ist farbig hervorgehoben (blau auf CT, gelb
   auf T); das Gegnerteam ist das andere.
3. **Rundenhistorie lesen.** `history_bar.py` liest sie ohne Sprachmodell aus 24 festen Slots:
   - Die Strichfarbe zeigt den Sieger.
   - Das Symbol zeigt den Grund, erkannt per Nächster-Nachbar-Vergleich mit gelernten Vorlagen.
   - Ob das Symbol über oder unter dem Strich steht, zeigt die Seite.
4. **Score und Zahlen lesen.** Das Vision-Modell liest Score, Kills und Deaths.
   - Granite bekommt nur vergrösserte Ausschnitte.
   - `check_rows` prüft die Werte auf Plausibilität und lässt fehlerhafte Spalten einzeln
     nachlesen.
   - `readers.enhance` lässt die Leiste die Historie und den Score überschreiben, wenn sie
     zueinander passen.
5. **Features berechnen und vorhersagen.**
   - `inference.observation` baut die Features.
   - `predict` holt `hist_full_buy_rate` aus `buy_rate_fv` (online) und das neueste Modell
     aus der Registry.

**Historien-Code** (Sicht Gegnerteam): `W`/`L` für Sieg oder Niederlage, dann der Grund:
`e` alle tot, `b` Bombe, `d` entschärft, `t` Zeit, `?` unbekannt. Beispiel: `WeLbLeWd`.

## Regeln (nicht brechen)

- **Abhängigkeiten.** Python 3.12 über uv. `demoparser2==0.42.0` bleibt gepinnt, weil 0.42.1
  kein Wheel für macOS-arm64 hat.
- **Keine versteckten Werte als Feature.** Nie `gt_*`, `post_*` oder Geld als Feature
  verwenden: Live ist das Geld der Gegner unsichtbar. Neue Features nur in `features.py`,
  damit Training und Inferenz identisch rechnen.
- **Teams über Spieler-IDs zuordnen, nie über die Seite.** Zur Halbzeit wechseln die Seiten.
  Coaches (Name beginnt mit «coach») werden herausgefiltert.
- **Kills und Deaths nach Zeitpunkt zählen**, ab dem Start von Runde 1. Tode nach dem
  Rundenende zählen sofort, so wie auf dem Scoreboard.
- **Bei Feature- oder Schemaänderungen die Version in `specs.py` erhöhen.** Sonst mischt der
  Upsert alte und neue Zeilen.
- **Nie committen:** `.env`, `data/`, `models/`. Dort liegen Spielernamen und Secrets.
  - Screenshots kommen nur über `tools/make_example_screenshots.py` ins Repo, das die Namen
    verpixelt.
  - Die `.gitignore`-Einträge `/data/` und `/models/` sind absichtlich verankert, damit
    `examples/store/` im Repo bleibt.
- **Hopsworks.** Nur in das Projekt aus `HOPSWORKS_PROJECT` schreiben. Ohne Freigabe des
  Owners nie in ein anderes Projekt.
- **Demo-Archive** (`.rar`, `.bz2` aus dem Netz) sind nicht vertrauenswürdig:
  - jedes in ein neues, leeres Verzeichnis entpacken,
  - nichts daraus ausführen,
  - Python, das sie liest, mit `-I` starten.
- **Neue Lern-Screenshots:**
  1. Die Wahrheit in `data/screenshots/truth.json` eintragen; `tools/screenshot_truth.py`
     liest sie aus der Demo.
  2. Danach `tools/learn_history_icons.py` laufen lassen.
  3. Zum Schluss `tools/eval_screenshots.py` laufen lassen. Ausgeliefert werden nur Bilder,
     die beim Lernen nicht dabei waren.
- **Dokumentation mitführen.** Neue Messwerte, Versionen oder Limitationen gehören ins
  `README.md`, in die Abschnitte «Modell» und «Reflexion und Limitationen». Zahlen nur aus
  einem tatsächlichen Lauf übernehmen.

## Tests

```bash
uv run python -m unittest discover -s tests -v
```

Die Tests laufen ohne Netz, ohne Keys und ohne Demos.

**Querschnitts-Ring:**

- `tests/test_pipeline.py`: Halbzeit-Regression, Coach-Filter, Tode nach dem Rundenende,
  Point-in-time-Kaufquote, LocalStore.
- `tests/test_vision.py`: Panel, Hervorhebung, Rundenhistorie (auch bei halber Auflösung),
  Parsing und Plausibilität von Granite mit einem Fake-Modell, `enhance`.
- `tests/test_app.py`: Inferenz mit dem Beispiel-Store, read-only, Store-Schalter,
  Fehlermeldungen der API.

**Weitere Tests:** `test_economy.py` (Regeln, Label) und `test_cli.py` (Economy-CLI).

**Lesequalität mit echtem Modell:** `uv run --extra granite python tools/eval_screenshots.py`
(oder `--provider claude_cli`). Das ist langsam, rund 20–35 s pro Bild mit Granite.

## Bekannte Grenzen

Die Details stehen im README unter «Reflexion und Limitationen».

- Kalibriert ist die Bildanalyse an 12 Screenshots aus zwei Matches: 5120 × 1440 und die
  Demo-Ansicht. Andere Auflösungen, HUD-Grössen oder das Live-Scoreboard können neue
  Farbschwellen brauchen.
- Das Symbol «Zeit abgelaufen» fehlt in den Lerndaten. Solche Runden erscheinen als `?`, der
  Sieger stimmt trotzdem.
- Das Testset besteht wegen des zeitlichen Splits nur aus Pro-Maps (16 Maps). Die Metriken
  schwanken entsprechend.
- Das Modell sagt «kann kaufen» vorher, nicht «wird kaufen». Overtime ist nicht abgedeckt.

Retros des Owners (lokal, nicht im Repo): `~/.agents/retro/cs2_money_round/`
