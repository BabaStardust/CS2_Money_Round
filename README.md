# CS2 Money Round – kann der Gegner diese Runde voll kaufen?

CAS MLOps Projektarbeit: drei Pipelines (Feature, Training, Inference) mit Hopsworks als
Featurestore und Model Registry. Während der Kaufphase macht man einen Screenshot vom
Scoreboard. Ein Vision-Modell liest ihn aus, und das Modell schätzt die Wahrscheinlichkeit,
dass das **gegnerische Team** einen **Full Buy** kaufen kann.

```
Demos (.dem) ──► Feature Pipeline ──► round_buy_state (FG) ──► Training Pipeline ──► full_buy_predictor (Registry)
                        │                                                                │
                        └──────────► buy_rate_history (FG, online) ◄──┐                  │
                                                                       │                  ▼
Screenshot ──► Vision-Modell ──► RT + Match-Aggregate ──────────► Inference Pipeline ──► P(Full Buy möglich)
```

## Daten

- **Quelle:** eigene CS2-Matches (Premier, MR12), zum Download angeboten auf Leetify
  (Links auf `replayNNN.valve.net`), als `.dem.bz2` gespeichert.
- **Stand:** 23 Matches vom 16.10. bis 06.11.2026 auf 7 Maps. Das ergibt 1'020 Team-Runden,
  davon 882 im Modellumfang.
- **Parser:** [`demoparser2`](https://github.com/LaihoE/demoparser) 0.42.0. Pro Runde liest
  er den Zustand beim Event `round_start` (Beginn der Kaufphase) sowie Rundenende, Kills und
  Bomb-Plants.
- **Event Time:** Upload-Zeit der Demo bei Valve (HTTP `Last-Modified`, ungefähr das Matchende).
  Sie wird in `data/raw/demos/upload_times.csv` geführt.
- **Historisch / aktuell:** Training mit den abgeschlossenen Matches. Die Inferenz nutzt
  einen Screenshot aus einem laufenden Match. Mangels laufender Matches beim Prüfen dient
  ein Demo-Screenshot als Ersatz.
- Demos und Screenshots enthalten Spielernamen und liegen deshalb **nicht im Repo**
  (`data/` steht in `.gitignore`).

Der Import trennt die Spalten streng nach Herkunft, damit kein verstecktes Wissen in die
Features gelangt:

| Präfix | Bedeutung | Verwendung |
|---|---|---|
| `obs_*` | live auf dem Scoreboard sichtbar (Score, Rundenhistorie, Kills/Deaths) | Features |
| `gt_*` | versteckte Wahrheit aus der Demo (Geld, gespeicherte Waffen, Rüstung) | nur Label |
| `post_*` | was nach der Kaufphase getragen wurde | nur Analyse |

## Target

`full_buy_possible` (0/1) bezieht sich auf ein Team zu Beginn der Kaufphase. Er ist 1, wenn
alle fünf Spieler mit dem vorhandenen Geld eine Rifle (AK-47, M4A4 oder M4A1-S) **und**
Kevlar mit Helm haben können.

- Ein Neukauf kostet T **$3'700** und CT **$3'900**.
- Gespeicherte Rifles und Rüstung zählen mit.
- Spieler dürfen einander Waffen droppen, Rüstung aber nicht.

Berechnet wird das von `cs2_money_round.buy.evaluate` mit `FULL_BUY_TARGET`. Pistolenrunden
(1, 13), Overtime und Runden mit weniger als fünf Spielern sind ausgeschlossen. Die
Label-Rate beträgt 59,5 %.

Gemeint ist, ob ein Full Buy *möglich* ist, nicht ob er tatsächlich gekauft wurde. Nur 17,5 %
der Team-Runden mit möglichem Full Buy enden die Kaufphase mit fünf Rifles und Rüstung, weil
Teams auch AWP, Utility oder Force Buys wählen.

## Features

Alle Features stammen aus Werten, die ein Spieler während der Kaufphase sehen kann. Eine
einzige Funktion (`cs2_money_round.features.compute_features`) rechnet sie für das Training
und die Inferenz, damit beide Seiten identisch sind.

| Feature | Art | Quelle bei der Inferenz | Bedeutung |
|---|---|---|---|
| `round_in_half` | RT | Screenshot (Score) | Runde in der Halbzeit (2–12) |
| `is_ct` | RT | Screenshot | Gegner ist CT (CT-Rifle ist teurer) |
| `score_team`, `score_opp` | RT | Screenshot | aktueller Spielstand aus Gegnersicht |
| `prev_round_won` | RT | Screenshot (Rundenhistorie) | Gegner hat die letzte Runde gewonnen |
| `prev_reason_bomb/defuse/time` | RT | Screenshot (Rundenhistorie) | wie die letzte Runde endete (Plant/Defuse bringen T-Geld) |
| `loss_streak` | AGG (Runden) | Rundenhistorie | Niederlagen in Folge |
| `loss_bonus_level` | AGG (Runden) | Rundenhistorie | Loss-Bonus-Stufe 0–4 nach Valve-Regel |
| `wins_last_3` | AGG (Runden) | Rundenhistorie | Siege in den letzten drei Runden der Halbzeit |
| `deaths_per_round` | AGG (Runden) | Summe Deaths auf dem Scoreboard | Tode pro bisheriger Runde |
| `kill_death_diff` | AGG (Runden) | Summe Kills − Deaths | Kill-Bilanz (Kill-Prämien) |
| `hist_full_buy_rate` | AGG (Matches) | **Featurestore** `buy_rate_history` | historische Quote «Full Buy möglich» je Seite und Loss-Bonus-Stufe über frühere Matches |

Im Training wird `hist_full_buy_rate` *point-in-time* berechnet: Jedes Match sieht nur Matches
mit früherer Event Time. Das eigene Match zählt nicht mit, damit das Label nicht durchsickert.
Geglättet wird mit einem Prior von 50 % und dem Gewicht von 10 Runden.

## Modell

Das Modell ist eine logistische Regression mit Standardisierung (scikit-learn). Sie liefert
eine kalibrierte Wahrscheinlichkeit, die die Inferenz als Konfidenz ausgibt.

Die Daten sind zeitlich nach ganzen Matches geteilt: Train sind die 18 ältesten Matches
(693 Zeilen), Test die 5 neuesten (189 Zeilen).

| Metrik (Test) | Modell | Baseline |
|---|---|---|
| Accuracy | 0,767 | 0,598 (immer «möglich») |
| ROC-AUC | 0,848 | 0,5 |
| Brier Score | 0,160 | 0,240 (Train-Rate als Konstante) |

Die stärksten Koeffizienten sind `prev_round_won`, `wins_last_3` und `loss_bonus_level`.
`hist_full_buy_rate` trägt fast nichts bei, weil die Loss-Bonus-Stufe dieselbe Information
schon enthält. Das Feature bleibt drin, weil es den Weg «aggregiertes Feature aus dem
Featurestore bei der Inferenz» zeigt.

## Pipelines

| Pipeline | Datei | Schritte |
|---|---|---|
| Feature | `pipelines/feature_pipeline.py` | 1. Demos parsen, inkrementell: Cache je Demo in `data/interim/rounds/` · 2. Features engineeren · 3. DataFrame mit Label · 4./5. Feature Groups `round_buy_state` (PK `match_id, round_number, team_side`, Event Time `match_time`) und `buy_rate_history` (PK `team_side, loss_bonus_level`, online) anlegen und befüllen |
| Training | `pipelines/training_pipeline.py` | 1. FG laden · 2. Feature View `full_buy_fv` · 3. Training Dataset mit zeitlichem Split per `train_test_split` · 4. Training und Evaluation gegen die Baseline · 5. `joblib` lokal speichern, dann `full_buy_predictor` in die Model Registry |
| Inference | `pipelines/inference_pipeline.py` | 1. Screenshot per Vision-Modell lesen, sonst gespeicherte Lesung oder manuelle Eingabe · 2. Feature View `buy_rate_fv` holen und neueste `hist_full_buy_rate` per Key laden · 3. neueste Modellversion aus der Registry · 4. P(Full Buy möglich) für das Gegnerteam |

Namen und Versionen aller Feature Groups, Views und des Modells stehen an einer Stelle:
`cs2_money_round/specs.py`.

**Screenshot → Features:**

- Das Gegnerteam ist das Team ohne hervorgehobene Zeile. Die hervorgehobene Zeile ist der
  eigene Spieler.
- Die Rundenhistorie wird beim Seitenwechsel zur Halbzeit auf das Team umgerechnet.
- Geld zeigt das Scoreboard live nur fürs eigene Team. Darum ist Geld nie ein Feature.
- Bei einem Demo-Screenshot ist das Geld der Gegner sichtbar. Die Pipeline zeigt es dann als
  reine Kontrolle an.

**Vision-Modell (austauschbar)** über `VISION_PROVIDER`:

- `anthropic`: Claude über die API, mit strukturierter Ausgabe nach `ScoreboardReading`.
- `openai_compatible`: lokales Modell, z. B. oMLX, LM Studio oder Ollama, über
  `VISION_BASE_URL`.
- `claude_cli`: Entwicklung ohne API-Key über `claude -p`.

Alle drei liefern dasselbe Pydantic-Schema (`cs2_money_round/vision/schema.py`).

## Setup und Ausführen

Voraussetzungen: Python 3.12 (Hopsworks verlangt < 3.14) und
[uv](https://docs.astral.sh/uv/). Alle Abhängigkeiten sind exakt in `uv.lock` festgehalten.

```bash
git clone git@github.com:BabaStardust/CS2_Money_Round.git && cd CS2_Money_Round
uv sync                           # legt .venv mit Python 3.12 an
cp .env.example .env              # HOPSWORKS_API_KEY, HOPSWORKS_PROJECT, Vision-Provider eintragen
uv run python tools/setup_hopsworks.py   # Projekt anlegen/prüfen (einmalig)
```

Demos: `.dem` oder `.dem.bz2` nach `data/raw/demos/` legen. Optional gehört in
`data/raw/demos/upload_times.csv` (`datei,last_modified`) die Upload-Zeit. Fehlt sie, dient
die Dateizeit als Ersatz.

```bash
uv run python pipelines/feature_pipeline.py --dry-run   # nur rechnen, Vorschau als CSV
uv run python pipelines/feature_pipeline.py             # Feature Groups befüllen
uv run python pipelines/training_pipeline.py            # Modell trainieren + registrieren
uv run python pipelines/inference_pipeline.py data/screenshots/runde.png
uv run python pipelines/inference_pipeline.py --reading data/inference/runde.reading.json --history WeLtWd
uv run python pipelines/inference_pipeline.py --team T --history WeWeLbLe --kills 21 --deaths 23   # ohne Screenshot
```

`FEATURE_STORE=local` schaltet ohne Hopsworks auf einen lokalen Ersatz um, und zwar für alle
drei Pipelines gleich:

- Feature Groups werden als Parquet unter `data/featurestore/` gespeichert.
- Modelle landen versioniert unter `models/registry/`.

Tests: `uv run python -m unittest discover -s tests -v`

## Reflexion und Limitationen

- **Hopsworks.**
  - Die Pipelines laufen seit 10.11.2026 Ende-zu-Ende gegen das Hopsworks-Projekt `cs2_ml`:
    zwei Feature Groups, Feature Views `full_buy_fv` und `buy_rate_fv`, Modell
    `full_buy_predictor` v1 in der Registry.
  - Der Free Tier erlaubt ein Projekt pro Account. Darum läuft die Arbeit in einem eigenen
    Account; dessen Registrierung hing am 09.11.2026 zuerst in einer Endlosschleife.
  - Beim ersten Insert in die frisch angelegte Online-Feature-Group `buy_rate_history` blieb der
    Online-Store leer, die Inferenz fand keine Kaufquote. Ein zweiter Insert füllte ihn. Die
    Feature Pipeline wartet seither nach dem Insert, bis die Zeilen online sind, und schiebt
    sie bei Bedarf einmal nach.
  - Der Client ist `hopsworks` 5.1.2, das Backend 5.2.2. Hopsworks warnt deshalb bei jedem
    Login; funktional gab es keine Probleme.
- **Wenig Daten.**
  - Es sind 23 Matches aus einem Spielerkreis und mehrheitlich Nuke und Dust2.
  - Das Testset umfasst nur 5 Matches, die Metriken schwanken entsprechend.
  - Mehr Demos (FACEIT, HLTV) würden helfen.
- **«Möglich» ≠ «gekauft».** Für die eigene Taktik zählt, ob der Gegner kaufen *kann*. Ob er
  tatsächlich kauft, sagt das Modell nicht vorher.
- **Rundenhistorie auf dem Screenshot.**
  - Bisher ist erst ein Screenshot geprüft, aus Runde 2 einer Demo, in zwei Läufen.
  - Geld, Kills, Deaths und Assists hat das Vision-Modell beide Male richtig gelesen.
  - Die kleine Symbolleiste der Rundenhistorie hat es nur in einem der zwei Läufe erkannt.
  - Ohne vollständige Historie bricht die Inferenz mit klarer Meldung ab. Die Historie lässt
    sich dann mit `--history` nachreichen.
  - Screenshots späterer Runden mit langer Historie sind noch zu prüfen.
  - Für diesen Screenshot sagt das Modell 55 % «möglich», laut Demo war es nicht möglich: Ein
    T-Spieler hatte nur $3'400. Runde 2 nach gewonnener Pistolenrunde ist ein typischer
    Grenzfall.
- **Regelprofil.**
  - Preise und Loss-Bonus stammen aus einem Startprofil (`Rules`, Valve-Stand 2025).
  - Sonderfälle fehlen: Teamkills, Suizide, Disconnects und Kaufzeit-Ende.
  - Overtime wird nicht vorhergesagt.
- **Demo-Import-Fehler gefunden und behoben.**
  - Die erste Version ordnete Score, Kills und Rundenhistorie nach der *Seite* zu. Nach dem
    Halbzeitwechsel war das falsch.
  - Jetzt geschieht die Zuordnung über die Spieler-IDs und wird gegen `team_rounds_total`
    der Demo geprüft (0 Abweichungen).
  - Ein Regressionstest liegt in `tests/test_pipeline.py`.
- **Event Time** ist die Upload-Zeit der Demo und nicht die exakte Rundenzeit. Für den
  Split nach ganzen Matches reicht das.

## Weiteres

- `docs/economy-prototype.md` beschreibt den ursprünglichen Economy-Prototyp (CLI
  `evaluate` / `simulate` / `estimate`). Das Label baut auf dessen `evaluate` auf.
- `tools/setup_hopsworks.py` legt das Hopsworks-Projekt an.
