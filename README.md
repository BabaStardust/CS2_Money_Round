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

- **Quellen:**
  - 23 eigene CS2-Matches (Premier, MR12) vom 16.10. bis 06.11.2026, zum Download angeboten
    auf Leetify (Links auf `replayNNN.valve.net`), als `.dem.bz2` gespeichert.
  - 51 Maps aus 21 Pro-Serien (Best-of-3, CCT, ESL Challenger League, European Pro League,
    NODWIN Clutch Series) von HLTV, als RAR-Archive geladen und entpackt. Gespielt bzw.
    bereitgestellt wurden sie vom 08. bis 10.11.2026.
- **Stand:** 74 gespielte Maps auf 7 verschiedenen Karten (Ancient, Anubis, Cache, Dust2, Inferno, Mirage, Nuke).
  Das ergibt 3'096 Team-Runden, davon 2'715 im Modellumfang (882 eigene, 1'833 Pro).
- **Parser:** [`demoparser2`](https://github.com/LaihoE/demoparser) 0.42.0. Pro Runde liest
  er den Zustand beim Event `round_start` (Beginn der Kaufphase) sowie Rundenende, Kills und
  Bomb-Plants.
- **Event Time:** Für eigene Matches die Upload-Zeit der Demo bei Valve (HTTP `Last-Modified`,
  ungefähr das Matchende), für HLTV-Demos die Dateizeit im Archiv. Beides steht in
  `data/raw/demos/upload_times.csv`.
- **Historisch / aktuell:** Training mit den abgeschlossenen Matches. Die Inferenz nutzt
  einen Screenshot aus einem laufenden Match. Mangels laufender Matches beim Prüfen dient
  ein Demo-Screenshot als Ersatz.
- Demos und Original-Screenshots enthalten Spielernamen und liegen deshalb **nicht im Repo**
  (`data/` steht in `.gitignore`). Im Repo liegen nur 4 zugeschnittene Beispiel-Screenshots mit
  verpixelten Namen (`examples/screenshots/`) und ein Schnappschuss von Modell und Kaufquoten
  (`examples/store/`, siehe Setup).
- **Nachbauen:** Alle 21 Pro-Serien mit Event und Maps stehen in
  [`docs/datenquellen.md`](docs/datenquellen.md); sie sind auf HLTV öffentlich.

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
- Gespeicherte Rifles und Rüstung zählen mit. Beschädigte Rüstung mit Helm gilt als
  ausgerüstet, sie muss nicht auf 100 aufgefüllt werden.
- Spieler dürfen einander Waffen droppen, Rüstung aber nicht.

Berechnet wird das von `cs2_money_round.buy.evaluate` mit `FULL_BUY_TARGET`. Pistolenrunden
(1, 13), Overtime und Runden mit weniger als fünf Spielern sind ausgeschlossen. Die
Label-Rate beträgt 60,1 % (eigene Matches 59,5 %, Pro-Maps 60,4 %).

Gemeint ist, ob ein Full Buy *möglich* ist, nicht ob er tatsächlich gekauft wurde. In den
eigenen Matches enden nur 17,5 % der Team-Runden mit möglichem Full Buy die Kaufphase mit fünf Rifles und Rüstung, weil
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

Die Daten sind zeitlich nach ganzen Maps geteilt: Train sind die 58 ältesten (2'174 Zeilen:
alle 23 eigenen Matches und 35 Pro-Maps), Test die 16 neuesten (541 Zeilen, alles Pro-Maps).
Das ist `full_buy_predictor` v5 in der Registry.

| Metrik (Test) | Modell | Baseline |
|---|---|---|
| Accuracy | 0,749 | 0,614 (immer «möglich») |
| ROC-AUC | 0,828 | 0,5 |
| Brier Score | 0,164 | 0,237 (Train-Rate als Konstante) |

Ein Gegencheck ausserhalb der Pipeline: Nur auf den 23 eigenen Matches trainiert und auf allen
51 Pro-Maps getestet, erreicht das Modell ROC-AUC 0,823 und Accuracy 0,751 (Baseline 0,604).
Das Muster der Economy überträgt sich also von Premier-Matches auf Pro-Matches.

Die stärksten Koeffizienten sind `wins_last_3`, `prev_round_won` und `loss_bonus_level`.
`hist_full_buy_rate` trägt wenig bei, weil die Loss-Bonus-Stufe dieselbe Information schon
grossteils enthält. Das Feature bleibt drin, weil es den Weg «aggregiertes Feature aus dem
Featurestore bei der Inferenz» zeigt.

## Pipelines

| Pipeline | Datei | Schritte |
|---|---|---|
| Feature | `pipelines/feature_pipeline.py` | 1. Demos parsen, inkrementell: Cache je Demo in `data/interim/rounds/` · 2. Features engineeren · 3. DataFrame mit Label · 4./5. Feature Groups `round_buy_state` (PK `match_id, round_number, team_side`, Event Time `match_time`) und `buy_rate_history` (PK `team_side, loss_bonus_level`, online) anlegen und befüllen |
| Training | `pipelines/training_pipeline.py` | 1. FG laden · 2. Feature View `full_buy_fv` · 3. Training Dataset mit zeitlichem Split per `train_test_split` · 4. Training und Evaluation gegen die Baseline · 5. `joblib` lokal speichern, dann `full_buy_predictor` in die Model Registry |
| Inference | `pipelines/inference_pipeline.py` und Web-App `app/server.py` | 1. Screenshot per Vision-Modell lesen, sonst gespeicherte Lesung oder manuelle Eingabe · 2. Feature View `buy_rate_fv` holen und neueste `hist_full_buy_rate` per Key laden · 3. neueste Modellversion aus der Registry · 4. P(Full Buy möglich) für das Gegnerteam |

Namen und Versionen aller Feature Groups, Views und des Modells stehen an einer Stelle:
`cs2_money_round/specs.py`. Kommandozeile und Web-App rufen dieselben Funktionen auf
(`cs2_money_round/inference.py`), rechnen also garantiert gleich.

**Screenshot → Features:**

- Das Gegnerteam ist das Team ohne hervorgehobene Zeile. Die hervorgehobene Zeile ist der
  eigene Spieler.
- Die Rundenhistorie wird beim Seitenwechsel zur Halbzeit auf das Team umgerechnet.
- Geld zeigt das Scoreboard live nur fürs eigene Team. Darum ist Geld nie ein Feature.
- Bei einem Demo-Screenshot ist das Geld der Gegner sichtbar. Die Pipeline zeigt es dann als
  reine Kontrolle an.

**Vision-Modell (austauschbar)** über `VISION_PROVIDER`:

- `anthropic`: Claude über die API, mit strukturierter Ausgabe nach `ScoreboardReading`.
- `claude_cli`: Claude ohne API-Key über `claude -p` (Entwicklung).
- `granite`: [IBM Granite Vision 3.3 2B](https://huggingface.co/ibm-granite/granite-vision-3.3-2b)
  lokal auf dem Mac (Apache 2.0, `uv sync --extra granite`, rund 6 GB Download beim ersten Lauf).
- `openai_compatible`: weiteres lokales Modell, z. B. oMLX, LM Studio oder Ollama, über
  `VISION_BASE_URL`.

Alle liefern dasselbe Pydantic-Schema (`cs2_money_round/vision/schema.py`).

**Feste Bereiche statt Raten.** Das Scoreboard ist immer gleich aufgebaut. Ein Zuschnitt
verschiebt und skaliert es nur. Darum sucht `vision/regions.py` zuerst das Scoreboard-Panel
(dunkle, halbtransparente Fläche mit Seitenverhältnis 1,78) und schneidet alle Bereiche
relativ dazu aus. Zwei Werte liest danach eine einfache Bildanalyse ohne Sprachmodell:

- **Rundenhistorie** (`vision/history_bar.py`): Je Runde ein Strich an fester Stelle. Gold
  heisst T hat gewonnen, Blau CT, Weiss ist die laufende Runde. Das Symbol über oder unter dem
  Strich zeigt den Grund. Es wird mit Vorlagen verglichen, die `tools/learn_history_icons.py`
  aus Screenshots mit bekannter Wahrheit aus den Demos lernt (`vision/history_icons.json`).
- **Eigene Zeile**: hervorgehoben in Teamfarbe (blau auf CT, gelb auf T). Das Gegnerteam ist
  das andere.

Die Bildanalyse ergänzt jedes Vision-Modell (`readers.enhance`). Passt die Zahl der Runden in
der Leiste zum Score, ersetzt sie die Historie des Modells; widerspricht der gelesene Score der
Leiste, gilt die Leiste. Granite bekommt nur vergrösserte Ausschnitte (Zahlenspalten, Namen,
Rundenzahlen), weil es auf dem ganzen Bild Geldbeträge erfindet.

**Granite in Schritten** (`vision/granite.py`):

1. Scoreboard-Panel finden; daraus alle Bereiche.
2. Eigene Zeile über die Zeilenfarbe → eigene Seite → Gegnerseite (ohne Modell).
3. Rundenverlauf aus der Symbolleiste (ohne Modell); daraus auch die Zahl gespielter Runden.
4. Score: Ausschnitt der zwei grossen Rundenzahlen, Granite liest; die Leiste prüft gegen.
   Pistolenrunde = Runde 1 oder 13, also Score-Summe 0 oder 12.
5. Tabelle: Ausschnitt nur der Zahlenspalten, Granite liest alle zehn Zeilen.
6. Plausibilitätsprüfung (`check_rows`): Geld ein Vielfaches von $50 und höchstens $16'000,
   keine Lücke, Deaths je Spieler ≤ gespielte Runden, Deaths eines Teams zwischen den Kills des
   Gegners − 1 und den Kills des Gegners + gespielte Runden (Tode ohne Kill: Bombe, Sturz,
   Teamkill). Schlägt eine Regel an, liest Granite genau diese Spalte für dieses Team einzeln
   nach, bei falscher Anzahl ein zweites Mal stärker vergrössert. Bleibt sie unplausibel, wird
   das Feld als unsicher gemeldet und in der Web-App gelb markiert.

Mit absichtlich verfälschten Werten getestet (falsches Geld, zu viele Deaths, fehlender Wert,
fehlende Zeilen): Granite korrigiert alle vier Fälle durch das Nachlesen, in 2–16 s.

**Screenshots** (Demo-Ansicht, 5120 × 1440, zwei eigene Premier-Matches, Wahrheit je Bild aus der Demo):

- 8 Bilder zum Lernen und Kalibrieren (`data/screenshots/kalibrierung`, nicht im Repo, Spielernamen).
- 4 Bilder zum Prüfen, im Repo unter `examples/screenshots/`: auf das Scoreboard zugeschnitten,
  Namen verpixelt (`tools/make_example_screenshots.py`), Wahrheit in `truth.json`. Sie waren beim
  Lernen nicht dabei.

Messung mit `tools/eval_screenshots.py` (Gegnerseite, Score, Historie, Kills, Deaths, Prognose):

| Vision | Bilder | Seite | Score | Historie | Kills | Deaths | Prognose = Label |
|---|---|---|---|---|---|---|---|
| Granite + Bildanalyse (Modell v5) | 4 Beispiele | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 | 3/4 |
| Granite + Bildanalyse (Modell v5) | 8 Lernbilder | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 5/8 |
| Claude (CLI) + Bildanalyse (Modell v4) | 4 Beispiele | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 |
| Claude (CLI) allein (Modell v4) | 4 Beispiele | 4/4 | 4/4 | 2/4 | 2/4 | 2/4 | 2/4 |

Claude allein liest die lange Historie (14 und 20 Runden) unvollständig; dann bricht die
Inferenz ab, ohne zu raten (Kills, Deaths und Prognose zählen dort als nicht erreicht). Die
falschen Prognosen bei richtig gelesenen Werten sind Fehler des Modells, nicht des Lesens
(Test-Accuracy 0,75); Anubis Runde 21 liegt mit 50 % genau auf der Schwelle. Granite braucht pro Bild rund 20–35 s, Claude
über die CLI rund 12 s. Die Rundenhistorie allein stimmt in 47 von 48 Fällen (12 Bilder in
voller, ¾ und ½ Auflösung).

## Web-App

Für die Prüfung ohne Kommandozeile: Screenshot hochladen oder ein Beispiel wählen, gelesene
Werte prüfen und korrigieren, Prognose berechnen. Ohne Screenshot lassen sich alle Werte von
Hand eingeben. Gestaltet nach dem öffentlichen FHNW-Styleguide (Schwarz, FHNW-Gelb, Inter), auch
fürs Handy.

```bash
uv run uvicorn app.server:app --port 8765                  # Claude (API oder CLI)
uv run --extra granite uvicorn app.server:app --port 8765  # zusätzlich Granite lokal
# → http://127.0.0.1:8765
```

**Schalter «Featurestore und Modell für die Inferenz»** oben auf der Seite:

| Quelle | Was | Verfügbar |
|---|---|---|
| Beispiel im Repo (Standard) | `examples/store`: Modell und `buy_rate_history` als Schnappschuss aus Hopsworks | immer, ohne Zugang |
| Hopsworks | Feature View `buy_rate_fv` (online) und Model Registry live | mit `HOPSWORKS_API_KEY` |
| Lokal | eigener Store aus `FEATURE_STORE=local` | wenn `data/featurestore` und `models/registry` existieren |

Die Wahl gilt pro Vorhersage und bleibt im Browser gespeichert. `APP_FEATURE_STORE=hopsworks`
macht Hopsworks zum Standard. Beispiel-Store und Hopsworks liefern mit Modell v5 dieselbe
Wahrscheinlichkeit. Zur Auswahl stehen ausserdem nur Vision-Modelle, die in der Umgebung laufen
(Claude-CLI installiert, API-Key gesetzt, Granite-Extra installiert). Jede Lesung wird unter
`data/inference/` gespeichert.

## Setup und Ausführen

Voraussetzungen: Python 3.12 (Hopsworks verlangt < 3.14) und
[uv](https://docs.astral.sh/uv/). Alle Abhängigkeiten sind exakt in `uv.lock` festgehalten.

**Schnellstart ohne Hopsworks und ohne Demos** (Inferenz mit dem mitgelieferten Schnappschuss):

```bash
git clone git@github.com:BabaStardust/CS2_Money_Round.git && cd CS2_Money_Round
uv sync                                                   # legt .venv mit Python 3.12 an
FEATURE_STORE=example uv run python pipelines/inference_pipeline.py --team T --history WeWeLbLe --kills 21 --deaths 23
uv run uvicorn app.server:app --port 8765                 # Web-App, Standard: Beispiel im Repo
```

Für Screenshots braucht es ein Vision-Modell: die Claude-CLI, einen `ANTHROPIC_API_KEY` oder
lokal Granite (`uv sync --extra granite`, dann `uv run --extra granite uvicorn …`).

**Vollständig mit Hopsworks** (alle drei Pipelines):

```bash
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

`FEATURE_STORE=example` nutzt den schreibgeschützten Schnappschuss `examples/store` (nur Inferenz;
neu erzeugen mit `uv run python tools/export_example_store.py`). `FEATURE_STORE=local` schaltet
ohne Hopsworks auf einen lokalen Ersatz um, und zwar für alle drei Pipelines gleich:

- Feature Groups werden als Parquet unter `data/featurestore/` gespeichert.
- Modelle landen versioniert unter `models/registry/`.

Tests: `uv run python -m unittest discover -s tests -v`

Screenshots prüfen: `uv run --extra granite python tools/eval_screenshots.py` (bzw. `--provider claude_cli`).

## Reflexion und Limitationen

- **Hopsworks.**
  - Die Pipelines laufen seit 10.11.2026 Ende-zu-Ende gegen das Hopsworks-Projekt `cs2_ml`:
    Feature Groups `round_buy_state` v3 und `buy_rate_history` v1, Feature Views
    `full_buy_fv` v3 und `buy_rate_fv` v1, Modell `full_buy_predictor` v5 in der Registry.
  - `round_buy_state` v1 (nur eigene Matches) hatte für die `gt_*`-Spalten den Typ double
    gespeichert; mit den Pro-Demos wurden sie Ganzzahlen und passten nicht mehr. Darum gibt es
    v2 mit festen Typen. v3 zählt Kills und Deaths nach Zeitpunkt (siehe Demo-Import-Fehler).
    Ältere Feature Groups und die Modelle v1 bis v4 bleiben als frühere Versionen stehen.
  - Der Free Tier erlaubt ein Projekt pro Account. Darum läuft die Arbeit in einem eigenen
    Account; dessen Registrierung hing am 09.11.2026 zuerst in einer Endlosschleife.
  - Beim ersten Insert in die frisch angelegte Online-Feature-Group `buy_rate_history` blieb der
    Online-Store leer, die Inferenz fand keine Kaufquote. Ein zweiter Insert füllte ihn. Die
    Feature Pipeline wartet seither nach dem Insert, bis die Zeilen online sind, und schiebt
    sie bei Bedarf einmal nach.
  - Am 10.11.2026 brach ein Upsert in `buy_rate_history` einmal mit einem HDFS-Lesefehler
    auf Hopsworks-Seite ab (`Generic HdfsObjectStore error`). Ein zweiter Lauf der Feature
    Pipeline ging durch; dank Cache und Upsert ist ein erneuter Lauf gefahrlos.
  - Der Client ist `hopsworks` 5.1.2, das Backend 5.2.2. Hopsworks warnt deshalb bei jedem
    Login; funktional gab es keine Probleme.
- **Wenig Daten.**
  - Es sind 74 Maps: 23 eigene Matches aus einem Spielerkreis und 51 Pro-Maps.
  - Das Testset umfasst 16 Maps, die Metriken schwanken entsprechend.
  - Weil der Split zeitlich ist, besteht das Testset nur aus Pro-Maps. Der Test misst also
    zugleich die Übertragung von Premier- auf Pro-Matches.
  - Maps derselben Best-of-3-Serie haben dieselben Teams und können auf beiden Seiten des
    Splits liegen.
- **HLTV-Demos brauchten eine Korrektur im Import.** Pro-Demos enthalten ein leeres
  Rundenende bei Tick 0 ohne Sieger, dadurch waren alle Runden um eins verschoben. Der Import
  ordnet Rundenenden jetzt über die Rundennummer zu und bricht ab, wenn die Nummern Lücken
  haben. Alle 74 Demos stimmen mit `team_rounds_total` überein.
- **Coaches in Pro-Demos.** In einigen Pro-Demos steht der Coach als sechster Eintrag im Team
  (Name `Coach|…`, $0, tot). Dadurch fehlte zuerst das Label für ganze Maps (nur Fünferteams
  werden bewertet). Der Import filtert Coaches jetzt heraus.
- **«Möglich» ≠ «gekauft».** Für die eigene Taktik zählt, ob der Gegner kaufen *kann*. Ob er
  tatsächlich kauft, sagt das Modell nicht vorher.
- **Screenshots.**
  - Kalibriert und geprüft an 12 Screenshots aus nur zwei Matches (Dust II, Anubis), alle in
    5120 × 1440 und in der Demo-Ansicht. Andere Maps, Auflösungen, HUD-Grössen oder das
    Live-Scoreboard können andere Werte brauchen; die Bereiche sind relativ zum Panel, die
    Farben aber fest gemessen.
  - Das Symbol «Zeit abgelaufen» kam in keinem Bild vor. Solche Runden liest die Leiste als
    Grund «unbekannt»; der Sieger stimmt trotzdem.
  - Bei kleinem Panel (unter rund 1000 px Breite) sinkt die Trefferquote der Symbole; eine
    Runde fehlt in 1 von 48 Fällen. Passt die Rundenzahl nicht zum Score, wird nachgefragt
    statt geraten.
  - Der einzige frühere «Lesefehler» von Granite (Deaths, Dust II Runde 7) war ein Fehler im
    Demo-Import, siehe unten. Die Web-App zeigt trotzdem alle Werte zum Korrigieren an.
  - Ein älterer Screenshot (`dust2_r2_demo.png`) war in der Höhe gestreckt; verzerrte Bilder
    erkennt die Panelsuche nicht und nimmt die Standardlage an.
- **Regelprofil.**
  - Preise und Loss-Bonus stammen aus einem Startprofil (`Rules`, Valve-Stand 2025).
  - Sonderfälle fehlen: Teamkills, Suizide, Disconnects und Kaufzeit-Ende.
  - Overtime wird nicht vorhergesagt.
- **Demo-Import-Fehler gefunden und behoben.**
  - Die erste Version ordnete Score, Kills und Rundenhistorie nach der *Seite* zu. Nach dem
    Halbzeitwechsel war das falsch.
  - Jetzt geschieht die Zuordnung über die Spieler-IDs und wird gegen `team_rounds_total`
    der Demo geprüft (0 Abweichungen).
  - Kills und Deaths wurden nach Rundennummer gezählt. Tode nach dem Rundenende (Bombe,
    Exit-Frags) tragen in der Demo aber schon die nächste Rundennummer, das Scoreboard zählt sie
    sofort. Gefunden über einen Screenshot (15 Deaths auf dem Bild, 14 im Import). Seit v3 wird
    nach Zeitpunkt gezählt, ab Start von Runde 1 (Aufwärm- und Messerrunden zählen nicht).
    Folge: Accuracy 0,732 → 0,749, ROC-AUC 0,827 → 0,828.
  - Regressionstests liegen in `tests/test_pipeline.py`.
- **Event Time** ist die Upload-Zeit der Demo und nicht die exakte Rundenzeit. Für den
  Split nach ganzen Matches reicht das.

## Weiteres

- `docs/economy-prototype.md` beschreibt den ursprünglichen Economy-Prototyp (CLI
  `evaluate` / `simulate` / `estimate`). Das Label baut auf dessen `evaluate` auf.
- `tools/setup_hopsworks.py` legt das Hopsworks-Projekt an.
