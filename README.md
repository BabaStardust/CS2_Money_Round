# CS2_Money_Round

Wie wahrscheinlich können alle fünf Gegner diese Runde mit AK/M4 und Rüstung antreten?

Der erste Prototyp enthält eine Full-Buy-Prüfung, einen Economy-Rundensimulator
und eine Auswertung gewichteter Szenarien. Er läuft lokal als Python-CLI ohne
Laufzeitabhängigkeiten. Screenshot-Erkennung, Demo-Import und ein trainiertes
ML-Modell sind noch nicht implementiert.

## Start

Python 3.11 oder neuer, im Projektordner:

```bash
python -m cs2_money_round evaluate examples/full_buy.json
python -m cs2_money_round simulate examples/pistol_loss.json
python -m cs2_money_round estimate examples/scenarios.json
python -m unittest discover -s tests -v
```

Optional: `uv sync`, danach steht auch `uv run cs2-money-round` zur Verfügung.

## Full-Buy-Definition

- Alle fünf Spieler haben eine AK-47, M4A4 oder M4A1-S und mindestens einen
  Rüstungspunkt. Auch eine auf der Gegenseite aufgehobene Rifle zählt.
- Standardmässig genügt Kevlar. Helm und volle Rüstung sind konfigurierbar.
- Vorhandene Rifles können im Team verteilt werden. Neue Waffen können von
  Mitspielern bezahlt und gedroppt werden; Geld und Rüstung sind nicht übertragbar.
- Das Modell prüft finanzielle Machbarkeit während einer verfügbaren Kaufphase,
  ohne Einschränkungen durch Position, abgelaufene Kaufzeit oder individuelle
  Loadouts. Es prognostiziert keine tatsächliche Kaufentscheidung.
- Jede Spieler-Geldsumme wird einzeln geprüft. Das Gesamtgeld allein genügt nicht.
- AWP, FAMAS, Galil und andere Waffen erfüllen die bewusst enge Definition nicht.
  Im Eingabeformat bedeutet `rifle: null` deshalb **keine passende Rifle**.

Im JSON kann `target` ergänzt werden:

```json
{"min_armor": 100, "helmet": true, "reserve": 1000}
```

`reserve` ist Geld, das jeder ausgerüstete Spieler nach dem Kauf mindestens
übrig haben muss. Es ist eine Budgetreserve, keine simulierte Utility-Ausrüstung.
Beim Standardziel ohne Helm kostet der Neukauf im Startprofil $3.350 als T und
$3.550 als CT. Für eine Komfortschwelle von $4.500 bzw. $4.900 setzt man `reserve`
auf die Differenz zum jeweiligen Neukaufpreis. Bei gespeichertem Equipment muss
nur die fehlende Ausrüstung plus Reserve finanziert werden.

`evaluate` liefert `full_buy_possible` und `max_equipped_players`. Bei einem
unvollständigen Buy beschreiben die Kosten/Kapazitäten die ausgewählte grösste
machbare Spielergruppe. `equipped_player_ids` ist eine machbare Gruppe, keine
Vorhersage darüber, wer tatsächlich kauft. Es gibt noch keinen Drop-Aktionsplan.

## Runden simulieren

Ein Simulationsdokument enthält `team` vor der Kaufphase und eine Liste `rounds`.
Jede Runde benötigt ein `outcome` und genau eine `Action` pro Spieler:

- `spent`: tatsächliche Nettoausgaben nach Rückerstattungen, inklusive Drops an
  andere Spieler. Kaufende und tragende Spieler sind nicht zwingend dieselben.
- `kill_income`: Summe der waffenabhängigen Kill-Prämien. Die Zuordnung von
  Kill-Waffen zu Prämien übernimmt später der Demo-Adapter.
- `died`, `end_rifle`, `end_armor`, `end_helmet`: Zustand am Rundenende. So werden
  gerettete und aufgehobene Waffen ohne erfundene Käufe übernommen.
- `outcome`: Sieger, Grund (`elimination`, `timeout`, `explosion`, `defuse`),
  Plant-Status, gegebenenfalls Akteur-IDs und Anzahl gestorbener Terroristen.

Der Simulator berechnet Siegprämie, Loss-Bonus, Bombenprämien, CT-Teamprämie,
Geldlimit und den Zustand für die nächste Runde. Jeder Betrag steht im Ledger.
T-Überlebende bei einer Niederlage durch Zeitablauf erhalten keinen Loss-Bonus.

`loss_bonus_before_round` ist die mögliche Auszahlung bei Verlust der gerade
simulierten Runde. `loss_bonus_received` ist die tatsächliche persönliche
Auszahlung. `loss_bonus_if_next_round_lost` beschreibt die darauffolgende Runde.

Für den regulären Seitenwechsel wird in der ersten Runde der neuen Hälfte
`"reset_half": true` gesetzt. Der Wechsel erfolgt ausdrücklich, nicht automatisch
nach zwölf Einträgen: eine Datei darf auch mitten im Match beginnen. Overtime,
Disconnect-Kompensation, Teamkills, Suizidstrafen und Käufe während der laufenden
Runde sind noch nicht modelliert. Der Simulator validiert grundlegende
Konsistenz, rekonstruiert aber keinen vollständigen Kampf- oder Kaufablauf.

## Wahrscheinlichkeiten

`estimate` prüft vollständige alternative Teamzustände mit positiven Gewichten.
Die Full-Buy-Wahrscheinlichkeit ist die normalisierte Gewichtsmasse der Zustände,
in denen fünf Rifle-plus-Rüstung-Ausrüstungen finanzierbar sind.

Die Gewichte im Beispiel sind **synthetisch und von Hand gesetzt**. Die Ausgabe
ist keine trainierte, kalibrierte Gegnerprognose. Der Simulator generiert auch
noch keine möglichen Zustände aus Scoreboard-Beobachtungen. Später muss ein
Inferenzmodell diese Zustände und Gewichte anhand der sichtbaren Historie liefern.

## Dataset-Vertrag

`schemas/round_record.schema.json` beschreibt ein Trainingsbeispiel pro Gegnerteam
und Runde; `examples/round_record.json` zeigt ein synthetisches Beispiel.
JSONL eignet sich für viele solcher Datensätze. Das Schema ist ein Vertrag für
den zukünftigen Import, noch kein fertiger Dataset-Importer.

- `match_id`, `game_build`, `rules_id`, Runde und stabile Team-/Spieler-IDs erhalten.
- Zielzeitpunkt: **vor der Kaufphase**. Geld und gespeicherte Ausrüstung müssen
  zu diesem Zeitpunkt gehören; Daten nach dem Kauf nicht damit vermischen.
- `observation`: nur zu diesem Zeitpunkt öffentlich verfügbare Informationen.
  `null` bedeutet unbekannt; eine leere Todesliste bedeutet bekannte null Tode.
- `ground_truth`: tatsächliche Gegner-Geldstände und gespeicherte Ausrüstung aus
  abgeschlossenen Demos. Optional tatsächliche Nettoausgaben der anschliessenden
  Kaufphase, ausschliesslich als Zielwerte/Analysewerte.
- Versteckte Gegnerwerte und spätere Ereignisse dürfen keine Modellmerkmale werden.
- Training/Test nach **ganzen Matches**, zusätzlich zeitlich und nach Spielversion
  trennen. Der spätere Full-Buy-Zielwert wird durch `evaluate` berechnet.
- Das Schema prüft nicht alle fachlichen Beziehungen: Import muss unter anderem
  eindeutige Spieler-IDs, Teamzuordnung, Regeln und Zeitpunkte zusätzlich prüfen.

## Regeln und Validierungsstand

`Rules` ist ein überschreibbares Startprofil
`competitive-baseline-v1-unverified`. Im CLI-Dokument kann ein `rules`-Objekt
beliebige Felder überschreiben; bei Regeländerungen auch eine eigene `id` setzen.
Das Profil ist **noch nicht gegen Demos der aktuellen Spielversion validiert**.
Unbekannte Spezialfälle sollen vor einem Einsatz für echte Prognosen abgeglichen
werden. Die Tests belegen das implementierte Profil, nicht dessen Aktualität.

Quellen zur Nachprüfung:

- [Valve: Loss-Bonus, 13.03.2019](https://blog.counter-strike.net/it/2019/03/23521/)
- [Valve: Begrenzung und Absenkung vom Maximum, 14.05.2019](https://blog.counter-strike.net/ro/2019/05/24208/)
- [Valve: Economy-Änderungen Mai 2024](https://store.steampowered.com/news/posts/?appids=730&enddate=1716499920&feed=steam_community_announcements)
- [Valve: CT-Prämie Juli 2025](https://store.steampowered.com/news/posts/?appids=730&enddate=1753835380&feed=steam_community_announcements)
- [demoparser: mögliche Demo-Felder](https://github.com/LaihoE/demoparser)

## Nächster Arbeitsschritt

Ein reales Dataset oder wenige abgeschlossene CS2-Demos importieren, Regelprofil
und tatsächliche Kaufphasen-Geldstände prüfen und eine erste nachvollziehbare
Verteilung möglicher Gegnerzustände aus beobachtbaren Rundenereignissen erstellen.
Danach Screenshot-Erkennung und ML gegen diese Basis evaluieren.
