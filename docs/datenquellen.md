# Datenquellen

## Pro-Matches (HLTV, öffentlich)

51 Maps aus 21 Best-of-3-Serien, heruntergeladen am 10.11.2026 als RAR-Archive von den
Match-Seiten auf [hltv.org](https://www.hltv.org) (Abschnitt «Demo»). Die Dateinamen der Demos
(`<team>-vs-<team>-m<nr>-<map>.dem`) stammen aus den Archiven.

| Event | Serie | Maps im Datensatz |
|---|---|---|
| CCT 2026 Challengers North America Series #1 | marsborne vs feistein | m1 Anubis, m2 Ancient |
| CCT 2026 Europe Series #13 | baks vs leo | m1 Nuke, m2 Cache |
| CCT 2026 Europe Series #13 | butterfly vs nexus | m1 Dust II, m2 Ancient, m3 Inferno |
| CCT 2026 Europe Series #14 | baks vs azuolas | m1 Inferno, m2 Ancient |
| CCT 2026 Europe Series #14 | ex-zero-tenacity vs illyrians | m1 Inferno, m2 Mirage |
| CCT 2026 Europe Series #14 Closed Qualifier | honved vs ryvex | m1 Dust II, m2 Ancient |
| CCT 2026 Europe Series #14 Closed Qualifier | lfo-10 vs prestige | m1 Mirage, m2 Dust II |
| CCT 2026 Europe Series #14 Closed Qualifier | maybe vs masonic | m1 Nuke, m2 Ancient, m3 Anubis |
| CCT 2026 Europe Series #14 Closed Qualifier | phantom-academy vs illyrians | m1 Dust II, m2 Cache, m3 Mirage |
| CCT 2026 South America Series #8 | gremio vs zetta | m1 Ancient, m2 Inferno |
| CCT 2026 South America Series #8 | low-cortisol vs rush | m1 Dust II, m2 Nuke |
| CCT 2026 South America Series #8 | your-end vs damajuana | m1 Cache, m2 Ancient, m3 Mirage |
| ESL Challenger League Season 52 Europe Cup 11 | aimclub vs 9ine | m1 Cache, m2 Dust II, m3 Anubis |
| ESL Challenger League Season 52 Europe Cup 11 | aimclub vs alliance | m1 Inferno, m2 Mirage |
| ESL Challenger League Season 52 Europe Cup 11 | alliance vs fokus | m1 Mirage, m2 Anubis, m3 Dust II |
| European Pro League Series 10 | balkan-outlaws vs ex-ence | m1 Anubis, m2 Dust II, m3 Mirage |
| European Pro League Series 10 | forze-reload vs mouz-nxt | m1 Ancient, m2 Dust II |
| European Pro League Series 10 | lfo-10 vs mouz-nxt | m1 Mirage, m2 Dust II, m3 Ancient |
| European Pro League Series 10 | walczaki vs noir-verse | m1 Nuke, m2 Mirage, m3 Anubis |
| NODWIN Clutch Series 14 | black-phoenix vs baks | m1 Cache, m2 Dust II |
| NODWIN Clutch Series 14 | brute vs illyrians | m1 Dust II, m2 Inferno |

## Eigene Matches (Leetify, privat)

23 Premier-Matches (MR12) vom 16.10. bis 06.11.2026 aus einem Spielerkreis, über Leetify als
`.dem.bz2` von Valves Replay-Servern geladen. Sie enthalten Namen und Steam-IDs der Mitspieler
und werden deshalb nicht weitergegeben. Für den Nachbau genügen die Pro-Maps allein für alle
drei Pipelines; ein Modell nur auf den eigenen Matches erreichte auf den Pro-Maps ROC-AUC 0,823,
das Muster ist also auf beiden Datenquellen dasselbe.

## Nachbauen

1. Demos der Serien oben von HLTV laden, entpacken und nach `data/raw/demos/` legen.
2. Optional `data/raw/demos/upload_times.csv` (`datei,last_modified`) anlegen; sonst gilt die Dateizeit als Event Time.
3. `uv run python pipelines/feature_pipeline.py`, danach `uv run python pipelines/training_pipeline.py`.
