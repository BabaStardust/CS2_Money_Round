"""Feature Pipeline: CS2-Demos → Features + Label → Feature Groups.

Schritte (gemäss Aufgabenstellung):
  1. Rohdaten holen: Demos aus data/raw/demos/ parsen (inkrementell, Cache je Demo in data/interim/rounds/)
  2. Features engineeren (Rundenhistorie, kumulierte Stats, Kaufquote aus früheren Matches)
  3. DataFrame mit Features und Label full_buy_possible erstellen
  4. Feature Groups mit Primary Key + Event Time erstellen
  5. DataFrames in die Feature Groups einfügen

Aufruf:
    uv run python pipelines/feature_pipeline.py             # inkl. Upload (FEATURE_STORE aus .env)
    uv run python pipelines/feature_pipeline.py --dry-run   # nur Schritte 1–3, kein Upload
    FEATURE_STORE=local uv run python pipelines/feature_pipeline.py   # lokaler Featurestore
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cs2_money_round.demo_import import import_demo  # noqa: E402
from cs2_money_round.features import LABEL, build_features, current_rates  # noqa: E402
from cs2_money_round.specs import RATES_FG, ROUNDS_FG  # noqa: E402
from cs2_money_round.store import store_from_env  # noqa: E402

DEMO_DIR = ROOT / "data" / "raw" / "demos"
CACHE_DIR = ROOT / "data" / "interim" / "rounds"
UPLOAD_TIMES = DEMO_DIR / "upload_times.csv"


def match_id_of(path: Path) -> str:
    return path.name.split(".")[0]


# ---------------------------------------------------------------- 1. Rohdaten
def load_rounds(demo_dir: Path) -> pd.DataFrame:
    """Parst nur Demos, die noch nicht im Cache sind; liefert alle Runden aller Demos."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    demos = sorted([*demo_dir.glob("*.dem"), *demo_dir.glob("*.dem.bz2")])
    if not demos:
        sys.exit(f"Keine Demos in {demo_dir} (erwartet *.dem oder *.dem.bz2).")
    frames, new = [], 0
    for demo in demos:
        cache = CACHE_DIR / f"{match_id_of(demo)}.parquet"
        if not cache.exists():
            df = import_demo(demo)
            df.to_parquet(cache, index=False)
            new += 1
            print(f"   neu geparst: {demo.name} ({df['round_number'].max()} Runden)")
        frames.append(pd.read_parquet(cache))
    print(f"   {len(demos)} Demos, davon {new} neu geparst, {len(demos) - new} aus dem Cache")
    return pd.concat(frames, ignore_index=True)


def attach_match_time(rounds: pd.DataFrame, demo_dir: Path) -> pd.DataFrame:
    """Event Time = Upload-Zeit der Demo bei Valve (Last-Modified ≈ Matchende, UTC).

    Fehlt sie in upload_times.csv, dient die Änderungszeit der Datei als Ersatz (mit Warnung).
    """
    times: dict[str, pd.Timestamp] = {}
    if UPLOAD_TIMES.exists():
        ut = pd.read_csv(UPLOAD_TIMES)
        times = {match_id_of(Path(f)): pd.to_datetime(t, utc=True) for f, t in zip(ut["datei"], ut["last_modified"])}
    missing = sorted(set(rounds["match_id"]) - set(times))
    for match_id in missing:
        files = [*demo_dir.glob(f"{match_id}.dem*")]
        times[match_id] = pd.Timestamp(files[0].stat().st_mtime, unit="s", tz="UTC")
        print(f"   ⚠ keine Upload-Zeit für {match_id}, nutze Dateizeit {times[match_id]:%d.%m.%Y %H:%M}")
    rounds = rounds.copy()
    rounds["match_time"] = rounds["match_id"].map(times)
    return rounds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="kein Upload, nur Vorschau")
    ap.add_argument("--demo-dir", type=Path, default=DEMO_DIR)
    args = ap.parse_args()
    load_dotenv(ROOT / ".env")

    print("1. Rohdaten: Demos parsen")
    rounds = attach_match_time(load_rounds(args.demo_dir), args.demo_dir)

    print("2./3. Features engineeren, DataFrame mit Label")
    feats = build_features(rounds)
    rates = current_rates(feats)
    print(f"   {len(rounds)} Team-Runden → {len(feats)} im Modellumfang "
          f"(ohne Pistolenrunden, Overtime, Teams ≠ 5 Spieler)")
    print(f"   {feats['match_id'].nunique()} Matches, {feats['match_time'].min():%d.%m.%Y} – {feats['match_time'].max():%d.%m.%Y}")
    print(f"   Full Buy möglich: {feats[LABEL].mean():.1%}")
    print(rates.to_string(index=False))

    if args.dry_run:
        out = ROOT / "data" / "processed" / "round_buy_state_preview.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        feats.to_csv(out, index=False)
        print(f"Dry-Run: kein Upload. Vorschau: {out.relative_to(ROOT)}")
        return

    print("4./5. Feature Groups erstellen + einfügen")
    store = store_from_env()
    store.insert(ROUNDS_FG, feats)
    store.insert(RATES_FG, rates)
    print(f"   [{store.kind}] {ROUNDS_FG.name} v{ROUNDS_FG.version}: {len(feats)} Zeilen, "
          f"{RATES_FG.name} v{RATES_FG.version}: {len(rates)} Zeilen")


if __name__ == "__main__":
    main()
