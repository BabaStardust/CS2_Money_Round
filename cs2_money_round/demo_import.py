"""CS2-Demo → Rohtabelle mit einer Zeile pro Runde und Team.

Zeitpunkt jeder Zeile ist der Beginn der Kaufphase (Event ``round_start``).
Die Spalten sind getrennt nach Herkunft:

- ``obs_*``: Werte, die auch im Live-Spiel auf dem Scoreboard sichtbar sind
  (Spielstand, Kills/Deaths bisher, Rundenhistorie). Nur daraus dürfen Features
  entstehen.
- ``gt_*``: versteckte Wahrheit aus der Demo (Geld, gespeicherte Ausrüstung).
  Daraus entsteht ausschliesslich das Label.
- ``post_*``: was nach der Kaufphase tatsächlich getragen wurde (nur Analyse).

Das Label ``full_buy_possible`` berechnet das bestehende ``buy.evaluate``.
Aufruf: ``uv run python -m cs2_money_round.demo_import <demo.dem|.dem.bz2> [...] --out data/processed/rounds.parquet``
"""

from __future__ import annotations

import argparse
import bz2
import shutil
import tempfile
from pathlib import Path

import pandas as pd

from .buy import FULL_BUY_TARGET, evaluate
from .model import Player, Rules, Side, Team

TEAM_NUM_TO_SIDE: dict[int, Side] = {2: "T", 3: "CT"}
RIFLE_NAMES = {"AK-47": "ak47", "M4A4": "m4a4", "M4A1-S": "m4a1_s"}
REASON_MAP = {
    "ct_killed": "elimination",
    "t_killed": "elimination",
    "bomb_exploded": "bomb",
    "target_bombed": "bomb",
    "bomb_defused": "defuse",
    "time_ran_out": "time",
    "target_saved": "time",
}
TICK_PROPS = ["balance", "team_num", "armor_value", "has_helmet", "inventory", "team_rounds_total"]
REGULATION_ROUNDS = 24  # Premier/Competitive MR12


def _rifle(inventory) -> str | None:
    for item in inventory if inventory is not None else []:
        if item in RIFLE_NAMES:
            return RIFLE_NAMES[item]
    return None


def _team_won(sides: dict, ids: set[str], winner: str) -> bool | None:
    """Hat das Team mit den Spielern ``ids`` die Runde gewonnen? Seite = Mehrheit der Spieler beim Rundenstart."""
    nums = [sides[sid] for sid in sides if str(sid) in ids]
    if not nums:
        return None
    side = TEAM_NUM_TO_SIDE.get(max(set(nums), key=nums.count))
    return side == winner if side else None


def _real_round_ends(ends: pd.DataFrame) -> pd.DataFrame:
    """Rundenenden in Rundenreihenfolge, Index 0 = Runde 1.

    Pro-Demos (HLTV) enthalten ein leeres ``round_end`` bei Tick 0 ohne Sieger, nach einem
    ``mp_restartgame`` auch Rundenenden, die die Wiederholung ersetzt. Darum zählt nur das letzte
    Ende mit Sieger je Rundennummer (``total_rounds_played`` nach dem Ende).
    """
    ends = ends[ends["winner"].isin(["T", "CT"])]
    ends = ends.drop_duplicates("total_rounds_played", keep="last").sort_values("total_rounds_played")
    numbers = ends["total_rounds_played"].astype(int).tolist()
    if numbers != list(range(1, len(numbers) + 1)):
        raise ValueError(f"Rundenenden nicht lückenlos 1..n: {numbers}")
    return ends.reset_index(drop=True)


def _decompressed(path: Path, workdir: Path) -> Path:
    if path.suffix != ".bz2":
        return path
    target = workdir / path.with_suffix("").name
    with bz2.open(path, "rb") as src, target.open("wb") as dst:
        shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)
    return target


def import_demo(path: Path, rules: Rules = Rules()) -> pd.DataFrame:
    from demoparser2 import DemoParser

    with tempfile.TemporaryDirectory() as tmp:
        demo = _decompressed(path, Path(tmp))
        parser = DemoParser(str(demo))
        header = parser.parse_header()
        starts = parser.parse_event("round_start", other=["total_rounds_played"])
        freeze_end = parser.parse_event("round_freeze_end", other=["total_rounds_played"])
        ends = _real_round_ends(parser.parse_event("round_end", other=["total_rounds_played"]))
        deaths = parser.parse_event("player_death", player=["team_num"], other=["total_rounds_played"])
        plants = parser.parse_event("bomb_planted", other=["total_rounds_played"])

        # Nur echte Runden: total_rounds_played beim Start = Anzahl bereits gespielter Runden.
        starts = starts.drop_duplicates("total_rounds_played", keep="last")
        starts = starts[starts["total_rounds_played"] < len(ends)]
        freeze_end = freeze_end.drop_duplicates("total_rounds_played", keep="last").set_index("total_rounds_played")

        start_ticks = starts["tick"].astype(int).tolist()
        post_ticks = [int(freeze_end.loc[r, "tick"]) for r in starts["total_rounds_played"] if r in freeze_end.index]
        state = parser.parse_ticks(TICK_PROPS, ticks=start_ticks + post_ticks)

    match_id = path.name.split(".")[0]
    ends = ends.reset_index(drop=True)
    # Spieler-Seiten je Rundenstart: nach der Halbzeit (und in der Overtime) wechseln die Teams die
    # Seite, darum werden Historie, Score und Kills über die Spieler-IDs zugeordnet, nicht über die Seite.
    start_by_round = dict(zip(starts["total_rounds_played"].astype(int), starts["tick"].astype(int)))
    sides_at = {
        r: dict(zip(state.loc[state["tick"] == t, "steamid"], state.loc[state["tick"] == t, "team_num"]))
        for r, t in start_by_round.items()
    }
    deaths = deaths.assign(
        victim=deaths["user_steamid"].astype(str), killer=deaths["attacker_steamid"].astype(str)
    )
    rows: list[dict] = []
    for _, start in starts.iterrows():
        played = int(start["total_rounds_played"])  # Runden vor dieser
        round_number = played + 1
        tick = int(start["tick"])
        snap = state[state["tick"] == tick]
        post_tick = int(freeze_end.loc[played, "tick"]) if played in freeze_end.index else None
        post = state[state["tick"] == post_tick] if post_tick is not None else state.iloc[0:0]
        prev = ends.iloc[played - 1] if played > 0 else None

        for team_num, side in TEAM_NUM_TO_SIDE.items():
            team_snap = snap[snap["team_num"] == team_num].sort_values("steamid")
            ids = set(team_snap["steamid"].astype(str))
            won = [_team_won(sides_at.get(i, {}), ids, ends.iloc[i]["winner"]) for i in range(played)]
            reasons = [REASON_MAP.get(r, "?") for r in ends["reason"].iloc[:played]]
            before = deaths[deaths["total_rounds_played"] < played]
            last = deaths[deaths["total_rounds_played"] == played - 1]
            score_team = sum(w is True for w in won)
            rounds_total = team_snap["team_rounds_total"].dropna()
            if len(rounds_total) and int(rounds_total.iloc[0]) != score_team:
                print(f"   ⚠ {match_id} Runde {round_number} {side}: Score {score_team} ≠ team_rounds_total {int(rounds_total.iloc[0])}")
            row = {
                "match_id": match_id,
                "map_name": header.get("map_name"),
                "patch_version": header.get("patch_version"),
                "round_number": round_number,
                "tick": tick,
                "team_side": side,
                "rules_id": rules.id,
                "n_players": len(team_snap),
                # --- live sichtbar (Scoreboard) ---
                "obs_score_team": score_team,
                "obs_score_opp": sum(w is False for w in won),
                "obs_round_history": "".join(
                    ("W" if w else "L") + r[0] for w, r in zip(won, reasons)
                ),
                "obs_team_deaths_total": int(before["victim"].isin(ids).sum()),
                "obs_team_kills_total": int((before["killer"].isin(ids) & ~before["victim"].isin(ids)).sum()),
                "obs_prev_round_team_deaths": int(last["victim"].isin(ids).sum()),
                "obs_prev_round_won": won[-1] if won else None,
                "obs_prev_round_reason": REASON_MAP.get(prev["reason"], prev["reason"]) if prev is not None else None,
                "obs_prev_bomb_planted": bool((plants["total_rounds_played"] == played - 1).any()) if played > 0 else None,
            }
            # --- versteckte Wahrheit → Label ---
            if len(team_snap) == 5:
                players = tuple(
                    Player(
                        id=str(p.steamid),
                        cash=min(int(p.balance), rules.money_cap),
                        rifle=_rifle(p.inventory),
                        armor=int(p.armor_value),
                        helmet=bool(p.has_helmet),
                    )
                    for p in team_snap.itertuples()
                )
                result = evaluate(Team(side=side, players=players), rules, FULL_BUY_TARGET)
                row.update(
                    gt_team_cash=sum(p.cash for p in players),
                    gt_min_cash=min(p.cash for p in players),
                    gt_saved_rifles=sum(p.rifle is not None for p in players),
                    gt_max_equipped_players=result["max_equipped_players"],
                    full_buy_possible=bool(result["full_buy_possible"]),
                )
            else:
                row.update(gt_team_cash=None, gt_min_cash=None, gt_saved_rifles=None,
                           gt_max_equipped_players=None, full_buy_possible=None)
            team_post = post[post["team_num"] == team_num]
            row["post_rifle_armor_players"] = (
                int(sum(_rifle(p.inventory) is not None and p.armor_value > 0 for p in team_post.itertuples()))
                if len(team_post) else None
            )
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("demos", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    frames = []
    for demo in args.demos:
        df = import_demo(demo)
        print(f"{demo.name}: {df['round_number'].max()} Runden, {len(df)} Zeilen, "
              f"Full Buy möglich {df['full_buy_possible'].mean():.0%}")
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"→ {args.out}: {len(out)} Zeilen aus {out['match_id'].nunique()} Demos")


if __name__ == "__main__":
    main()
