import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

from .buy import BuyTarget, estimate, evaluate
from .model import Rules, team_from_dict
from .simulation import Action, Outcome, reset_half, settle


def reject_constant(value: str) -> None:
    raise ValueError(f"Invalid JSON constant {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="CS2 full-buy feasibility and round accounting")
    parser.add_argument("command", choices=["evaluate", "estimate", "simulate"])
    parser.add_argument("input", type=Path, help="JSON input file")
    args = parser.parse_args()
    try:
        data = json.loads(args.input.read_text(), parse_constant=reject_constant)
        rules = Rules(**data.get("rules", {}))
        target = BuyTarget(**data.get("target", {}))
        if args.command == "evaluate":
            result = evaluate(team_from_dict(data["team"]), rules, target)
        elif args.command == "estimate":
            result = estimate([(s["weight"], team_from_dict(s["team"])) for s in data["scenarios"]], rules, target)
        else:
            team = team_from_dict(data["team"])
            rounds = []
            for index, event in enumerate(data["rounds"], 1):
                if event.get("reset_half", False):
                    team = reset_half(team, rules)
                before = evaluate(team, rules, target)
                team, ledger = settle(team, tuple(Action(**a) for a in event["actions"]), Outcome(**event["outcome"]), rules)
                rounds.append({"sequence": index, "buy_before_round": before, "ledger": ledger,
                               "next_round_buy": evaluate(team, rules, target)})
            result = {"rounds": rounds, "final_team": asdict(team)}
        print(json.dumps(result, indent=2, allow_nan=False))
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as error:
        print(f"Input error: {error}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
