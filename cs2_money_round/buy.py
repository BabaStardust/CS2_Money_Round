"""Exact feasibility under the declared cash/equipment/drop assumptions."""
from dataclasses import asdict, dataclass
from itertools import combinations
from math import fsum, isfinite

from .model import Player, Rules, Team, integer


@dataclass(frozen=True)
class BuyTarget:
    min_armor: int = 1
    helmet: bool = False
    # Extra cash retained by each equipped player, e.g. for utility purchases.
    reserve: int = 0

    def __post_init__(self) -> None:
        integer(self.min_armor, "min_armor", minimum=1, maximum=100)
        integer(self.reserve, "reserve")
        if type(self.helmet) is not bool:
            raise ValueError("helmet must be boolean")


# Projektdefinition Full Buy (Owner, 09.11.2026): Rifle plus Kevlar mit Helm.
# Neukauf: T $3.700 (AK + $1.000), CT $3.900 (M4 + $1.000). Gespeicherte Rifle
# und vorhandene Rüstung mit Helm zählen, sie müssen nicht neu gekauft werden.
FULL_BUY_TARGET = BuyTarget(helmet=True)


def armor_cost(player: Player, target: BuyTarget, rules: Rules) -> int:
    cost = rules.kevlar_price if player.armor < target.min_armor else 0
    if target.helmet and not player.helmet:
        # A helmet-only upgrade is available at full armor. Otherwise the
        # complete armor/helmet package is needed and also restores armor.
        if player.armor == 100:
            cost += rules.helmet_upgrade_price
        else:
            cost = rules.kevlar_price + rules.helmet_upgrade_price
    return cost


def evaluate(team: Team, rules: Rules = Rules(), target: BuyTarget = BuyTarget()) -> dict:
    team.validate(rules)
    price = rules.rifle_price(team.side)
    saved = sum(p.rifle is not None for p in team.players)
    costs = [armor_cost(p, target, rules) for p in team.players]
    # Enumerate just 32 subsets. Cash cannot move; rifles can. Each buyer's
    # individual floor matters: five spare $600 balances cannot buy one AK.
    for size in range(5, -1, -1):
        for selected in combinations(range(5), size):
            remaining = [p.cash - (costs[i] + target.reserve if i in selected else 0)
                         for i, p in enumerate(team.players)]
            if min(remaining) < 0:
                continue
            capacity = sum(cash // price for cash in remaining)
            if saved + capacity >= size:
                return {"full_buy_possible": size == 5, "max_equipped_players": size,
                        "equipped_player_ids": [team.players[i].id for i in selected],
                        "team_cash": sum(p.cash for p in team.players),
                        "saved_rifles": saved, "new_rifles_needed": max(0, size - saved),
                        "rifle_purchase_capacity": capacity,
                        "personal_armor_costs": dict(zip((p.id for p in team.players), costs)),
                        "reserve_per_equipped_player": target.reserve,
                        "target": asdict(target), "rules_id": rules.id}
    raise AssertionError("Empty subset is always feasible")


def estimate(scenarios: list[tuple[float, Team]], rules: Rules = Rules(),
             target: BuyTarget = BuyTarget()) -> dict:
    if not scenarios or any(isinstance(w, bool) or not isinstance(w, (int, float))
                            or not isfinite(w) or w <= 0 for w, _ in scenarios):
        raise ValueError("Scenarios require finite, positive weights")
    if len({team.side for _, team in scenarios}) != 1:
        raise ValueError("All scenarios must describe the same side")
    total = fsum(w for w, _ in scenarios)
    if not isfinite(total):
        raise ValueError("Scenario weight total must be finite")
    results = [(w / total, evaluate(team, rules, target)) for w, team in scenarios]
    return {"full_buy_probability": fsum(w for w, r in results if r["full_buy_possible"]),
            "at_least_four_probability": fsum(w for w, r in results if r["max_equipped_players"] >= 4),
            "scenario_count": len(results), "rules_id": rules.id,
            "probability_basis": "User-supplied scenario weights; not calibrated or learned",
            "scenarios": [{"normalized_weight": w, **r} for w, r in results]}
