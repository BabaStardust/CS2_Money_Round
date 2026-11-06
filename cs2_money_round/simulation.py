"""Round accounting. Inputs describe actual actions, not inferred purchases."""
from dataclasses import dataclass
from typing import Literal

from .model import Player, Rules, Side, Team, integer


@dataclass(frozen=True)
class Action:
    player_id: str
    spent: int
    kill_income: int
    died: bool
    end_rifle: str | None
    end_armor: int
    end_helmet: bool = False

    def __post_init__(self) -> None:
        integer(self.spent, "spent")
        integer(self.kill_income, "kill_income")
        if type(self.died) is not bool:
            raise ValueError("died must be boolean")
        Player(self.player_id, 0, self.end_rifle, self.end_armor, self.end_helmet)
        if self.died and (self.end_rifle is not None or self.end_armor or self.end_helmet):
            raise ValueError("Dead players cannot retain equipment")


@dataclass(frozen=True)
class Outcome:
    winner: Side
    reason: Literal["elimination", "timeout", "explosion", "defuse"]
    bomb_planted: bool = False
    planter_id: str | None = None
    defuser_id: str | None = None
    t_deaths: int = 0

    def __post_init__(self) -> None:
        if self.winner not in ("T", "CT"):
            raise ValueError("winner must be T or CT")
        if self.reason not in ("elimination", "timeout", "explosion", "defuse"):
            raise ValueError("Invalid round end reason")
        integer(self.t_deaths, "t_deaths", maximum=5)
        if type(self.bomb_planted) is not bool:
            raise ValueError("bomb_planted must be boolean")
        if self.reason == "timeout" and (self.winner != "CT" or self.bomb_planted):
            raise ValueError("Timeout requires CT win without a plant")
        if self.reason in ("explosion", "defuse"):
            expected = "T" if self.reason == "explosion" else "CT"
            if self.winner != expected or not self.bomb_planted:
                raise ValueError("Bomb result requires plant and correct winner")
        if self.winner == "CT" and self.bomb_planted and self.reason != "defuse":
            raise ValueError("CT cannot win a planted-bomb round without defusing")
        if self.planter_id is not None and not self.bomb_planted:
            raise ValueError("Planter requires bomb plant")
        if self.defuser_id is not None and self.reason != "defuse":
            raise ValueError("Defuser requires defuse")


def settle(team: Team, actions: tuple[Action, ...], outcome: Outcome,
           rules: Rules = Rules()) -> tuple[Team, dict]:
    team.validate(rules)
    by_id = {a.player_id: a for a in actions}
    if len(by_id) != 5 or len(actions) != 5 or set(by_id) != {p.id for p in team.players}:
        raise ValueError("Exactly one action per player is required")
    if team.side == "T" and sum(a.died for a in actions) != outcome.t_deaths:
        raise ValueError("t_deaths disagrees with T player actions")
    if team.side == "CT" and outcome.planter_id in by_id:
        raise ValueError("CT player cannot plant")
    if team.side == "T" and outcome.defuser_id in by_id:
        raise ValueError("T player cannot defuse")
    if team.side == "T" and outcome.bomb_planted and outcome.planter_id not in by_id:
        raise ValueError("T round needs its planter id to account for personal income")
    if team.side == "CT" and outcome.reason == "defuse" and outcome.defuser_id not in by_id:
        raise ValueError("CT round needs its defuser id")
    if outcome.defuser_id in by_id and by_id[outcome.defuser_id].died:
        raise ValueError("Defuser must survive the completed round")
    if outcome.reason == "elimination" and team.side != outcome.winner and not all(a.died for a in actions):
        raise ValueError("Eliminated losing team must have five deaths")
    won = team.side == outcome.winner
    loss_before = rules.loss_bonus(team.loss_level)
    if won:
        base = rules.win_objective if outcome.reason in ("explosion", "defuse") else rules.win_elimination
    else:
        base = loss_before
    new_level = max(0, team.loss_level - 1) if won else min(rules.max_loss_level, team.loss_level + 1)
    next_players, ledger = [], []
    for p in team.players:
        a = by_id[p.id]
        if a.spent > p.cash:
            raise ValueError(f"{p.id}: spending exceeds buy-phase cash")
        loss_paid = base if not won else 0
        if not won and team.side == "T" and outcome.reason == "timeout" and not a.died:
            loss_paid = 0
        win_paid = base if won else 0
        plant_loss = rules.t_plant_loss if team.side == "T" and not won and outcome.bomb_planted else 0
        objective = ((rules.plant_reward if p.id == outcome.planter_id else 0)
                     + (rules.defuse_reward if p.id == outcome.defuser_id else 0))
        ct_award = rules.ct_award_per_t_death * outcome.t_deaths if team.side == "CT" else 0
        uncapped = p.cash - a.spent + a.kill_income + loss_paid + win_paid + plant_loss + objective + ct_award
        cash = min(rules.money_cap, uncapped)
        next_players.append(Player(p.id, cash, a.end_rifle, a.end_armor, a.end_helmet))
        ledger.append({"player_id": p.id, "opening_cash": p.cash, "spent": a.spent,
                       "kill_income": a.kill_income, "loss_bonus_received": loss_paid,
                       "win_income": win_paid, "plant_loss_income": plant_loss,
                       "personal_objective_income": objective, "ct_team_income": ct_award,
                       "cap_discarded": uncapped - cash, "closing_cash": cash, "died": a.died})
    next_team = Team(team.side, tuple(next_players), new_level)
    return next_team, {"rules_id": rules.id, "loss_bonus_before_round": loss_before,
                       "loss_bonus_if_next_round_lost": rules.loss_bonus(new_level), "players": ledger}


def reset_half(team: Team, rules: Rules = Rules()) -> Team:
    """Explicit regular-half reset; overtime uses a separate rules profile."""
    return Team("CT" if team.side == "T" else "T",
                tuple(Player(p.id, rules.start_money) for p in team.players), rules.initial_loss_level)
