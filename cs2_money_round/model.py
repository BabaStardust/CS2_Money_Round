from dataclasses import dataclass
from typing import Literal

Side = Literal["T", "CT"]
RIFLES = frozenset({"ak47", "m4a4", "m4a1_s"})


def integer(value: int, name: str, minimum: int = 0, maximum: int | None = None) -> None:
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")


@dataclass(frozen=True)
class Rules:
    # Starting profile, not a claim of validation against the current game build.
    id: str = "competitive-baseline-v1-unverified"
    ak_price: int = 2700
    m4_price: int = 2900
    kevlar_price: int = 650
    helmet_upgrade_price: int = 350
    money_cap: int = 16000
    start_money: int = 800
    loss_base: int = 1400
    loss_step: int = 500
    max_loss_level: int = 4
    initial_loss_level: int = 1
    win_elimination: int = 3250
    win_objective: int = 3500
    t_plant_loss: int = 600
    plant_reward: int = 300
    defuse_reward: int = 300
    ct_award_per_t_death: int = 50

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if name != "id":
                integer(value, name)
        if min(self.ak_price, self.m4_price, self.kevlar_price) == 0:
            raise ValueError("Equipment prices must be positive")
        if self.initial_loss_level > self.max_loss_level or self.start_money > self.money_cap:
            raise ValueError("Invalid starting state")

    def rifle_price(self, side: Side) -> int:
        if side not in ("T", "CT"):
            raise ValueError("side must be T or CT")
        return self.ak_price if side == "T" else self.m4_price

    def loss_bonus(self, level: int) -> int:
        integer(level, "loss_level", maximum=self.max_loss_level)
        return self.loss_base + level * self.loss_step


@dataclass(frozen=True)
class Player:
    id: str
    cash: int
    rifle: str | None = None
    armor: int = 0
    helmet: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("Player id must be a nonempty string")
        integer(self.cash, "cash")
        integer(self.armor, "armor", maximum=100)
        if self.rifle is not None and self.rifle not in RIFLES:
            raise ValueError("rifle must be ak47, m4a4, m4a1_s or null")
        if type(self.helmet) is not bool:
            raise ValueError("helmet must be boolean")


@dataclass(frozen=True)
class Team:
    side: Side
    players: tuple[Player, ...]
    loss_level: int = 1

    def __post_init__(self) -> None:
        if self.side not in ("T", "CT"):
            raise ValueError("side must be T or CT")
        if len(self.players) != 5 or len({p.id for p in self.players}) != 5:
            raise ValueError("A team needs exactly five players with unique ids")
        integer(self.loss_level, "loss_level")

    def validate(self, rules: Rules) -> None:
        rules.loss_bonus(self.loss_level)
        if any(p.cash > rules.money_cap for p in self.players):
            raise ValueError("Player cash exceeds rules money cap")


def team_from_dict(data: dict) -> Team:
    return Team(side=data["side"], players=tuple(Player(**p) for p in data["players"]),
                loss_level=data.get("loss_level", 1))
