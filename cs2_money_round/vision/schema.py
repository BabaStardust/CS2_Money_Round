"""Was das Vision-Modell aus einem CS2-Scoreboard-Screenshot liest.

Das Schema ist anbieterneutral: Jeder Reader (Anthropic, lokales Modell) muss
genau dieses Objekt liefern. ``None`` heisst immer «auf dem Bild nicht lesbar
oder nicht sichtbar», nie «null». So bleibt erkennbar, ob ein Wert fehlt.

Wichtig für das Modell: Im Live-Spiel ist das Geld der Gegner NICHT sichtbar,
in einer Demo (Zuschauersicht) schon. Der Reader liest, was zu sehen ist; welche
Felder ein Modellmerkmal werden dürfen, entscheidet die Inferenz-Pipeline.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Side = Literal["CT", "T"]
WinReason = Literal["elimination", "bomb", "defuse", "time"]


class PlayerReading(BaseModel):
    name: str = Field(description="Spielername wie angezeigt, ohne Clan-Tag")
    money: int | None = Field(description="Geld in Dollar; None, wenn die Spalte für diesen Spieler leer oder verdeckt ist")
    kills: int | None
    deaths: int | None
    assists: int | None
    alive: bool | None = Field(description="False, wenn die Zeile als tot markiert ist; None, wenn nicht erkennbar")
    highlighted: bool = Field(description="True nur für die hervorgehobene Zeile (eigener Spieler / Zuschauer-Perspektive)")


class TeamReading(BaseModel):
    score: int | None = Field(description="Gewonnene Runden dieses Teams")
    alive_count: int | None = Field(description="Zahl aus 'Alive: x/5'")
    loss_bonus_level: int | None = Field(
        description="Loss-Bonus-Stufe dieses Teams als Anzahl gefüllter Balken/Striche; None, wenn für dieses Team nicht angezeigt"
    )
    players: list[PlayerReading]


class RoundResult(BaseModel):
    round_number: int
    winner: Side
    reason: WinReason | None = Field(description="Aus dem Symbol in der Rundenhistorie; None, wenn nicht eindeutig")


class ScoreboardReading(BaseModel):
    map_name: str | None = Field(description="z. B. 'Dust II'")
    game_mode: str | None = Field(description="z. B. 'Premier'")
    ct: TeamReading
    t: TeamReading
    round_history: list[RoundResult] = Field(description="Bereits gespielte Runden, älteste zuerst; leer, wenn keine gespielt oder nicht lesbar")
    is_demo_view: bool = Field(description="True, wenn Geld beider Teams sichtbar ist oder Demo-Steuerelemente zu sehen sind")
    uncertain_fields: list[str] = Field(description="Felder, bei denen das Ablesen unsicher war, z. B. 't.players[2].money'")
