"""Lokaler Vision-Reader mit IBM Granite Vision 3.3 2B (Apache 2.0, läuft auf dem Mac via MPS).

Granite ist klein und liefert kein JSON nach Schema. Auf dem ganzen Screenshot erfindet es
Geldbeträge; auf vergrösserten Ausschnitten liest es zuverlässig. Darum:
  1. Scoreboard-Panel finden (``regions.locate_panel``)
  2. Je Bereich eine einfache Frage stellen: Zahlenspalten als CSV, Namen, Score
  3. Hervorgehobene Zeile ohne Modell über die Zeilenfarbe bestimmen (``regions.highlighted_rows``)
  4. Antworten mit festen Regeln in ein :class:`ScoreboardReading` übersetzen

Die Rundenhistorie (kleine Symbole) liest Granite nicht verlässlich; sie kommt aus der
Bildanalyse der Symbolleiste (``history_bar``, über ``readers.enhance``).

Installation: ``uv sync --extra granite`` (torch, transformers). Modell: ``GRANITE_MODEL``,
Standard ``ibm-granite/granite-vision-3.3-2b``.
"""

from __future__ import annotations

import io
import re
from functools import lru_cache

from .regions import column_box, crop, highlighted_rows, locate_panel
from .schema import PlayerReading, ScoreboardReading, TeamReading

DEFAULT_MODEL = "ibm-granite/granite-vision-3.3-2b"

NUMBERS_PROMPT = "Convert this table to CSV with the columns Money, Kills, Deaths, Assists. One line per row."
NAMES_PROMPT = "List the player names, one per line, top to bottom."
COLUMN_PROMPT = "This is the {name} column of a scoreboard. List every number from top to bottom, one per line."
SCORE_PROMPT = (
    "There are two big numbers: the top one belongs to COUNTER-TERRORISTS, the bottom one to TERRORISTS. "
    "Answer as: CT=<n> T=<n>"
)

MONEY = re.compile(r"\$\s?(\d{1,2}[,.']?\d{3}|\d{1,5})")


def parse_numbers(text: str) -> list[dict]:
    """CSV-Antwort → Geld und K/D/A je Zeile. Zeilen ohne ``$``-Betrag (Kopfzeile) werden übersprungen.

    Geld = erstes ``$``-Feld, die drei Ganzzahlen danach = Kills, Deaths, Assists. Eine
    vorangestellte Zeilennummer (``0,$2600,…``) stört so nicht.
    """
    rows = []
    for line in text.splitlines():
        m = MONEY.search(line)
        if not m:
            continue
        after = [int(x) for x in re.findall(r"(?<![\d$])\d+", line[m.end():])]
        rows.append({
            "money": int(re.sub(r"\D", "", m.group(1))),
            "kills": after[0] if len(after) > 0 else None,
            "deaths": after[1] if len(after) > 1 else None,
            "assists": after[2] if len(after) > 2 else None,
        })
    return rows


def parse_names(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def parse_score(text: str) -> tuple[int | None, int | None]:
    ct = re.search(r"CT\s*=\s*(\d+)", text)
    t = re.search(r"(?<![C\w])T\s*=\s*(\d+)", text)
    return (int(ct.group(1)) if ct else None, int(t.group(1)) if t else None)


FIELDS = ("money", "kills", "deaths", "assists")
SIDES = ("CT", "T")
MONEY_CAP = 16000


def check_rows(rows: list[dict], played: int | None) -> dict[tuple[str, str], str]:
    """Plausibilität der Tabelle → {(Seite, Spalte): Grund}. Leer = alles plausibel.

    - genau 10 Zeilen, keine Lücken
    - Geld: Vielfaches von $50, höchstens $16'000
    - Deaths je Spieler ≤ gespielte Runden
    - Deaths eines Teams zwischen (Kills des Gegners − 1) und (Kills des Gegners + gespielte Runden):
      jeder Kill ist ein Tod beim Gegner; dazu kommen Tode ohne Kill (Bombe, Sturz, Teamkill)
    """
    if len(rows) != 10:
        return {(side, f): f"{len(rows)} statt 10 Zeilen" for side in SIDES for f in FIELDS}
    problems: dict[tuple[str, str], str] = {}
    for i, r in enumerate(rows):
        side = SIDES[i // 5]
        m = r["money"]
        if m is None or m % 50 or m > MONEY_CAP:
            problems[(side, "money")] = f"Geld {m} unplausibel"
        for f in ("kills", "deaths", "assists"):
            if r[f] is None:
                problems[(side, f)] = "Wert fehlt"
        if played is not None and r["deaths"] is not None and r["deaths"] > played:
            problems[(side, "deaths")] = f"{r['deaths']} Deaths nach {played} Runden"
    if not any(f in ("kills", "deaths") for _, f in problems):
        for side, other in (("CT", "T"), ("T", "CT")):
            deaths = sum(r["deaths"] for r in rows[SIDES.index(side) * 5: SIDES.index(side) * 5 + 5])
            kills = sum(r["kills"] for r in rows[SIDES.index(other) * 5: SIDES.index(other) * 5 + 5])
            upper = kills + (played if played is not None else 99)
            if not kills - 1 <= deaths <= upper:
                reason = f"Deaths {side} {deaths} passen nicht zu Kills {other} {kills}"
                problems[(side, "deaths")] = reason
                problems[(other, "kills")] = reason
    return problems


def parse_column(text: str, money: bool) -> list[int]:
    """Antwort auf eine Spalte → Zahlen von oben nach unten (eine je Zeile)."""
    values = []
    for line in text.splitlines():
        m = MONEY.search(line) if money else re.search(r"\d+", line)
        if m:
            values.append(int(re.sub(r"\D", "", m.group(1) if money else m.group(0))))
    return values


def build_reading(rows: list[dict], names: str, score: str, highlighted: list[int],
                  notes: list[str] | None = None) -> ScoreboardReading:
    """Geprüfte Zeilen und Antworten → Lesung. Oben stehen die CT, unten die T (je fünf Zeilen)."""
    name_list = parse_names(names)
    ct_score, t_score = parse_score(score)
    uncertain: list[str] = list(notes or [])
    if len(rows) != 10:
        uncertain.append(f"players ({len(rows)} statt 10 Zeilen gelesen)")
        rows = []
    if any(r[f] is None for r in rows for f in FIELDS):
        rows = []
    if len(name_list) != 10:
        if rows:
            uncertain.append(f"players[*].name ({len(name_list)} statt 10 Namen gelesen)")
        name_list = [f"Spieler {i % 5 + 1}" for i in range(10)]
    if ct_score is None or t_score is None:
        uncertain.append("score")
    if len(highlighted) != 1:
        uncertain.append("highlighted")

    def team(offset: int, score_value: int | None) -> TeamReading:
        players = [
            PlayerReading(
                name=name_list[offset + i], money=r["money"], kills=r["kills"], deaths=r["deaths"],
                assists=r["assists"], alive=None, highlighted=highlighted == [offset + i],
            )
            for i, r in enumerate(rows[offset: offset + 5])
        ]
        return TeamReading(score=score_value, alive_count=None, loss_bonus_level=None, players=players)

    return ScoreboardReading(
        map_name=None, game_mode=None,
        ct=team(0, ct_score), t=team(5, t_score),
        round_history=[],
        is_demo_view=bool(rows),
        uncertain_fields=[*uncertain, "round_history (Granite liest die Symbolleiste nicht)"],  # ergänzt enhance()
    )


@lru_cache(maxsize=2)
def _load(model_id: str):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=torch.bfloat16 if device != "cpu" else torch.float32
    )
    return processor, model.to(device), device  # type: ignore[arg-type]


class GraniteReader:
    """Granite Vision lokal: Panel finden, Bereiche ausschneiden, einfache Fragen stellen."""

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self.model = model or DEFAULT_MODEL
        self.panel_source: str | None = None

    def _ask(self, image, question: str, max_new_tokens: int = 400) -> str:
        processor, model, device = _load(self.model)
        conversation = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": question}]}]
        inputs = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(device)
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)  # type: ignore[attr-defined]
        return processor.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    @staticmethod
    def _played(img, panel, score: str) -> int | None:
        """Gespielte Runden: aus der Symbolleiste, sonst aus dem gelesenen Score."""
        from .history_bar import read_history

        rounds, _ = read_history(img, panel)
        if rounds:
            return len(rounds)
        ct, t = parse_score(score)
        return None if ct is None or t is None else ct + t

    def read_column(self, img, panel, field: str, side: str) -> list[int] | None:
        """Eine Spalte eines Teams einzeln lesen, bei falscher Anzahl ein zweites Mal stärker vergrössert."""
        for min_width in (160, 320):
            text = self._ask(crop(img, panel, column_box(field, side), min_width=min_width),
                             COLUMN_PROMPT.format(name=field.capitalize()), 120)
            values = parse_column(text, money=field == "money")
            if len(values) == 5 and (field != "money" or all(v % 50 == 0 and v <= MONEY_CAP for v in values)):
                return values
        return None

    def verified_rows(self, img, panel, rows: list[dict], played: int | None) -> tuple[list[dict], list[str]]:
        """Plausibilität prüfen; betroffene Spalten je Team einzeln nachlesen; Reste als unsicher melden."""
        problems = check_rows(rows, played)
        if not problems:
            return rows, []
        if len(rows) != 10:
            rows = [{f: None for f in FIELDS} for _ in range(10)]
        reread = []
        for side, field in sorted(problems):
            values = self.read_column(img, panel, field, side)
            if values is not None:
                offset = SIDES.index(side) * 5
                for i, v in enumerate(values):
                    rows[offset + i][field] = v
                reread.append(f"{side} {field}")
        remaining = check_rows(rows, played)
        notes = [f"Plausibilitätsprüfung: {', '.join(reread)} einzeln nachgelesen (Bildanalyse)"] if reread else []
        notes += [f"{side.lower()}.{field}: {reason}" for (side, field), reason in sorted(remaining.items())]
        return rows, notes

    def read(self, image: bytes, media_type: str) -> ScoreboardReading:  # noqa: ARG002 (Protokoll)
        from PIL import Image

        img = Image.open(io.BytesIO(image)).convert("RGB")
        panel, self.panel_source = locate_panel(img)
        rows = parse_numbers(self._ask(crop(img, panel, "numbers", min_width=700), NUMBERS_PROMPT, 500))
        names = self._ask(crop(img, panel, "names", min_width=900), NAMES_PROMPT, 300)
        score = self._ask(crop(img, panel, "scores", min_width=220), SCORE_PROMPT, 200)
        played = self._played(img, panel, score)
        rows, notes = self.verified_rows(img, panel, rows, played)
        reading = build_reading(rows, names, score, highlighted_rows(img, panel), notes)
        if self.panel_source != "erkannt":
            reading.uncertain_fields.append("panel (nicht erkannt, Standardlage angenommen)")
        return reading
