"""Austauschbare Vision-Reader für Scoreboard-Screenshots.

Auswahl über Umgebungsvariablen (siehe ``.env.example``):

- ``VISION_PROVIDER=anthropic`` → Claude über die Anthropic-API (``ANTHROPIC_API_KEY``)
- ``VISION_PROVIDER=openai_compatible`` → lokales Modell mit OpenAI-kompatibler
  Schnittstelle (z. B. oMLX, LM Studio, Ollama), ``VISION_BASE_URL`` + optional ``VISION_API_KEY``

Beide liefern ein validiertes :class:`ScoreboardReading`. Wer einen weiteren
Anbieter braucht, implementiert ``read(image_bytes, media_type)``.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Protocol

from .schema import ScoreboardReading

PROMPT = """Du liest einen Screenshot aus Counter-Strike 2 aus. Massgeblich ist das Scoreboard in der Bildmitte.

Lies genau ab, was sichtbar ist, und rate nie:
- Oben steht das Team COUNTER-TERRORISTS (CT), unten TERRORISTS (T). Die grosse Zahl links ist die Rundenzahl dieses Teams.
- Pro Spielerzeile: Name, Money, Kills, Deaths, Assists. Ist die Money-Zelle leer oder verdeckt, setze money auf null.
- Die hervorgehobene Zeile (farbiger Balken) bekommt highlighted=true, alle anderen false.
- 'Alive: x/5' unter dem Teamnamen ist alive_count.
- Loss Bonus: Anzahl gefüllter Striche/Balken je Team. Ist für ein Team nichts angezeigt, null.
- Rundenhistorie (Symbolleiste zwischen den Teams): je gespielter Runde Sieger und Grund
  (Totenkopf = elimination, Bombe/Explosion = bomb, Zange/Entschärfung = defuse, Uhr = time).
  Wenn die Leiste nicht eindeutig lesbar ist, gib eine leere Liste zurück.
- HUD-Elemente ausserhalb des Scoreboards nur nutzen, um einen Wert zu bestätigen, nicht um ihn zu ersetzen.
- Trage jedes Feld, bei dem du unsicher bist, in uncertain_fields ein."""


class VisionReader(Protocol):
    def read(self, image: bytes, media_type: str) -> ScoreboardReading: ...


def _media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".png":
        return "image/png"
    if suffix in (".jpg", ".jpeg"):
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    raise ValueError(f"Nicht unterstütztes Bildformat: {path.suffix}")


class AnthropicReader:
    """Claude mit strukturierter Ausgabe (``messages.parse``) und serverseitigem Fallback bei Ablehnung."""

    def __init__(self, model: str) -> None:
        import anthropic

        self._anthropic = anthropic
        self._client = anthropic.Anthropic()  # liest ANTHROPIC_API_KEY
        self.model = model

    def read(self, image: bytes, media_type: str) -> ScoreboardReading:
        response = self._client.beta.messages.parse(
            model=self.model,
            max_tokens=16000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_config={"effort": "medium"},
            output_format=ScoreboardReading,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": base64.standard_b64encode(image).decode("ascii"),
                            },
                        },
                        {"type": "text", "text": PROMPT},
                    ],
                }
            ],
        )
        if response.stop_reason == "refusal":
            raise RuntimeError(f"Vision-Modell hat abgelehnt: {response.stop_details}")
        if response.stop_reason == "max_tokens" or response.parsed_output is None:
            raise RuntimeError(f"Keine vollständige Ausgabe (stop_reason={response.stop_reason})")
        return response.parsed_output


class OpenAICompatibleReader:
    """Lokales Modell über eine OpenAI-kompatible API (z. B. oMLX) mit JSON-Schema-Ausgabe."""

    def __init__(self, model: str, base_url: str, api_key: str | None) -> None:
        from openai import OpenAI

        self._client = OpenAI(base_url=base_url, api_key=api_key or "local")
        self.model = model

    def read(self, image: bytes, media_type: str) -> ScoreboardReading:
        data_url = f"data:{media_type};base64,{base64.standard_b64encode(image).decode('ascii')}"
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": PROMPT},
                    ],
                }
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "scoreboard_reading", "schema": ScoreboardReading.model_json_schema()},
            },
        )
        content = response.choices[0].message.content or ""
        return ScoreboardReading.model_validate(json.loads(content))


class ClaudeCliReader:
    """Claude Code im Headless-Modus (``claude -p``) mit JSON-Schema-Ausgabe.

    Für die Entwicklung ohne API-Key: nutzt die Anmeldung der lokalen Claude-Code-
    Installation. Das Bild wird über das Read-Werkzeug gelesen; andere Werkzeuge sind aus.
    """

    def __init__(self, model: str, binary: str = "claude") -> None:
        self.model = model
        self.binary = binary

    def read(self, image: bytes, media_type: str) -> ScoreboardReading:
        import subprocess
        import tempfile

        suffix = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}[media_type]
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / f"screenshot{suffix}"
            image_path.write_bytes(image)
            # Der API-Key aus .env gehört zum AnthropicReader; die CLI nutzt ihre eigene Anmeldung.
            env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
            result = subprocess.run(
                [
                    self.binary, "-p", f"{PROMPT}\n\nLies zuerst das Bild {image_path} mit dem Read-Werkzeug.",
                    "--model", self.model,
                    "--output-format", "json",
                    "--json-schema", json.dumps(ScoreboardReading.model_json_schema()),
                    "--tools", "Read",
                    "--allowedTools", "Read",
                    "--add-dir", tmp,
                    "--no-session-persistence",
                ],
                capture_output=True, text=True, timeout=600, env=env, cwd=tmp,
            )
        if result.returncode != 0 and not result.stdout.strip():
            raise RuntimeError(f"claude -p fehlgeschlagen ({result.returncode}): {result.stderr[-500:]}")
        messages = json.loads(result.stdout)
        messages = messages if isinstance(messages, list) else [messages]
        final = next((m for m in reversed(messages) if m.get("type") == "result"), {})
        payload = final.get("structured_output")
        if final.get("is_error") or payload is None:
            raise RuntimeError(f"Keine strukturierte Ausgabe von claude -p: {str(final.get('result'))[:500]}")
        return ScoreboardReading.model_validate(payload)


def reader_from_env() -> VisionReader:
    provider = os.environ.get("VISION_PROVIDER", "anthropic")
    model = os.environ.get("VISION_MODEL", "claude-sonnet-5-5")
    if provider == "anthropic":
        return AnthropicReader(model)
    if provider == "claude_cli":
        return ClaudeCliReader(model)
    if provider == "openai_compatible":
        return OpenAICompatibleReader(model, os.environ["VISION_BASE_URL"], os.environ.get("VISION_API_KEY"))
    raise ValueError(f"Unbekannter VISION_PROVIDER: {provider}")


def read_screenshot(path: str | Path, reader: VisionReader | None = None) -> ScoreboardReading:
    path = Path(path)
    return (reader or reader_from_env()).read(path.read_bytes(), _media_type(path))
