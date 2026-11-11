"""Web-App für die Inferenz: Screenshot hochladen oder Werte von Hand eingeben → Full-Buy-Prognose.

Start (lokal):
    uv run uvicorn app.server:app --port 8765
    # mit Granite als lokalem Vision-Modell:
    uv run --extra granite uvicorn app.server:app --port 8765

Die App ruft dieselben Funktionen wie die Inference Pipeline auf (``cs2_money_round.inference``):
Featurestore und Model Registry kommen aus ``FEATURE_STORE`` (Hopsworks oder lokal).
"""

from __future__ import annotations

import importlib.util
import io
import os
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from cs2_money_round.inference import (
    InferenceError, LoadedModel, demo_money_check, history_from_reading, load_model, observation,
    predict, target_side, team_totals,
)
from cs2_money_round.store import STORE_KINDS, available_stores, make_store
from cs2_money_round.vision import PROVIDERS, ScoreboardReading, make_reader, read_image

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"
EXAMPLES = ROOT / "examples" / "screenshots"
READING_DIR = ROOT / "data" / "inference"
MAX_UPLOAD = 20 * 1024 * 1024
MEDIA_TYPES = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}

load_dotenv(ROOT / ".env")
app = FastAPI(title="CS2 Full-Buy-Prognose")

_lock = threading.Lock()  # Store und Modell je Quelle nur einmal laden
_vision_lock = threading.Lock()  # lokale Vision-Modelle (Granite) nicht parallel nutzen
_stores: dict = {}
_models: dict[str, LoadedModel] = {}


def default_store() -> str | None:
    """Standard der Web-App: der mitgelieferte Beispiel-Store (läuft ohne Zugang), sonst ``FEATURE_STORE``.

    ``APP_FEATURE_STORE`` überschreibt den Standard, z. B. ``APP_FEATURE_STORE=hopsworks``.
    """
    stores = available_stores()
    for kind in (os.environ.get("APP_FEATURE_STORE"), "example", os.environ.get("FEATURE_STORE")):
        if kind in stores:
            return kind
    return stores[0] if stores else None


def _check_store(kind: str | None) -> str:
    kind = kind or default_store()
    if kind not in available_stores():
        raise HTTPException(400, f"Featurestore {kind} ist hier nicht verfügbar.")
    return kind  # type: ignore[return-value]


def _store(kind: str):
    with _lock:
        if kind not in _stores:
            _stores[kind] = make_store(kind)
        return _stores[kind]


def _model(kind: str) -> LoadedModel:
    store = _store(kind)
    with _lock:
        if kind not in _models:
            _models[kind] = load_model(store)
        return _models[kind]


def available_providers() -> dict[str, str]:
    """Nur Anbieter, die in dieser Umgebung laufen können."""
    out = {}
    if shutil.which("claude"):
        out["claude_cli"] = PROVIDERS["claude_cli"]
    if os.environ.get("ANTHROPIC_API_KEY"):
        out["anthropic"] = PROVIDERS["anthropic"]
    if importlib.util.find_spec("transformers") and importlib.util.find_spec("torch"):
        out["granite"] = PROVIDERS["granite"]
    if os.environ.get("VISION_BASE_URL"):
        out["openai_compatible"] = PROVIDERS["openai_compatible"]
    return out


@app.get("/api/status")
def status() -> dict:
    """Vision-Modelle und Featurestores, die in dieser Umgebung laufen, mit den Standards."""
    providers = available_providers()
    default = os.environ.get("VISION_PROVIDER", "anthropic")
    return {
        "stores": {k: STORE_KINDS[k] for k in available_stores()},
        "default_store": default_store(),
        "providers": providers,
        "default_provider": default if default in providers else next(iter(providers), None),
    }


@app.get("/api/model")
def model_info(store: str | None = None) -> dict:
    """Modell der gewählten Quelle laden (Hopsworks: beim ersten Mal einige Sekunden)."""
    kind = _check_store(store)
    try:
        m = _model(kind)
    except Exception as e:  # noqa: BLE001 – Fehler der Registry dem User zeigen
        raise HTTPException(502, f"Modell aus {STORE_KINDS[kind]} nicht ladbar: {type(e).__name__}: {e}") from e
    return {"store": kind, "label": STORE_KINDS[kind], "version": m.version,
            "roc_auc": m.metrics.get("roc_auc"), "accuracy": m.metrics.get("accuracy")}


@app.get("/api/examples")
def examples() -> list[dict]:
    """Beispiel-Screenshots in ``examples/screenshots`` (mit optionaler Beschreibung ``<name>.txt``)."""
    if not EXAMPLES.is_dir():
        return []
    out = []
    for p in sorted(EXAMPLES.iterdir()):
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
            note = p.with_suffix(".txt")
            out.append({"name": p.name, "note": note.read_text().strip() if note.exists() else ""})
    return out


@app.get("/examples/{name}")
def example_file(name: str):
    path = (EXAMPLES / name).resolve()
    if path.parent != EXAMPLES.resolve() or not path.is_file():
        raise HTTPException(404, "Beispiel nicht gefunden")
    return FileResponse(path)


def _image(data: bytes) -> str:
    """Prüft, dass die Datei ein Bild ist; gibt den Media-Type zurück."""
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(io.BytesIO(data)) as im:
            fmt = im.format or ""
            im.verify()
    except (UnidentifiedImageError, OSError) as e:
        raise HTTPException(400, "Die Datei ist kein lesbares Bild (PNG, JPEG oder WebP).") from e
    if fmt not in MEDIA_TYPES:
        raise HTTPException(400, f"Bildformat {fmt or 'unbekannt'} wird nicht unterstützt (PNG, JPEG oder WebP).")
    return MEDIA_TYPES[fmt]


def derive(reading: ScoreboardReading) -> dict:
    """Vorschlag je möglicher Gegnerseite: Rundenhistorie, Kills, Deaths (None = nicht lesbar)."""
    out = {}
    for side in ("CT", "T"):
        team, opp = (reading.ct, reading.t) if side == "CT" else (reading.t, reading.ct)
        history = None
        if team.score is not None and opp.score is not None:
            history = history_from_reading(reading, side, team.score + opp.score + 1)
        kills, deaths = team_totals(reading, side)
        out[side] = {
            "history": history,
            "kills": kills,
            "deaths": deaths,
            "score": [team.score, opp.score],
            "demo_check": demo_money_check(reading, side),
        }
    return out


@app.post("/api/read")
def read(provider: str = Form(...), file: UploadFile | None = File(None), example: str | None = Form(None)) -> dict:
    """Schritt 1 (Live-Daten): Screenshot mit dem gewählten Vision-Modell lesen."""
    if provider not in available_providers():
        raise HTTPException(400, f"Vision-Modell {provider} ist hier nicht verfügbar.")
    if file is not None:
        data = file.file.read(MAX_UPLOAD + 1)
        name = Path(file.filename or "upload").stem
    elif example:
        path = (EXAMPLES / example).resolve()
        if path.parent != EXAMPLES.resolve() or not path.is_file():
            raise HTTPException(404, "Beispiel nicht gefunden")
        data, name = path.read_bytes(), path.stem
    else:
        raise HTTPException(400, "Bitte einen Screenshot hochladen oder ein Beispiel wählen.")
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "Bild ist grösser als 20 MB.")
    media_type = _image(data)

    started = time.monotonic()
    try:
        with _vision_lock:
            reading = read_image(data, media_type, make_reader(provider))
    except Exception as e:  # noqa: BLE001 – Fehler des Vision-Modells dem User zeigen
        raise HTTPException(502, f"Vision-Modell fehlgeschlagen: {type(e).__name__}: {e}") from e
    seconds = round(time.monotonic() - started, 1)

    READING_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    (READING_DIR / f"{stamp}-{name}-{provider}.reading.json").write_text(reading.model_dump_json(indent=2))
    return {
        "reading": reading.model_dump(),
        "derived": derive(reading),
        "default_side": target_side(reading, None),
        "provider": provider,
        "seconds": seconds,
    }


class PredictRequest(BaseModel):
    side: str = Field(pattern="^(CT|T)$")
    history: str
    kills: int = Field(ge=0, le=999)
    deaths: int = Field(ge=0, le=999)
    reading: ScoreboardReading | None = None
    store: str | None = None  # example | hopsworks | local; leer = Standard der App


@app.post("/api/predict")
def predict_route(req: PredictRequest) -> dict:
    """Schritte 2–4: Kaufquote aus dem Featurestore, Modell aus der Registry, Prediction."""
    try:
        obs, notes = observation(req.side, req.history, req.kills, req.deaths, req.reading)
        kind = _check_store(req.store)
        result = predict(obs, _store(kind), _model(kind), notes)
    except InferenceError as e:
        raise HTTPException(422, str(e)) from e
    rate = result.hist_rate
    return {
        "side": result.side,
        "round_number": result.round_number,
        "score": result.score,
        "history": result.history,
        "skipped": result.skipped,
        "probability": result.probability,
        "possible": result.possible,
        "confidence": result.confidence,
        "rt": result.rt,
        "agg": result.agg,
        "hist_rate": None if rate is None else {**rate, "updated_at": rate["updated_at"].isoformat()},
        "model_version": result.model_version,
        "model_auc": result.model_auc,
        "notes": result.notes,
        "store": kind,
        "store_label": STORE_KINDS[kind],
    }


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
