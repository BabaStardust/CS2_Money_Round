"""Screenshot → strukturierte Scoreboard-Werte (austauschbares Vision-Modell)."""

from .readers import PROVIDERS, enhance, make_reader, read_image, read_screenshot, reader_from_env
from .schema import ScoreboardReading

__all__ = [
    "PROVIDERS", "ScoreboardReading", "enhance", "make_reader", "read_image", "read_screenshot", "reader_from_env",
]
