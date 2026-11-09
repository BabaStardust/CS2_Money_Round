"""Screenshot → strukturierte Scoreboard-Werte (austauschbares Vision-Modell)."""

from .readers import read_screenshot, reader_from_env
from .schema import ScoreboardReading

__all__ = ["ScoreboardReading", "read_screenshot", "reader_from_env"]
