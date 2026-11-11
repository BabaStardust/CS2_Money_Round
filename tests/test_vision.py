"""Bildanalyse und Vision-Parsing an den ausgelieferten Beispiel-Screenshots (Wahrheit aus den Demos)."""

import json
import unittest
from pathlib import Path

from PIL import Image

from cs2_money_round.inference import history_from_reading, target_side
from cs2_money_round.vision.granite import (
    GraniteReader, build_reading, check_rows, parse_column, parse_numbers, parse_score,
)
from cs2_money_round.vision.history_bar import read_history
from cs2_money_round.vision.readers import enhance
from cs2_money_round.vision.regions import highlighted_rows, locate_panel
from cs2_money_round.vision.schema import PlayerReading, ScoreboardReading, TeamReading

EXAMPLES = Path(__file__).resolve().parent.parent / "examples" / "screenshots"
TRUTH = json.loads((EXAMPLES / "truth.json").read_text())


def load(name: str) -> Image.Image:
    return Image.open(EXAMPLES / name).convert("RGB")


def code(rounds, side: str, round_now: int) -> str:
    """RoundResults → 'WeLb…' aus Sicht von ``side`` (gleiche Logik wie die Inferenz)."""
    players = [PlayerReading(name="x", money=None, kills=0, deaths=0, assists=0, alive=None, highlighted=False)]
    reading = ScoreboardReading(
        map_name=None, game_mode=None, round_history=rounds, is_demo_view=True, uncertain_fields=[],
        ct=TeamReading(score=None, alive_count=None, loss_bonus_level=None, players=players),
        t=TeamReading(score=None, alive_count=None, loss_bonus_level=None, players=players),
    )
    return history_from_reading(reading, side, round_now) or ""


class PanelTests(unittest.TestCase):
    def test_panel_found_in_crop_and_further_crops(self):
        for name in TRUTH:
            img = load(name)
            (x0, y0, x1, y1), how = locate_panel(img)
            self.assertEqual(how, "erkannt", name)
            self.assertAlmostEqual((x1 - x0) / (y1 - y0), 1424 / 800, delta=0.05, msg=name)
            # weiterer Zuschnitt verschiebt das Panel nur
            (cx0, cy0, _, _), _ = locate_panel(img.crop((20, 30, img.width - 10, img.height - 5)))
            self.assertAlmostEqual(cx0, x0 - 20, delta=6, msg=name)
            self.assertAlmostEqual(cy0, y0 - 30, delta=6, msg=name)

    def test_highlighted_row_is_own_player_in_team_colour(self):
        for name, t in TRUTH.items():
            img = load(name)
            rows = highlighted_rows(img, locate_panel(img)[0])
            own_is_ct = t["gegner"] == "T"
            self.assertEqual(len(rows), 1, name)
            self.assertEqual(rows[0] < 5, own_is_ct, name)


class HistoryBarTests(unittest.TestCase):
    def test_history_matches_demo_including_reasons(self):
        for name, t in TRUTH.items():
            img = load(name)
            rounds, notes = read_history(img, locate_panel(img)[0])
            self.assertEqual(code(rounds, t["gegner"], t["runde"]), t["historie_gegner"], name)
            self.assertEqual(notes, [], name)

    def test_history_survives_half_resolution(self):
        name = "dust2_runde15_gegner-CT.png"
        img = load(name)
        small = img.resize((img.width // 2, img.height // 2))
        rounds, _ = read_history(small, locate_panel(small)[0])
        self.assertEqual(code(rounds, "CT", 15), TRUTH[name]["historie_gegner"])


class GraniteParsingTests(unittest.TestCase):
    # echte Granite-Antworten auf die Ausschnitte (Zahlenspalten, Rundenzahlen)
    NUMBERS = ",Money,Kills,Deaths,Assists\n0,$2600,2,1,0\n1,$2000,0,1,1\n2,$2000,0,1,0\n3,$2000,0,1,0\n4,$2300,0,1,0\n" \
              "5,$4900,5,0,0\n6,$4050,0,1,2\n7,$3850,0,0,0\n8,$3550,0,0,0\n9,$3400,0,1,0\n"
    SCORE = "The top big number ... is 0, and the bottom ... is 1. Therefore CT=<n> T=<n> would be:\n\nCT = 0\nT = 1\n"

    def test_numbers_and_score(self):
        rows = parse_numbers(self.NUMBERS)
        self.assertEqual(len(rows), 10)
        self.assertEqual(rows[0], {"money": 2600, "kills": 2, "deaths": 1, "assists": 0})
        self.assertEqual(rows[5]["money"], 4900)
        self.assertEqual(parse_score(self.SCORE), (0, 1))

    def test_build_reading_assigns_teams_and_highlight(self):
        reading = build_reading(parse_numbers(self.NUMBERS), "a\nb", self.SCORE, highlighted=[4])
        self.assertEqual([p.money for p in reading.t.players], [4900, 4050, 3850, 3550, 3400])
        self.assertTrue(reading.ct.players[4].highlighted)
        self.assertEqual(target_side(reading, None), "T")
        self.assertEqual(reading.ct.players[0].name, "Spieler 1")  # Namen unvollständig → Platzhalter

    def test_incomplete_table_is_flagged_not_guessed(self):
        reading = build_reading(parse_numbers("$2600,2,1,0\n"), "", "CT = 0", highlighted=[])
        self.assertEqual(reading.ct.players, [])
        self.assertTrue(any(f.startswith("players") for f in reading.uncertain_fields))


def rows(money, kills, deaths, assists=None):
    return [{"money": m, "kills": k, "deaths": d, "assists": a}
            for m, k, d, a in zip(money, kills, deaths, assists or [0] * 10)]


# Dust II Runde 7 (Wahrheit vom Bild): CT oben, T unten; 6 Runden gespielt, ein Bombentod nach Rundenende
R07 = rows([5400, 4350, 4750, 3600, 5250, 12650, 10900, 4550, 15150, 6400],
           [3, 3, 4, 3, 1, 8, 7, 5, 4, 3], [6, 5, 4, 6, 6, 2, 2, 4, 3, 4])


class PlausibilityTests(unittest.TestCase):
    def test_true_scoreboard_passes(self):
        self.assertEqual(check_rows(R07, played=6), {})

    def test_misreads_are_caught(self):
        wrong_money = [dict(r) for r in R07]
        wrong_money[0]["money"] = 5410
        self.assertIn(("CT", "money"), check_rows(wrong_money, 6))
        too_many_deaths = [dict(r) for r in R07]
        too_many_deaths[6]["deaths"] = 7  # mehr Deaths als Runden
        self.assertIn(("T", "deaths"), check_rows(too_many_deaths, 6))
        dropped = [dict(r) for r in R07]
        dropped[7]["deaths"] = 0  # Deaths T 11 < Kills CT 14 − 1
        problems = check_rows(dropped, 6)
        self.assertIn(("T", "deaths"), problems)
        self.assertIn(("CT", "kills"), problems)
        self.assertEqual(len(check_rows(R07[:9], 6)), 8)  # fehlende Zeile: alles nachlesen

    def test_parse_column(self):
        self.assertEqual(parse_column("$12650\n$10900\n$4550", money=True), [12650, 10900, 4550])
        self.assertEqual(parse_column("8\n7\n5\n4\n3", money=False), [8, 7, 5, 4, 3])

    def test_failing_column_is_reread_per_team(self):
        class FakeGranite(GraniteReader):
            asked: list = []

            def _ask(self, image, question, max_new_tokens=400):
                self.asked.append(question)
                return "2\n2\n4\n3\n4"  # Deaths T, korrekt

        broken = [dict(r) for r in R07]
        broken[6]["deaths"] = None
        img = load("dust2_runde05_gegner-T.png")
        fixed, notes = FakeGranite().verified_rows(img, locate_panel(img)[0], broken, played=6)
        self.assertEqual([r["deaths"] for r in fixed[5:]], [2, 2, 4, 3, 4])
        self.assertEqual(len(FakeGranite.asked), 1)
        self.assertIn("einzeln nachgelesen", notes[0])
        self.assertEqual(len(notes), 1)  # nach dem Nachlesen plausibel


class EnhanceTests(unittest.TestCase):
    def test_bar_fixes_misread_score_and_fills_history(self):
        name = "anubis_runde03_gegner-CT.png"
        players = [PlayerReading(name=str(i), money=1000, kills=1, deaths=1, assists=0, alive=None, highlighted=False)
                   for i in range(5)]
        reading = ScoreboardReading(
            map_name=None, game_mode=None, round_history=[], is_demo_view=True, uncertain_fields=[],
            ct=TeamReading(score=2, alive_count=None, loss_bonus_level=None, players=players),
            t=TeamReading(score=5, alive_count=None, loss_bonus_level=None, players=[p.model_copy() for p in players]),
        )
        out = enhance(reading, (EXAMPLES / name).read_bytes())
        self.assertEqual((out.ct.score, out.t.score), (2, 0))
        self.assertEqual(code(out.round_history, "CT", 3), TRUTH[name]["historie_gegner"])
        self.assertTrue(out.t.players[4].highlighted)  # eigene Zeile gelb auf T
        self.assertEqual(target_side(out, None), "CT")


if __name__ == "__main__":
    unittest.main()
