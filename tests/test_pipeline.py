import importlib.util
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from cs2_money_round.demo_import import _team_won
from cs2_money_round.features import LABEL, compute_features, historical_rates, history_features
from cs2_money_round.store import FeatureGroupSpec, FeatureViewSpec, LocalStore
from cs2_money_round.vision.schema import PlayerReading, RoundResult, ScoreboardReading, TeamReading

ROOT = Path(__file__).resolve().parent.parent


def load_pipeline(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "pipelines" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


inference = load_pipeline("inference_pipeline")


class HistoryFeatureTests(unittest.TestCase):
    def test_loss_level_rises_and_falls_within_half(self):
        f = history_features("WeLeLeLe", round_number=5)
        self.assertEqual(f, {"loss_streak": 3, "loss_bonus_level": 3, "wins_last_3": 0})
        self.assertEqual(history_features("LeLeWe", round_number=4)["loss_bonus_level"], 2)

    def test_second_half_starts_fresh(self):
        first_half = "Le" * 12
        f = history_features(first_half + "We", round_number=14)
        self.assertEqual(f, {"loss_streak": 0, "loss_bonus_level": 0, "wins_last_3": 1})

    def test_compute_features_reads_previous_round(self):
        obs = pd.DataFrame([{
            "round_number": 4, "team_side": "CT", "obs_round_history": "WeLbLd",
            "obs_score_team": 1, "obs_score_opp": 2, "obs_prev_round_won": False,
            "obs_prev_round_reason": "defuse", "obs_team_kills_total": 9, "obs_team_deaths_total": 12,
        }])
        x = compute_features(obs).iloc[0]
        self.assertEqual((x.is_ct, x.prev_round_won, x.prev_reason_defuse, x.kill_death_diff), (1, 0, 1, -3))
        self.assertAlmostEqual(x.deaths_per_round, 4.0)


class HistoricalRateTests(unittest.TestCase):
    def test_rate_uses_only_earlier_matches(self):
        t0 = pd.Timestamp("2026-10-01", tz="UTC")
        df = pd.DataFrame({
            "match_id": ["a", "a", "b", "c"],
            "match_time": [t0, t0, t0 + pd.Timedelta(days=1), t0 + pd.Timedelta(days=2)],
            "team_side": ["T"] * 4,
            "loss_bonus_level": [1] * 4,
            LABEL: [1, 1, 0, 1],
        })
        rates = historical_rates(df)
        self.assertEqual(rates.iloc[0], 0.5)            # erstes Match: nur Prior, nicht das eigene Label
        self.assertAlmostEqual(rates.iloc[2], 7 / 12)   # b sieht a (2/2), nicht c
        self.assertAlmostEqual(rates.iloc[3], 7 / 13)   # c sieht a und b (2/3)


class HalftimeTests(unittest.TestCase):
    """Regression: nach der Halbzeit wechseln die Teams die Seite (Score und Historie folgen dem Team)."""

    def test_demo_import_maps_round_winner_by_player_ids(self):
        first_half_sides = {"1": 2, "2": 2, "3": 3}       # Spieler 1,2 waren T
        self.assertTrue(_team_won(first_half_sides, {"1", "2"}, "T"))
        self.assertFalse(_team_won(first_half_sides, {"3"}, "T"))
        self.assertIsNone(_team_won(first_half_sides, {"9"}, "T"))

    def test_screenshot_history_flips_first_half_for_current_side(self):
        rounds = [RoundResult(round_number=i, winner="T", reason="elimination") for i in range(1, 13)]
        rounds.append(RoundResult(round_number=13, winner="CT", reason="defuse"))
        reading = ScoreboardReading(
            map_name=None, game_mode=None, round_history=rounds, is_demo_view=False, uncertain_fields=[],
            ct=TeamReading(score=13, alive_count=5, loss_bonus_level=None, players=[]),
            t=TeamReading(score=0, alive_count=5, loss_bonus_level=None, players=[]),
        )
        # Heute CT = in Halbzeit 1 T: hat alle 12 Runden + Runde 13 gewonnen.
        self.assertEqual(inference.history_from_reading(reading, "CT", 14), "We" * 12 + "Wd")
        self.assertEqual(inference.history_from_reading(reading, "T", 14), "Le" * 12 + "Ld")
        self.assertIsNone(inference.history_from_reading(reading, "CT", 15))  # Runde fehlt

    def test_target_is_team_without_highlight(self):
        def players(highlight):
            return [PlayerReading(name=str(i), money=None, kills=0, deaths=0, assists=0, alive=True,
                                  highlighted=highlight and i == 0) for i in range(5)]
        reading = ScoreboardReading(
            map_name=None, game_mode=None, round_history=[], is_demo_view=False, uncertain_fields=[],
            ct=TeamReading(score=0, alive_count=5, loss_bonus_level=None, players=players(True)),
            t=TeamReading(score=0, alive_count=5, loss_bonus_level=None, players=players(False)),
        )
        self.assertEqual(inference.target_side(reading, None), "T")
        self.assertEqual(inference.target_side(reading, "CT"), "CT")


class LocalStoreTests(unittest.TestCase):
    def test_upsert_split_lookup_and_model_versions(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalStore(Path(tmp))
            fg = FeatureGroupSpec("g", 1, "", ["k"], "t", online=True)
            view = FeatureViewSpec("v", 1, "", fg, ["x"], ["y"])
            t = pd.Timestamp("2026-10-01", tz="UTC")
            store.insert(fg, pd.DataFrame({"k": [1, 2], "t": [t, t + pd.Timedelta(days=2)], "x": [0.1, 0.2], "y": [0, 1]}))
            store.insert(fg, pd.DataFrame({"k": [2], "t": [t + pd.Timedelta(days=2)], "x": [0.9], "y": [1]}))
            self.assertEqual(len(store.read_group(fg, ["k"])), 2)
            X_tr, X_te, y_tr, y_te = store.train_test_split(view, t + pd.Timedelta(days=1), t, t + pd.Timedelta(days=3))
            self.assertEqual((len(X_tr), len(X_te), X_te["x"].iloc[0]), (1, 1, 0.9))
            self.assertEqual(store.feature_vector(view, {"k": 2}), {"x": 0.9})
            self.assertIsNone(store.feature_vector(view, {"k": 7}))

            model_dir = Path(tmp) / "m"
            model_dir.mkdir()
            (model_dir / "model.pkl").write_bytes(b"x")
            for _ in range(2):
                store.save_model("demo", model_dir, {"roc_auc": 0.7}, "", X_tr, view)
            latest = store.latest_model("demo")
            self.assertEqual((latest.version, latest.metrics["roc_auc"]), (2, 0.7))


if __name__ == "__main__":
    unittest.main()
