"""Web-App und Beispiel-Store: Inferenz ohne Hopsworks-Zugang, Umschalten der Quelle."""

import os
import unittest
from unittest import mock

from fastapi.testclient import TestClient

from cs2_money_round.inference import load_model, observation, predict
from cs2_money_round.specs import MODEL_NAME
from cs2_money_round.store import ExampleStore, available_stores


class ExampleStoreTests(unittest.TestCase):
    def test_example_store_runs_inference_without_hopsworks(self):
        store = ExampleStore()
        model = load_model(store)
        self.assertEqual(model.version, store.latest_model(MODEL_NAME).version)
        obs, _ = observation("T", "WeLbLeLe", 14, 20)
        result = predict(obs, store, model)
        assert result.hist_rate is not None and result.probability is not None
        self.assertGreater(result.hist_rate["n_rounds"], 0)
        self.assertTrue(0 < result.probability < 1)
        self.assertFalse(result.possible)  # drei Niederlagen in Folge, Loss Bonus 3

    def test_example_store_is_read_only(self):
        with self.assertRaises(PermissionError):
            ExampleStore().insert(None, None)

    def test_hopsworks_only_offered_with_key(self):
        with mock.patch.dict(os.environ, {"HOPSWORKS_API_KEY": ""}):
            self.assertNotIn("hopsworks", available_stores())
        with mock.patch.dict(os.environ, {"HOPSWORKS_API_KEY": "x"}):
            self.assertIn("hopsworks", available_stores())


class AppTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(os.environ, {"HOPSWORKS_API_KEY": "", "APP_FEATURE_STORE": ""})
        patcher.start()
        self.addCleanup(patcher.stop)
        from app.server import app

        self.client = TestClient(app)

    def test_status_defaults_to_example_store(self):
        s = self.client.get("/api/status").json()
        self.assertEqual(s["default_store"], "example")
        self.assertNotIn("hopsworks", s["stores"])

    def test_predict_with_selected_store(self):
        body = {"side": "T", "history": "WeLbLeLe", "kills": 14, "deaths": 20, "store": "example"}
        res = self.client.post("/api/predict", json=body)
        self.assertEqual(res.status_code, 200, res.text)
        self.assertEqual(res.json()["store"], "example")
        self.assertEqual(self.client.get("/api/model", params={"store": "example"}).status_code, 200)

    def test_unavailable_store_is_rejected(self):
        body = {"side": "T", "history": "WeLe", "kills": 1, "deaths": 2, "store": "hopsworks"}
        self.assertEqual(self.client.post("/api/predict", json=body).status_code, 400)

    def test_invalid_history_is_explained(self):
        body = {"side": "T", "history": "WeLx", "kills": 1, "deaths": 2}
        res = self.client.post("/api/predict", json=body)
        self.assertEqual(res.status_code, 422)
        self.assertIn("Rundenhistorie", res.json()["detail"])


if __name__ == "__main__":
    unittest.main()
