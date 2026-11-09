"""Featurestore und Model Registry: Hopsworks oder lokaler Ersatz mit derselben Schnittstelle.

Die Pipelines sprechen nur mit dieser Schnittstelle. ``FEATURE_STORE=hopsworks``
(Standard) nutzt Hopsworks mit ``HOPSWORKS_API_KEY`` und ``HOPSWORKS_PROJECT``;
``FEATURE_STORE=local`` legt Feature Groups als Parquet unter ``data/featurestore/``
und Modelle unter ``models/registry/`` ab. Der lokale Ersatz bildet nur nach, was
die Pipelines brauchen (Upsert per Primary Key, zeitlicher Split, Lookup per Key,
versionierte Modelle). Er ist kein Hopsworks-Ersatz für den Betrieb.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class FeatureGroupSpec:
    name: str
    version: int
    description: str
    primary_key: list[str]
    event_time: str
    online: bool = False


@dataclass(frozen=True)
class FeatureViewSpec:
    name: str
    version: int
    description: str
    group: FeatureGroupSpec
    columns: list[str]
    labels: list[str]


@dataclass
class ModelHandle:
    name: str
    version: int
    path: Path
    metrics: dict


class Store(Protocol):
    kind: str

    def insert(self, spec: FeatureGroupSpec, df: pd.DataFrame) -> None: ...
    def read_group(self, spec: FeatureGroupSpec, columns: list[str]) -> pd.DataFrame: ...
    def train_test_split(
        self, view: FeatureViewSpec, split: pd.Timestamp, start: pd.Timestamp, end: pd.Timestamp
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: ...
    def feature_vector(self, view: FeatureViewSpec, entry: dict) -> dict | None: ...
    def save_model(self, name: str, model_dir: Path, metrics: dict, description: str,
                   input_example: pd.DataFrame, view: FeatureViewSpec) -> int: ...
    def latest_model(self, name: str) -> ModelHandle: ...


# --------------------------------------------------------------------------- Hopsworks
class HopsworksStore:
    kind = "hopsworks"

    def __init__(self) -> None:
        import hopsworks

        self.project = hopsworks.login(
            api_key_value=os.environ["HOPSWORKS_API_KEY"], project=os.environ["HOPSWORKS_PROJECT"]
        )
        self.fs = self.project.get_feature_store()

    def _group(self, spec: FeatureGroupSpec):
        return self.fs.get_or_create_feature_group(
            name=spec.name,
            version=spec.version,
            description=spec.description,
            primary_key=spec.primary_key,
            event_time=spec.event_time,
            online_enabled=spec.online,
        )

    def _view(self, view: FeatureViewSpec):
        fg = self.fs.get_feature_group(view.group.name, version=view.group.version)
        return self.fs.get_or_create_feature_view(
            name=view.name,
            version=view.version,
            description=view.description,
            query=fg.select(view.columns + view.labels),
            labels=view.labels,
        )

    def insert(self, spec, df):
        self._group(spec).insert(df, write_options={"wait_for_job": True})

    def read_group(self, spec, columns):
        return self.fs.get_feature_group(spec.name, version=spec.version).select(columns).read()

    def train_test_split(self, view, split, start, end):
        return self._view(view).train_test_split(
            train_start=start, train_end=split, test_start=split, test_end=end,
            description=f"Zeitlicher Split nach Match bei {split:%Y-%m-%d %H:%M} UTC",
        )

    def feature_vector(self, view, entry):
        df = self._view(view).get_feature_vector(entry, return_type="pandas", allow_missing=True)
        if df is None or df.empty or df.isna().all(axis=None):
            return None
        return df.iloc[0].to_dict()

    def save_model(self, name, model_dir, metrics, description, input_example, view):
        mr = self.project.get_model_registry()
        model = mr.sklearn.create_model(
            name=name,
            metrics={k: v for k, v in metrics.items() if isinstance(v, float)},
            description=description,
            input_example=input_example,
            feature_view=self._view(view),
        )
        model.save(str(model_dir))
        return model.version

    def latest_model(self, name):
        mr = self.project.get_model_registry()
        meta = max(mr.get_models(name), key=lambda m: m.version)
        return ModelHandle(name, meta.version, Path(meta.download()), dict(meta.training_metrics or {}))


# --------------------------------------------------------------------------- lokal
class LocalStore:
    kind = "local"

    def __init__(self, root: Path = ROOT) -> None:
        self.fs_dir = root / "data" / "featurestore"
        self.registry = root / "models" / "registry"

    def _path(self, spec: FeatureGroupSpec) -> Path:
        return self.fs_dir / f"{spec.name}_v{spec.version}.parquet"

    def insert(self, spec, df):
        """Upsert per Primary Key wie ``fg.insert`` in Hopsworks."""
        self.fs_dir.mkdir(parents=True, exist_ok=True)
        path = self._path(spec)
        if path.exists():
            df = pd.concat([pd.read_parquet(path), df], ignore_index=True)
        df = df.drop_duplicates(spec.primary_key, keep="last").sort_values(spec.event_time)
        df.to_parquet(path, index=False)
        (self.fs_dir / f"{spec.name}_v{spec.version}.json").write_text(json.dumps(
            {"description": spec.description, "primary_key": spec.primary_key,
             "event_time": spec.event_time, "online": spec.online}, indent=2) + "\n")

    def read_group(self, spec, columns):
        path = self._path(spec)
        if not path.exists():
            raise FileNotFoundError(f"Feature Group {spec.name} v{spec.version} fehlt – zuerst die Feature Pipeline laufen lassen.")
        return pd.read_parquet(path, columns=columns or None)

    def train_test_split(self, view, split, start, end):
        t = view.group.event_time
        df = self.read_group(view.group, list(dict.fromkeys(view.columns + view.labels + [t])))
        train = df[(df[t] >= start) & (df[t] < split)]
        test = df[(df[t] >= split) & (df[t] < end)]
        return train[view.columns], test[view.columns], train[view.labels], test[view.labels]

    def feature_vector(self, view, entry):
        df = self.read_group(view.group, [])
        for key, value in entry.items():
            df = df[df[key] == value]
        if df.empty:
            return None
        return df.sort_values(view.group.event_time).iloc[-1][view.columns].to_dict()

    def save_model(self, name, model_dir, metrics, description, input_example, view):
        base = self.registry / name
        versions = [int(p.name[1:]) for p in base.glob("v*") if p.name[1:].isdigit()]
        version = max(versions, default=0) + 1
        target = base / f"v{version}"
        shutil.copytree(model_dir, target)
        (target / "registry.json").write_text(json.dumps(
            {"name": name, "version": version, "description": description, "metrics": metrics,
             "feature_view": {"name": view.name, "version": view.version},
             "input_example": input_example.to_dict(orient="records")}, indent=2, default=str) + "\n")
        return version

    def latest_model(self, name):
        base = self.registry / name
        versions = sorted(int(p.name[1:]) for p in base.glob("v*") if p.name[1:].isdigit())
        if not versions:
            raise FileNotFoundError(f"Kein Modell {name} in {base} – zuerst die Training Pipeline laufen lassen.")
        path = base / f"v{versions[-1]}"
        metrics = json.loads((path / "registry.json").read_text())["metrics"]
        return ModelHandle(name, versions[-1], path, metrics)


def store_from_env() -> Store:
    kind = os.environ.get("FEATURE_STORE", "hopsworks")
    if kind == "hopsworks":
        return HopsworksStore()
    if kind == "local":
        return LocalStore()
    raise ValueError(f"Unbekannter FEATURE_STORE: {kind} (hopsworks | local)")
