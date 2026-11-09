"""Legt das Hopsworks-Projekt aus .env an (falls nötig) und prüft Featurestore + Model Registry.

Idempotent – kann beliebig oft ausgeführt werden:
    uv run python tools/setup_hopsworks.py
    docker run --rm --env-file .env cs2-money-round tools/setup_hopsworks.py
"""

import os

import hopsworks
from dotenv import load_dotenv

load_dotenv(".env")

API_KEY = os.environ["HOPSWORKS_API_KEY"]
PROJECT = os.environ["HOPSWORKS_PROJECT"]
DESCRIPTION = "CAS MLOps Projektarbeit: CS2 Full-Buy-Vorhersage Gegnerteam (FTI)"


def main() -> None:
    try:
        project = hopsworks.login(api_key_value=API_KEY, project=PROJECT)
        print(f"Projekt existiert bereits: {project.name}")
    except Exception:
        # Projektnamen sind auf Hopsworks Serverless global eindeutig → bei 409 anderen Namen wählen
        project = hopsworks.create_project(PROJECT, description=DESCRIPTION)
        print(f"Projekt erstellt: {project.name} (id {project.id})")

    fs = project.get_feature_store()
    mr = project.get_model_registry()
    print(f"Featurestore: {fs.name}")
    print(f"Model Registry: {mr.project_name}")


if __name__ == "__main__":
    main()
