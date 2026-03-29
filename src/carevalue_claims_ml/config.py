from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ProjectSettings:
    database_url: str
    contract_id: str
    feature_window_months: int
    label_horizon_months: int
    random_seed: int
    model_families: list[str]
    model_params: dict[str, dict[str, float | int | str]]


def load_settings(path: Path) -> ProjectSettings:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return ProjectSettings(
        database_url=str(raw["database_url"]),
        contract_id=str(raw.get("contract_id", "DEMO")),
        feature_window_months=int(raw.get("feature_window_months", 6)),
        label_horizon_months=int(raw.get("label_horizon_months", 3)),
        random_seed=int(raw.get("random_seed", 7)),
        model_families=list(raw.get("model_families", ["risk_high_cost", "cost_forecast", "risk_advanced"])),
        model_params=dict(raw.get("model_params", {})),
    )
