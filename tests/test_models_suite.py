from __future__ import annotations

from pathlib import Path

import pandas as pd

from carevalue_claims_ml.models import train_model_suite


def _toy_features() -> pd.DataFrame:
    rows = []
    for member in range(1, 41):
        for month in range(1, 7):
            rows.append(
                {
                    "member_id": f"M{member:06d}",
                    "month": f"2023-{month:02d}-01",
                    "age": 40 + (member % 20),
                    "is_female": member % 2,
                    "allowed_last_window": float(member * month),
                    "ip_last_window": float((member + month) % 3),
                    "ed_last_window": float((member + month) % 4),
                    "allowed_last_month": float(member * max(1, month - 1)),
                    "care_management_touch": int((member + month) % 5 == 0),
                }
            )
    return pd.DataFrame(rows)


def _toy_labels(features: pd.DataFrame) -> pd.DataFrame:
    df = features[["member_id", "month"]].copy()
    df["future_allowed_sum"] = (
        features["allowed_last_window"] * 0.7
        + features["ip_last_window"] * 300.0
        + features["ed_last_window"] * 120.0
    )
    threshold = df["future_allowed_sum"].quantile(0.7)
    df["label_high_cost"] = (df["future_allowed_sum"] >= threshold).astype(int)
    return df


def test_train_model_suite_writes_artifacts(tmp_path: Path):
    features = _toy_features()
    labels = _toy_labels(features)
    results = train_model_suite(features, labels, tmp_path, suite="advanced")
    assert "risk_high_cost" in results
    assert "cost_forecast" in results
    assert "risk_advanced" in results
    assert "risk_temporal" in results
    assert "cost_interval" in results
    assert "uplift_proxy" in results
    for result in results.values():
        assert result.artifact_path.exists()
        metadata = tmp_path / f"{result.model_name}_{result.run_id}.metadata.json"
        assert metadata.exists()


def test_train_model_suite_family_selection(tmp_path: Path):
    features = _toy_features()
    labels = _toy_labels(features)
    selected = ["risk_high_cost", "anomaly_cost_spike", "contract_sensitive_ranker"]
    results = train_model_suite(features, labels, tmp_path, suite="maximal", model_families=selected)
    assert set(results.keys()) == set(selected)


def test_train_cost_outcome_use_case_families(tmp_path: Path):
    features = _toy_features()
    labels = _toy_labels(features)
    selected = [
        "vbc_cost_optimizer",
        "outcome_improvement_optimizer",
        "claims_behavior_predictor",
        "provider_advisory_ranker",
    ]
    results = train_model_suite(features, labels, tmp_path, suite="maximal", model_families=selected)
    assert set(results.keys()) == set(selected)
