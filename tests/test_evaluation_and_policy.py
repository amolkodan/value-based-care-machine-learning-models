from __future__ import annotations

from pathlib import Path

import pandas as pd

from carevalue_claims_ml.agent_orchestrator import run_agentic_pipeline
from carevalue_claims_ml.evaluation import evaluate_predictions, write_leaderboard_artifacts
from carevalue_claims_ml.policy_simulation import simulate_policy


def _prediction_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "member_id": "M000001",
                "month": "2023-01-01",
                "label_high_cost": 1,
                "future_allowed_sum": 2400.0,
                "score": 0.91,
                "age": 72,
                "is_female": 1,
                "dual_status_proxy": 1,
            },
            {
                "member_id": "M000002",
                "month": "2023-01-01",
                "label_high_cost": 0,
                "future_allowed_sum": 300.0,
                "score": 0.11,
                "age": 39,
                "is_female": 0,
                "dual_status_proxy": 0,
            },
            {
                "member_id": "M000003",
                "month": "2023-01-01",
                "label_high_cost": 1,
                "future_allowed_sum": 1500.0,
                "score": 0.68,
                "age": 57,
                "is_female": 1,
                "dual_status_proxy": 0,
            },
        ]
    )


def test_evaluation_and_leaderboard_outputs(tmp_path: Path):
    df = _prediction_frame()
    result = evaluate_predictions(df)
    assert "roc_auc" in result.metrics
    assert result.subgroup_metrics
    leaderboard_path, model_card_path = write_leaderboard_artifacts(
        tmp_path, model_name="risk_test", run_id="run1", result=result
    )
    assert leaderboard_path.exists()
    assert model_card_path.exists()


def test_policy_and_agent_pipeline():
    df = _prediction_frame()
    policy = simulate_policy(df, budget=2)
    assert policy["members_targeted"] == 2.0
    recs = run_agentic_pipeline(df[["member_id", "month", "score"]].copy())
    assert "recommended_action" in recs.columns
    assert "expected_contract_delta" in recs.columns
