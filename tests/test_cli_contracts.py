from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from carevalue_claims_ml.cli import app


runner = CliRunner()


def _write_predictions(path: Path) -> None:
    df = pd.DataFrame(
        [
            {"member_id": "M000001", "month": "2023-01-01", "label_high_cost": 1, "score": 0.85, "future_allowed_sum": 1800.0},
            {"member_id": "M000002", "month": "2023-01-01", "label_high_cost": 0, "score": 0.12, "future_allowed_sum": 220.0},
            {"member_id": "M000003", "month": "2023-01-01", "label_high_cost": 1, "score": 0.65, "future_allowed_sum": 1300.0},
        ]
    )
    df.to_csv(path, index=False)


def test_cli_leaderboard_and_policy(tmp_path: Path):
    pred_path = tmp_path / "pred.csv"
    _write_predictions(pred_path)

    res_lb = runner.invoke(
        app,
        [
            "models",
            "leaderboard",
            str(pred_path),
            "--model-name",
            "risk_cli",
            "--run-id",
            "run_cli",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert res_lb.exit_code == 0
    assert (tmp_path / "leaderboard.csv").exists()

    res_policy = runner.invoke(
        app,
        [
            "policy",
            "simulate",
            str(pred_path),
            "--budget",
            "2",
            "--output-path",
            str(tmp_path / "policy.json"),
        ],
    )
    assert res_policy.exit_code == 0
    payload = json.loads((tmp_path / "policy.json").read_text(encoding="utf-8"))
    assert payload["members_targeted"] == 2.0

    recs_path = tmp_path / "recs.csv"
    pd.read_csv(pred_path).assign(recommended_action="care_navigation_call").to_csv(recs_path, index=False)
    res_scenario = runner.invoke(app, ["policy", "scenario", str(recs_path), "--output-path", str(tmp_path / "scn.json")])
    assert res_scenario.exit_code == 0
    assert (tmp_path / "scn.json").exists()

    res_enforce = runner.invoke(
        app,
        ["policy", "enforce", str(recs_path), "--output-path", str(tmp_path / "enforced.csv"), "--outreach-budget", "1"],
    )
    assert res_enforce.exit_code == 0
    assert (tmp_path / "enforced.csv").exists()


def test_cli_agents_run_validate_and_eval(tmp_path: Path):
    pred_path = tmp_path / "pred.csv"
    _write_predictions(pred_path)
    rec_path = tmp_path / "recs.csv"
    audit_path = tmp_path / "audit.csv"
    contract_path = tmp_path / "contract.json"

    res_run = runner.invoke(
        app,
        [
            "agents",
            "run",
            str(pred_path),
            "--output-path",
            str(rec_path),
            "--audit-path",
            str(audit_path),
            "--contract-path",
            str(contract_path),
        ],
    )
    assert res_run.exit_code == 0
    assert rec_path.exists()
    assert audit_path.exists()
    assert contract_path.exists()

    res_validate = runner.invoke(app, ["agents", "validate-contract", str(contract_path)])
    assert res_validate.exit_code == 0

    base_path = tmp_path / "base.csv"
    pd.read_csv(rec_path).to_csv(base_path, index=False)
    res_eval = runner.invoke(
        app,
        [
            "agents",
            "evaluate",
            str(rec_path),
            str(base_path),
            "--budget",
            "2",
            "--output-path",
            str(tmp_path / "agent_eval.json"),
        ],
    )
    assert res_eval.exit_code == 0
    assert (tmp_path / "agent_eval.json").exists()
