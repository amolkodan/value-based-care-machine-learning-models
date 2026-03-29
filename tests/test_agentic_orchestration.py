from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from carevalue_claims_ml.agent_contracts import AgentHandoffContract, validate_handoff_contract
from carevalue_claims_ml.agent_orchestrator import (
    AgentGuardrails,
    build_handoff_contract,
    generate_audit_log,
    run_agentic_pipeline,
    write_contract,
)
from carevalue_claims_ml.memory_store import SharedContextStore
from carevalue_claims_ml.policy_simulation import evaluate_agent_strategy, enforce_policy_safety


def _scores() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"member_id": "M1", "month": "2023-01-01", "score": 0.9, "dual_status_proxy": 1, "eligible_for_outreach": 1},
            {"member_id": "M2", "month": "2023-01-01", "score": 0.52, "dual_status_proxy": 0, "eligible_for_outreach": 1},
            {"member_id": "M3", "month": "2023-01-01", "score": 0.1, "dual_status_proxy": 0, "eligible_for_outreach": 0},
        ]
    )


def test_pipeline_guardrails_and_audit(tmp_path: Path):
    context = SharedContextStore()
    recs = run_agentic_pipeline(_scores(), guardrails=AgentGuardrails(max_outreach=2), context_store=context)
    assert "recommended_action" in recs.columns
    assert recs["autonomous_action_blocked"].all()
    audit_path = tmp_path / "audit.csv"
    audit = generate_audit_log(recs, {"alerts": "none"}, audit_path)
    assert audit_path.exists()
    assert "why_not" in audit.columns
    assert context.get("latest_quality") is not None


def test_contract_build_validate_write(tmp_path: Path):
    contract = build_handoff_contract("test_stage", _scores())
    validate_handoff_contract(contract)
    out = tmp_path / "contract.json"
    write_contract(contract, out)
    payload = json.loads(out.read_text(encoding="utf-8"))
    loaded = AgentHandoffContract(stage=payload["stage"], payload=payload["payload"], version=payload["version"])
    validate_handoff_contract(loaded)


def test_policy_eval_and_safety():
    recs = run_agentic_pipeline(_scores())
    safe = enforce_policy_safety(recs, max_outreach=2, abstain_threshold=0.2)
    assert "policy_block_reason" in safe.columns
    metrics = evaluate_agent_strategy(safe, safe.copy(), budget=2)
    assert "avoided_cost_delta" in metrics
