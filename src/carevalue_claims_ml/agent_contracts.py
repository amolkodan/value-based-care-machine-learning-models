from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class AgentHandoffContract:
    stage: str
    payload: list[dict]
    version: str
    run_id: str = "manual"
    contract_id: str = "DEMO"
    policy_version: str = "v1"
    upstream_stage_ids: list[str] | None = None
    quality_status: str = "ok"
    generated_at: str = ""


def validate_handoff_contract(contract: AgentHandoffContract) -> None:
    if not contract.stage:
        raise ValueError("Contract stage is required.")
    if contract.version != "v1":
        raise ValueError("Unsupported contract version.")
    if not isinstance(contract.payload, list):
        raise ValueError("Contract payload must be a list of records.")
    if not contract.generated_at:
        raise ValueError("Contract generated_at is required.")
    if contract.upstream_stage_ids is not None and not isinstance(contract.upstream_stage_ids, list):
        raise ValueError("Contract upstream_stage_ids must be a list.")


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
