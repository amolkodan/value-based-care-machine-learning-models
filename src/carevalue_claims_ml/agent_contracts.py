from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentHandoffContract:
    stage: str
    payload: list[dict]
    version: str


def validate_handoff_contract(contract: AgentHandoffContract) -> None:
    if not contract.stage:
        raise ValueError("Contract stage is required.")
    if contract.version != "v1":
        raise ValueError("Unsupported contract version.")
    if not isinstance(contract.payload, list):
        raise ValueError("Contract payload must be a list of records.")
