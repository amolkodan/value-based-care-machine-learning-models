from __future__ import annotations

from pathlib import Path


def retrieve_reference_context(
    model_cards_path: Path, benchmark_summary: str, contract_rules: str
) -> str:
    base = model_cards_path.read_text(encoding="utf-8") if model_cards_path.exists() else ""
    return "\n".join([base[:2000], benchmark_summary[:1000], contract_rules[:1000]])


def build_prompt(member_payload: dict, retrieved_context: str) -> str:
    return (
        "You are a healthcare analytics assistant. Provide recommendation-only output.\n"
        f"Context:\n{retrieved_context}\n"
        f"Member payload:\n{member_payload}\n"
        "Return strict JSON with keys: recommended_action, rationale, confidence."
    )


def deterministic_postprocess(raw_output: str) -> dict[str, str | float]:
    # Deterministic safe fallback parser; no eval, no code execution.
    action = "abstain"
    rationale = "Fallback deterministic parser applied."
    confidence = 0.0
    if "care_navigation_call" in raw_output:
        action = "care_navigation_call"
        confidence = 0.7
    elif "pharmacy_followup" in raw_output:
        action = "pharmacy_followup"
        confidence = 0.6
    elif "digital_nudge" in raw_output:
        action = "digital_nudge"
        confidence = 0.55
    return {"recommended_action": action, "rationale": rationale, "confidence": confidence}
