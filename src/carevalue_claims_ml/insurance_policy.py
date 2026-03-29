from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class InsurancePolicyConfig:
    scenario: str = "base"
    shared_savings_rate: float = 0.5
    downside_cap: float = 50000.0
    risk_corridor: float = 0.08
    quality_floor: float = 0.6
    outreach_budget: int = 100


def apply_contract_constraints(recommendations: pd.DataFrame, cfg: InsurancePolicyConfig) -> pd.DataFrame:
    out = recommendations.copy()
    if "expected_contract_delta" not in out.columns:
        out["expected_contract_delta"] = out.get("score", 0).astype(float) * 50.0
    out["policy_status"] = "approved"
    out["shared_savings_projection"] = out["expected_contract_delta"].astype(float) * cfg.shared_savings_rate
    out.loc[out["shared_savings_projection"] < 0, "policy_status"] = "review_downside"
    out.loc[out.index[cfg.outreach_budget :], "policy_status"] = "deferred_budget"
    out["risk_corridor_breach"] = (
        out["expected_contract_delta"].astype(float).abs() > cfg.downside_cap * cfg.risk_corridor
    ).astype(int)
    return out


def run_policy_scenarios(recommendations: pd.DataFrame) -> dict[str, dict[str, float]]:
    scenarios = {
        "optimistic": InsurancePolicyConfig(scenario="optimistic", shared_savings_rate=0.6),
        "base": InsurancePolicyConfig(scenario="base", shared_savings_rate=0.5),
        "stress": InsurancePolicyConfig(scenario="stress", shared_savings_rate=0.35, downside_cap=40000.0),
    }
    results: dict[str, dict[str, float]] = {}
    for name, cfg in scenarios.items():
        constrained = apply_contract_constraints(recommendations, cfg)
        cost_component = float(constrained.get("expected_contract_delta", pd.Series([0.0])).sum())
        outcome_component = float(constrained.get("uplift_score", constrained.get("score", pd.Series([0.0]))).sum())
        results[name] = {
            "approved": float((constrained["policy_status"] == "approved").sum()),
            "deferred_budget": float((constrained["policy_status"] == "deferred_budget").sum()),
            "projected_shared_savings": float(constrained["shared_savings_projection"].sum()),
            "cost_reduction_score": cost_component,
            "outcome_improvement_score": outcome_component,
            "cost_outcome_tradeoff_index": float(cost_component * 0.6 + outcome_component * 0.4),
        }
    return results
