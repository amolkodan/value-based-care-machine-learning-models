from __future__ import annotations

import pandas as pd

from carevalue_claims_ml.models import simulate_policy_allocation


def simulate_policy(scores: pd.DataFrame, budget: int) -> dict[str, float]:
    metrics = simulate_policy_allocation(scores, budget=budget)
    metrics["budget"] = float(budget)
    metrics["budget_utilization"] = (
        metrics["members_targeted"] / float(budget) if budget > 0 else 0.0
    )
    return metrics


def enforce_policy_safety(
    recommendations: pd.DataFrame,
    max_outreach: int,
    abstain_threshold: float = 0.15,
) -> pd.DataFrame:
    out = recommendations.copy().sort_values("score", ascending=False)
    out["policy_block_reason"] = ""
    if "score" in out.columns:
        out.loc[out["score"].astype(float) < abstain_threshold, "recommended_action"] = "abstain_low_confidence"
        out.loc[out["recommended_action"] == "abstain_low_confidence", "policy_block_reason"] = (
            "low_confidence"
        )
    if len(out) > max_outreach:
        out.loc[out.index[max_outreach:], "recommended_action"] = "abstain_budget_cap"
        out.loc[out.index[max_outreach:], "policy_block_reason"] = "budget_cap"
    return out


def evaluate_agent_strategy(
    agent_output: pd.DataFrame,
    baseline_output: pd.DataFrame,
    budget: int,
) -> dict[str, float]:
    agent_top = agent_output.sort_values("score", ascending=False).head(budget)
    base_top = baseline_output.sort_values("score", ascending=False).head(budget)
    avoided_cost_agent = float((agent_top.get("expected_contract_delta", 0)).sum())
    avoided_cost_base = float((base_top.get("expected_contract_delta", 0)).sum())
    precision_agent = float((agent_top["recommended_action"] != "abstain_low_confidence").mean())
    precision_base = float((base_top["recommended_action"] != "abstain_low_confidence").mean())
    fairness_delta = float(
        agent_top.get("dual_status_proxy", pd.Series([0.0])).mean()
        - base_top.get("dual_status_proxy", pd.Series([0.0])).mean()
    )
    calibration_shift = float(agent_top["score"].mean() - base_top["score"].mean())
    cost_delta = float(agent_top.get("expected_contract_delta", pd.Series([0.0])).sum() - base_top.get("expected_contract_delta", pd.Series([0.0])).sum())
    outcome_delta = float(agent_top.get("uplift_score", agent_top["score"]).mean() - base_top.get("uplift_score", base_top["score"]).mean())
    return {
        "avoided_cost_delta": avoided_cost_agent - avoided_cost_base,
        "intervention_precision_delta": precision_agent - precision_base,
        "fairness_delta": fairness_delta,
        "budget_adherence": float(len(agent_top) / budget) if budget > 0 else 0.0,
        "calibration_shift": calibration_shift,
        "cost_reduction_delta": cost_delta,
        "outcome_improvement_delta": outcome_delta,
        "cost_outcome_blended_delta": float(cost_delta * 0.6 + outcome_delta * 0.4),
    }
