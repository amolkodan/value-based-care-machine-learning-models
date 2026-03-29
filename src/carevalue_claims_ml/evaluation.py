from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score


@dataclass(frozen=True)
class EvaluationResult:
    metrics: dict[str, float]
    subgroup_metrics: list[dict[str, float | str]]


def evaluate_predictions(df: pd.DataFrame) -> EvaluationResult:
    work = df.copy()
    y_true = work["label_high_cost"].astype(int).values
    y_score = work["score"].astype(float).values
    metrics = {
        "roc_auc": float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else 0.5,
        "avg_precision": float(average_precision_score(y_true, y_score)),
    }
    if "future_allowed_sum" in work.columns:
        metrics["mae_cost_proxy"] = float(
            mean_absolute_error(work["future_allowed_sum"].astype(float), work["score"].astype(float))
        )
    subgroup_metrics = evaluate_fairness_slices(work)
    return EvaluationResult(metrics=metrics, subgroup_metrics=subgroup_metrics)


def evaluate_fairness_slices(df: pd.DataFrame) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    work = df.copy()
    if "age" in work.columns:
        work["age_band"] = pd.cut(
            work["age"],
            bins=[0, 35, 50, 65, 120],
            labels=["young", "mid", "senior", "elder"],
            include_lowest=True,
        )
    for col in ["is_female", "age_band", "dual_status_proxy"]:
        if col not in work.columns:
            continue
        for value, sub in work.groupby(col, dropna=False):
            if sub.empty:
                continue
            y = sub["label_high_cost"].astype(int).values
            s = sub["score"].astype(float).values
            auc = float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else 0.5
            rows.append({"slice_col": col, "slice_value": str(value), "roc_auc": auc, "n": float(len(sub))})
    return rows


def write_leaderboard_artifacts(
    output_dir: Path, model_name: str, run_id: str, result: EvaluationResult
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    leaderboard_path = output_dir / "leaderboard.csv"
    model_card_path = output_dir / f"model_card_{model_name}_{run_id}.json"

    row = {"model_name": model_name, "run_id": run_id, **result.metrics}
    lb = pd.DataFrame([row])
    if leaderboard_path.exists():
        prev = pd.read_csv(leaderboard_path)
        lb = pd.concat([prev, lb], ignore_index=True)
    lb.sort_values(["roc_auc", "avg_precision"], ascending=False, inplace=True, na_position="last")
    lb.to_csv(leaderboard_path, index=False)

    model_card = {
        "model_name": model_name,
        "run_id": run_id,
        "metrics": result.metrics,
        "subgroup_metrics": result.subgroup_metrics,
    }
    model_card_path.write_text(json.dumps(model_card, indent=2), encoding="utf-8")
    return leaderboard_path, model_card_path
