from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class TrainingResult:
    run_id: str
    model_name: str
    metrics: dict[str, float]
    artifact_path: Path


TrainerFn = Callable[[pd.DataFrame, pd.DataFrame, Path], TrainingResult]


def generate_run_id() -> str:
    import datetime

    return datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _feature_hash(columns: list[str]) -> str:
    # Stable, readable hash-like fingerprint from ordered feature names.
    return str(abs(hash(tuple(columns))))


def _write_metadata(
    output_dir: Path,
    run_id: str,
    model_name: str,
    metrics: dict[str, float],
    feature_columns: list[str],
    task: str,
    cohort: str = "all_members",
    version: str = "v1",
) -> Path:
    meta = {
        "run_id": run_id,
        "model_id": f"{model_name}:{run_id}",
        "model_name": model_name,
        "task": task,
        "cohort": cohort,
        "version": version,
        "featureset_hash": _feature_hash(feature_columns),
        "feature_columns": feature_columns,
        "metrics": metrics,
    }
    metadata_path = output_dir / f"{model_name}_{run_id}.metadata.json"
    metadata_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return metadata_path


def train_risk_model(features: pd.DataFrame, label_df: pd.DataFrame, output_dir: Path) -> TrainingResult:
    df = features.merge(label_df, on=["member_id", "month"], how="inner").dropna()
    y = df["label_high_cost"].astype(int).values
    X = df.drop(columns=["member_id", "month", "label_high_cost", "future_allowed_sum"], errors="ignore")

    numeric_columns = list(X.columns)
    pre = ColumnTransformer([("num", StandardScaler(), numeric_columns)], remainder="drop")

    base_model = LogisticRegression(max_iter=2000, class_weight="balanced")
    calibrated = CalibratedClassifierCV(base_model, method="isotonic", cv=3)
    pipeline = Pipeline([("pre", pre), ("model", calibrated)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=7, stratify=y
    )

    pipeline.fit(X_train, y_train)
    prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, prob)),
        "avg_precision": float(average_precision_score(y_test, prob)),
    }

    run_id = generate_run_id()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = output_dir / f"risk_model_{run_id}.joblib"
    joblib.dump(pipeline, artifact)

    _write_metadata(output_dir, run_id, "risk_high_cost", metrics, numeric_columns, task="classification")
    return TrainingResult(run_id=run_id, model_name="risk_high_cost", metrics=metrics, artifact_path=artifact)


def train_cost_model(features: pd.DataFrame, label_df: pd.DataFrame, output_dir: Path) -> TrainingResult:
    df = features.merge(label_df, on=["member_id", "month"], how="inner").dropna()
    y = df["future_allowed_sum"].astype(float).values
    X = df.drop(columns=["member_id", "month", "label_high_cost", "future_allowed_sum"], errors="ignore")

    numeric_columns = list(X.columns)
    pre = ColumnTransformer([("num", StandardScaler(), numeric_columns)], remainder="drop")

    model = GradientBoostingRegressor(random_state=7)
    pipeline = Pipeline([("pre", pre), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=7)

    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)

    metrics = {
        "mae": float(mean_absolute_error(y_test, pred)),
        "p90_abs_error": float(np.quantile(np.abs(y_test - pred), 0.9)),
    }

    run_id = generate_run_id()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = output_dir / f"cost_model_{run_id}.joblib"
    joblib.dump(pipeline, artifact)

    _write_metadata(output_dir, run_id, "cost_forecast", metrics, numeric_columns, task="regression")
    return TrainingResult(run_id=run_id, model_name="cost_forecast", metrics=metrics, artifact_path=artifact)


def train_advanced_risk_model(
    features: pd.DataFrame, label_df: pd.DataFrame, output_dir: Path
) -> TrainingResult:
    df = features.merge(label_df, on=["member_id", "month"], how="inner").dropna()
    y = df["label_high_cost"].astype(int).values
    X = df.drop(columns=["member_id", "month", "label_high_cost", "future_allowed_sum"], errors="ignore")
    numeric_columns = list(X.columns)
    pre = ColumnTransformer([("num", StandardScaler(), numeric_columns)], remainder="drop")

    gbm = GradientBoostingClassifier(random_state=7)
    rf = RandomForestClassifier(n_estimators=200, random_state=7, class_weight="balanced")
    gbm_pipe = Pipeline([("pre", pre), ("model", gbm)])
    rf_pipe = Pipeline([("pre", pre), ("model", rf)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=7, stratify=y
    )
    gbm_pipe.fit(X_train, y_train)
    rf_pipe.fit(X_train, y_train)
    prob = 0.5 * gbm_pipe.predict_proba(X_test)[:, 1] + 0.5 * rf_pipe.predict_proba(X_test)[:, 1]
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, prob)),
        "avg_precision": float(average_precision_score(y_test, prob)),
    }

    run_id = generate_run_id()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = output_dir / f"risk_advanced_{run_id}.joblib"
    joblib.dump({"gbm": gbm_pipe, "rf": rf_pipe}, artifact)
    _write_metadata(output_dir, run_id, "risk_advanced", metrics, numeric_columns, task="classification")
    return TrainingResult(run_id=run_id, model_name="risk_advanced", metrics=metrics, artifact_path=artifact)


def train_temporal_risk_model(
    features: pd.DataFrame, label_df: pd.DataFrame, output_dir: Path
) -> TrainingResult:
    df = features.merge(label_df, on=["member_id", "month"], how="inner").dropna()
    df["month"] = pd.to_datetime(df["month"])
    df = df.sort_values("month")
    y = df["label_high_cost"].astype(int).values
    X = df.drop(columns=["member_id", "month", "label_high_cost", "future_allowed_sum"], errors="ignore")
    numeric_columns = list(X.columns)
    pre = ColumnTransformer([("num", StandardScaler(), numeric_columns)], remainder="drop")
    model = Pipeline([("pre", pre), ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))])

    folds = TimeSeriesSplit(n_splits=3)
    fold_scores: list[float] = []
    for train_idx, test_idx in folds.split(X):
        model.fit(X.iloc[train_idx], y[train_idx])
        prob = model.predict_proba(X.iloc[test_idx])[:, 1]
        if len(np.unique(y[test_idx])) > 1:
            fold_scores.append(float(roc_auc_score(y[test_idx], prob)))
    model.fit(X, y)

    run_id = generate_run_id()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = output_dir / f"risk_temporal_{run_id}.joblib"
    joblib.dump(model, artifact)
    metrics = {
        "temporal_cv_roc_auc_mean": float(np.mean(fold_scores)) if fold_scores else float("nan"),
        "temporal_cv_roc_auc_std": float(np.std(fold_scores)) if fold_scores else float("nan"),
    }
    _write_metadata(output_dir, run_id, "risk_temporal", metrics, numeric_columns, task="classification")
    return TrainingResult(run_id=run_id, model_name="risk_temporal", metrics=metrics, artifact_path=artifact)


def train_cost_interval_model(
    features: pd.DataFrame, label_df: pd.DataFrame, output_dir: Path
) -> TrainingResult:
    df = features.merge(label_df, on=["member_id", "month"], how="inner").dropna()
    y = df["future_allowed_sum"].astype(float).values
    X = df.drop(columns=["member_id", "month", "label_high_cost", "future_allowed_sum"], errors="ignore")
    numeric_columns = list(X.columns)
    pre = ColumnTransformer([("num", StandardScaler(), numeric_columns)], remainder="drop")

    q10 = Pipeline(
        [("pre", pre), ("model", GradientBoostingRegressor(loss="quantile", alpha=0.1, random_state=7))]
    )
    q50 = Pipeline(
        [("pre", pre), ("model", GradientBoostingRegressor(loss="quantile", alpha=0.5, random_state=7))]
    )
    q90 = Pipeline(
        [("pre", pre), ("model", GradientBoostingRegressor(loss="quantile", alpha=0.9, random_state=7))]
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=7)
    for pipe in [q10, q50, q90]:
        pipe.fit(X_train, y_train)
    p50 = q50.predict(X_test)
    interval_width = np.mean(q90.predict(X_test) - q10.predict(X_test))
    metrics = {
        "mae_p50": float(mean_absolute_error(y_test, p50)),
        "avg_interval_width_q10_q90": float(interval_width),
    }

    run_id = generate_run_id()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = output_dir / f"cost_interval_{run_id}.joblib"
    joblib.dump({"q10": q10, "q50": q50, "q90": q90}, artifact)
    _write_metadata(output_dir, run_id, "cost_interval", metrics, numeric_columns, task="regression")
    return TrainingResult(run_id=run_id, model_name="cost_interval", metrics=metrics, artifact_path=artifact)


def train_uplift_proxy_model(
    features: pd.DataFrame, label_df: pd.DataFrame, output_dir: Path, treatment_col: str = "care_management_touch"
) -> TrainingResult:
    df = features.merge(label_df, on=["member_id", "month"], how="inner")
    if treatment_col not in df.columns:
        df[treatment_col] = (df["allowed_last_window"] > df["allowed_last_window"].median()).astype(int)
    df = df.dropna()

    y = df["label_high_cost"].astype(int).values
    treated = df[treatment_col].astype(int).values
    X = df.drop(
        columns=["member_id", "month", "label_high_cost", "future_allowed_sum", treatment_col],
        errors="ignore",
    )
    numeric_columns = list(X.columns)
    pre = ColumnTransformer([("num", StandardScaler(), numeric_columns)], remainder="drop")
    model_treated = Pipeline([("pre", pre), ("model", LogisticRegression(max_iter=2000))])
    model_control = Pipeline([("pre", pre), ("model", LogisticRegression(max_iter=2000))])

    tr_idx = treated == 1
    ct_idx = treated == 0
    model_treated.fit(X.loc[tr_idx], y[tr_idx] if np.any(tr_idx) else y)
    model_control.fit(X.loc[ct_idx], y[ct_idx] if np.any(ct_idx) else y)

    uplift = model_treated.predict_proba(X)[:, 1] - model_control.predict_proba(X)[:, 1]
    top_decile = np.quantile(uplift, 0.9)
    top_mask = uplift >= top_decile
    uplift_signal = float(np.mean(y[top_mask]) - np.mean(y[~top_mask])) if np.any(~top_mask) else 0.0
    metrics = {"uplift_top_decile_delta_label_rate": uplift_signal}

    run_id = generate_run_id()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = output_dir / f"uplift_proxy_{run_id}.joblib"
    joblib.dump({"treated": model_treated, "control": model_control}, artifact)
    _write_metadata(output_dir, run_id, "uplift_proxy", metrics, numeric_columns, task="uplift")
    return TrainingResult(run_id=run_id, model_name="uplift_proxy", metrics=metrics, artifact_path=artifact)


def train_anomaly_cost_spike_model(
    features: pd.DataFrame, label_df: pd.DataFrame, output_dir: Path
) -> TrainingResult:
    df = features.merge(label_df, on=["member_id", "month"], how="inner").dropna()
    X = df.drop(columns=["member_id", "month", "label_high_cost", "future_allowed_sum"], errors="ignore")
    numeric_columns = list(X.columns)
    pre = ColumnTransformer([("num", StandardScaler(), numeric_columns)], remainder="drop")
    model = Pipeline([("pre", pre), ("model", IsolationForest(random_state=7, contamination=0.1))])
    model.fit(X)
    anomaly_score = -model.decision_function(X)
    metrics = {"avg_anomaly_score": float(np.mean(anomaly_score))}
    run_id = generate_run_id()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = output_dir / f"anomaly_cost_spike_{run_id}.joblib"
    joblib.dump(model, artifact)
    _write_metadata(output_dir, run_id, "anomaly_cost_spike", metrics, numeric_columns, task="anomaly")
    return TrainingResult(run_id=run_id, model_name="anomaly_cost_spike", metrics=metrics, artifact_path=artifact)


def train_risk_trajectory_segment_model(
    features: pd.DataFrame, label_df: pd.DataFrame, output_dir: Path
) -> TrainingResult:
    df = features.merge(label_df, on=["member_id", "month"], how="inner").dropna()
    X = df.drop(columns=["member_id", "month", "label_high_cost", "future_allowed_sum"], errors="ignore")
    numeric_columns = list(X.columns)
    pre = ColumnTransformer([("num", StandardScaler(), numeric_columns)], remainder="drop")
    model = Pipeline([("pre", pre), ("model", KMeans(n_clusters=4, random_state=7, n_init=10))])
    model.fit(X)
    labels = model.named_steps["model"].labels_
    metrics = {"n_segments": float(len(np.unique(labels))), "largest_segment_share": float(np.max(np.bincount(labels)) / len(labels))}
    run_id = generate_run_id()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = output_dir / f"risk_trajectory_segment_{run_id}.joblib"
    joblib.dump(model, artifact)
    _write_metadata(output_dir, run_id, "risk_trajectory_segment", metrics, numeric_columns, task="segmentation")
    return TrainingResult(run_id=run_id, model_name="risk_trajectory_segment", metrics=metrics, artifact_path=artifact)


def train_uplift_stronger_model(
    features: pd.DataFrame, label_df: pd.DataFrame, output_dir: Path
) -> TrainingResult:
    df = features.merge(label_df, on=["member_id", "month"], how="inner").dropna()
    if "care_management_touch" not in df.columns:
        df["care_management_touch"] = (df["allowed_last_window"] > df["allowed_last_window"].median()).astype(int)
    y = df["label_high_cost"].astype(int).values
    treated = df["care_management_touch"].astype(int).values
    X = df.drop(columns=["member_id", "month", "label_high_cost", "future_allowed_sum", "care_management_touch"], errors="ignore")
    numeric_columns = list(X.columns)
    pre = ColumnTransformer([("num", StandardScaler(), numeric_columns)], remainder="drop")
    tr_model = Pipeline([("pre", pre), ("model", GradientBoostingClassifier(random_state=7))])
    ct_model = Pipeline([("pre", pre), ("model", GradientBoostingClassifier(random_state=17))])
    tr_mask = treated == 1
    ct_mask = treated == 0
    tr_model.fit(X.loc[tr_mask] if np.any(tr_mask) else X, y[tr_mask] if np.any(tr_mask) else y)
    ct_model.fit(X.loc[ct_mask] if np.any(ct_mask) else X, y[ct_mask] if np.any(ct_mask) else y)
    uplift = tr_model.predict_proba(X)[:, 1] - ct_model.predict_proba(X)[:, 1]
    metrics = {"uplift_mean": float(np.mean(uplift)), "uplift_p90": float(np.quantile(uplift, 0.9))}
    run_id = generate_run_id()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = output_dir / f"uplift_stronger_{run_id}.joblib"
    joblib.dump({"treated": tr_model, "control": ct_model}, artifact)
    _write_metadata(output_dir, run_id, "uplift_stronger", metrics, numeric_columns, task="uplift")
    return TrainingResult(run_id=run_id, model_name="uplift_stronger", metrics=metrics, artifact_path=artifact)


def train_contract_sensitive_ranker(
    features: pd.DataFrame, label_df: pd.DataFrame, output_dir: Path
) -> TrainingResult:
    df = features.merge(label_df, on=["member_id", "month"], how="inner").dropna()
    y = (df["future_allowed_sum"].astype(float) * 1.2 + 200 * df["label_high_cost"].astype(float)).values
    X = df.drop(columns=["member_id", "month", "label_high_cost", "future_allowed_sum"], errors="ignore")
    numeric_columns = list(X.columns)
    pre = ColumnTransformer([("num", StandardScaler(), numeric_columns)], remainder="drop")
    model = Pipeline([("pre", pre), ("model", GradientBoostingRegressor(random_state=21))])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=7)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {"ranker_mae": float(mean_absolute_error(y_test, pred))}
    run_id = generate_run_id()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = output_dir / f"contract_sensitive_ranker_{run_id}.joblib"
    joblib.dump(model, artifact)
    _write_metadata(output_dir, run_id, "contract_sensitive_ranker", metrics, numeric_columns, task="ranking")
    return TrainingResult(run_id=run_id, model_name="contract_sensitive_ranker", metrics=metrics, artifact_path=artifact)


def simulate_policy_allocation(
    member_scores: pd.DataFrame,
    budget: int,
    uplift_col: str = "uplift_score",
    cost_col: str = "expected_cost",
) -> dict[str, float]:
    if member_scores.empty or budget <= 0:
        return {"members_targeted": 0.0, "expected_savings": 0.0, "avg_target_uplift": 0.0}
    df = member_scores.copy()
    if uplift_col not in df.columns:
        df[uplift_col] = df["score"].astype(float)
    if cost_col not in df.columns:
        df[cost_col] = np.maximum(0.0, 300.0 + 200.0 * df["score"].astype(float))
    df["policy_value"] = df[uplift_col].astype(float) * df[cost_col].astype(float)
    selected = df.sort_values("policy_value", ascending=False).head(int(budget))
    return {
        "members_targeted": float(len(selected)),
        "expected_savings": float((selected["policy_value"] * 0.15).sum()),
        "avg_target_uplift": float(selected[uplift_col].mean() if len(selected) else 0.0),
    }


def train_model_suite(
    features: pd.DataFrame,
    label_df: pd.DataFrame,
    output_dir: Path,
    suite: str = "maximal",
    model_families: list[str] | None = None,
) -> dict[str, TrainingResult]:
    registry: dict[str, TrainerFn] = {
        "risk_high_cost": train_risk_model,
        "cost_forecast": train_cost_model,
        "risk_advanced": train_advanced_risk_model,
        "risk_temporal": train_temporal_risk_model,
        "cost_interval": train_cost_interval_model,
        "uplift_proxy": train_uplift_proxy_model,
        "anomaly_cost_spike": train_anomaly_cost_spike_model,
        "risk_trajectory_segment": train_risk_trajectory_segment_model,
        "uplift_stronger": train_uplift_stronger_model,
        "contract_sensitive_ranker": train_contract_sensitive_ranker,
    }
    if model_families:
        selected = model_families
    elif suite == "baseline":
        selected = ["risk_high_cost", "cost_forecast"]
    elif suite == "advanced":
        selected = ["risk_high_cost", "cost_forecast", "risk_advanced", "risk_temporal", "cost_interval", "uplift_proxy"]
    else:
        selected = list(registry.keys())
    results: dict[str, TrainingResult] = {}
    for family in selected:
        if family in registry:
            results[family] = registry[family](features, label_df, output_dir)
    return results


def load_model(artifact_path: Path):
    return joblib.load(artifact_path)


def score_model(pipeline, features: pd.DataFrame) -> pd.DataFrame:
    df = features.copy()
    X = df.drop(columns=["member_id", "month"], errors="ignore")
    if isinstance(pipeline, dict) and "treated" in pipeline and "control" in pipeline:
        score = pipeline["treated"].predict_proba(X)[:, 1] - pipeline["control"].predict_proba(X)[:, 1]
    elif isinstance(pipeline, dict) and "q50" in pipeline:
        score = pipeline["q50"].predict(X)
    elif isinstance(pipeline, dict) and "gbm" in pipeline and "rf" in pipeline:
        score = 0.5 * pipeline["gbm"].predict_proba(X)[:, 1] + 0.5 * pipeline["rf"].predict_proba(X)[:, 1]
    elif hasattr(pipeline, "decision_function"):
        score = pipeline.decision_function(X)
    elif hasattr(pipeline, "named_steps") and "model" in pipeline.named_steps and isinstance(pipeline.named_steps["model"], KMeans):
        clusters = pipeline.predict(X)
        score = clusters.astype(float)
    elif hasattr(pipeline, "predict_proba"):
        score = pipeline.predict_proba(X)[:, 1]
    else:
        score = pipeline.predict(X)
    out = df[["member_id", "month"]].copy()
    out["score"] = score
    if np.issubdtype(out["score"].dtype, np.number):
        out["score"] = out["score"].astype(float).replace([math.inf, -math.inf], np.nan).fillna(0.0)
    return out
