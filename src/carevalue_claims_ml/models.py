from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class TrainingResult:
    run_id: str
    model_name: str
    metrics: dict[str, float]
    artifact_path: Path


def generate_run_id() -> str:
    import datetime

    return datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


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

    return TrainingResult(run_id=run_id, model_name="cost_forecast", metrics=metrics, artifact_path=artifact)


def load_model(artifact_path: Path):
    return joblib.load(artifact_path)


def score_model(pipeline, features: pd.DataFrame) -> pd.DataFrame:
    df = features.copy()
    X = df.drop(columns=["member_id", "month"], errors="ignore")
    if hasattr(pipeline, "predict_proba"):
        score = pipeline.predict_proba(X)[:, 1]
    else:
        score = pipeline.predict(X)
    out = df[["member_id", "month"]].copy()
    out["score"] = score
    return out
