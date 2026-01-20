from __future__ import annotations

from datetime import date
from pathlib import Path

import typer
from rich import print
from sqlalchemy import text

from carevalue_claims_ml.attribution import attribute_members_to_providers
from carevalue_claims_ml.benchmarks import calculate_pmpm, score_shared_savings
from carevalue_claims_ml.config import load_settings
from carevalue_claims_ml.data_generation import SyntheticDataConfig, generate_synthetic_dataset
from carevalue_claims_ml.db import create_db_engine, get_database_url
from carevalue_claims_ml.etl import build_member_months, initialize_schema
from carevalue_claims_ml.features import build_high_cost_label, build_member_month_features
from carevalue_claims_ml.loader import load_generated_folder
from carevalue_claims_ml.models import load_model, score_model, train_cost_model, train_risk_model
from carevalue_claims_ml.reporting import write_summary_report

app = typer.Typer(add_completion=False)
db_app = typer.Typer(add_completion=False)
data_app = typer.Typer(add_completion=False)
features_app = typer.Typer(add_completion=False)
models_app = typer.Typer(add_completion=False)
report_app = typer.Typer(add_completion=False)

app.add_typer(db_app, name="db")
app.add_typer(data_app, name="data")
app.add_typer(features_app, name="features")
app.add_typer(models_app, name="models")
app.add_typer(report_app, name="report")


@app.callback()
def main():
    pass


@db_app.command("init")
def db_init(schema_path: Path = Path("sql/schema/postgres.sql"), database_url: str | None = None):
    engine = create_db_engine(get_database_url(database_url))
    initialize_schema(engine, schema_path)
    print("Schema initialized")


@data_app.command("generate")
def data_generate(
    output: Path = Path("data/generated"),
    member_count: int = 600,
    months: int = 24,
    start_month: str = "2023-01-01",
    random_seed: int = 7,
):
    cfg = SyntheticDataConfig(
        member_count=int(member_count),
        months=int(months),
        start_month=date.fromisoformat(start_month),
        random_seed=int(random_seed),
    )
    generate_synthetic_dataset(output, cfg)
    print(f"Generated synthetic dataset in {output}")


@data_app.command("load")
def data_load(input_dir: Path = Path("data/generated"), database_url: str | None = None):
    engine = create_db_engine(get_database_url(database_url))
    counts = load_generated_folder(engine, input_dir)
    print(counts)


@features_app.command("build")
def features_build(
    settings_path: Path = Path("config/settings.yaml"),
    member_months_sql_path: Path = Path("src/carevalue_claims_ml/sql_build_member_months.sql"),
    database_url: str | None = None,
):
    settings = load_settings(settings_path)
    engine = create_db_engine(get_database_url(database_url or settings.database_url))
    build_member_months(engine, member_months_sql_path)
    attributed = attribute_members_to_providers(engine, settings.contract_id)
    print(f"Member-months built, attribution rows upserted: {attributed}")


@models_app.command("train")
def models_train(
    settings_path: Path = Path("config/settings.yaml"),
    output_dir: Path = Path("models"),
    database_url: str | None = None,
):
    settings = load_settings(settings_path)
    engine = create_db_engine(get_database_url(database_url or settings.database_url))
    features_df = build_member_month_features(engine, settings.feature_window_months)
    labels_df = build_high_cost_label(engine, settings.label_horizon_months)
    risk_result = train_risk_model(features_df, labels_df, output_dir)
    cost_result = train_cost_model(features_df, labels_df, output_dir)
    print({"risk": risk_result.metrics, "cost": cost_result.metrics})
    print({"risk_artifact": str(risk_result.artifact_path), "cost_artifact": str(cost_result.artifact_path)})


@models_app.command("score")
def models_score(
    artifact: Path,
    model_name: str,
    run_id: str,
    settings_path: Path = Path("config/settings.yaml"),
    database_url: str | None = None,
):
    settings = load_settings(settings_path)
    engine = create_db_engine(get_database_url(database_url or settings.database_url))
    features_df = build_member_month_features(engine, settings.feature_window_months)
    pipeline = load_model(artifact)
    scored = score_model(pipeline, features_df)

    with engine.begin() as connection:
        connection.execute(
            text("delete from vbc.model_scores where model_name = :m and run_id = :r"),
            {"m": model_name, "r": run_id},
        )
        rows = [
            {
                "model_name": model_name,
                "run_id": run_id,
                "member_id": r.member_id,
                "month": r.month,
                "score": float(r.score),
            }
            for r in scored.itertuples(index=False)
        ]
        connection.execute(
            text(
                "insert into vbc.model_scores (model_name, run_id, member_id, month, score) "
                "values (:model_name, :run_id, :member_id, :month, :score)"
            ),
            rows,
        )
    print(f"Scored {len(scored)} rows into vbc.model_scores")


@report_app.command("summary")
def report_summary(
    settings_path: Path = Path("config/settings.yaml"),
    output_dir: Path = Path("reports"),
    database_url: str | None = None,
):
    settings = load_settings(settings_path)
    engine = create_db_engine(get_database_url(database_url or settings.database_url))
    pmpm_df = calculate_pmpm(engine)
    savings_df = score_shared_savings(engine, settings.contract_id)
    out_path = write_summary_report(output_dir, pmpm_df, savings_df)
    print(f"Wrote {out_path}")
