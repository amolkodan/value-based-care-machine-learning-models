from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import typer
from rich import print
from sqlalchemy import text

from carevalue_claims_ml.attribution import attribute_members_to_providers
from carevalue_claims_ml.benchmarks import calculate_pmpm, score_shared_savings
from carevalue_claims_ml.config import load_settings
from carevalue_claims_ml.data_generation import SyntheticDataConfig, generate_synthetic_dataset
from carevalue_claims_ml.db import create_db_engine, get_database_url
from carevalue_claims_ml.etl import build_member_months, initialize_schema
from carevalue_claims_ml.evaluation import evaluate_predictions, write_leaderboard_artifacts
from carevalue_claims_ml.episodes import build_bundled_episodes, score_episode_risk
from carevalue_claims_ml.features import build_high_cost_label, build_member_month_features
from carevalue_claims_ml.journey_signals import (
    merge_medical_and_pharmacy_claims,
    monthly_utilization_features,
)
from carevalue_claims_ml.loader import load_generated_folder
from carevalue_claims_ml.models import (
    load_model,
    score_model,
    train_cost_model,
    train_model_suite,
    train_risk_model,
)
from carevalue_claims_ml.agent_orchestrator import (
    build_handoff_contract,
    generate_audit_log,
    run_agentic_pipeline,
    write_contract,
)
from carevalue_claims_ml.agent_contracts import AgentHandoffContract, validate_handoff_contract
from carevalue_claims_ml.llm_optional import deterministic_postprocess, retrieve_reference_context
from carevalue_claims_ml.insurance_policy import InsurancePolicyConfig, apply_contract_constraints, run_policy_scenarios
from carevalue_claims_ml.policy_simulation import evaluate_agent_strategy, simulate_policy
from carevalue_claims_ml.reporting import write_summary_report

app = typer.Typer(add_completion=False)
db_app = typer.Typer(add_completion=False)
data_app = typer.Typer(add_completion=False)
features_app = typer.Typer(add_completion=False)
models_app = typer.Typer(add_completion=False)
report_app = typer.Typer(add_completion=False)
policy_app = typer.Typer(add_completion=False)
agents_app = typer.Typer(add_completion=False)
episodes_app = typer.Typer(add_completion=False)
benchmarks_app = typer.Typer(add_completion=False)
careops_app = typer.Typer(add_completion=False)
journey_app = typer.Typer(add_completion=False)

app.add_typer(db_app, name="db")
app.add_typer(data_app, name="data")
app.add_typer(features_app, name="features")
app.add_typer(models_app, name="models")
app.add_typer(report_app, name="report")
app.add_typer(policy_app, name="policy")
app.add_typer(agents_app, name="agents")
app.add_typer(episodes_app, name="episodes")
app.add_typer(benchmarks_app, name="benchmarks")
app.add_typer(careops_app, name="careops")
app.add_typer(journey_app, name="journey")


@app.callback()
def main():
    pass


@app.command("libraries")
def libraries():
    print(
        {
            "umbrella": "VBC Intelligence OS",
            "sublibraries": [
                "vbc_intel_core",
                "vbc_intel_episodes",
                "vbc_intel_policy",
                "vbc_intel_benchmarks",
                "vbc_intel_careops",
            ],
        }
    )


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


@models_app.command("train-suite")
def models_train_suite(
    suite: str = "maximal",
    families: str = "",
    settings_path: Path = Path("config/settings.yaml"),
    output_dir: Path = Path("models"),
    database_url: str | None = None,
):
    settings = load_settings(settings_path)
    engine = create_db_engine(get_database_url(database_url or settings.database_url))
    features_df = build_member_month_features(engine, settings.feature_window_months)
    labels_df = build_high_cost_label(engine, settings.label_horizon_months)
    family_list = [f.strip() for f in families.split(",") if f.strip()] if families else settings.model_families
    results = train_model_suite(
        features_df,
        labels_df,
        output_dir=output_dir,
        suite=suite,
        model_families=family_list,
    )
    payload = {
        name: {"metrics": res.metrics, "artifact": str(res.artifact_path), "run_id": res.run_id}
        for name, res in results.items()
    }
    print(payload)


@models_app.command("train-use-cases")
def models_train_use_cases(
    settings_path: Path = Path("config/settings.yaml"),
    output_dir: Path = Path("models"),
    database_url: str | None = None,
):
    settings = load_settings(settings_path)
    engine = create_db_engine(get_database_url(database_url or settings.database_url))
    features_df = build_member_month_features(engine, settings.feature_window_months)
    labels_df = build_high_cost_label(engine, settings.label_horizon_months)
    families = [
        "vbc_cost_optimizer",
        "outcome_improvement_optimizer",
        "claims_behavior_predictor",
        "provider_advisory_ranker",
    ]
    results = train_model_suite(features_df, labels_df, output_dir=output_dir, suite="maximal", model_families=families)
    print({k: {"metrics": v.metrics, "artifact": str(v.artifact_path)} for k, v in results.items()})


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


@models_app.command("evaluate")
def models_evaluate(
    predictions: Path,
    output_path: Path = Path("reports/evaluation.json"),
):
    df = pd.read_csv(predictions)
    result = evaluate_predictions(df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"metrics": result.metrics, "subgroup_metrics": result.subgroup_metrics}, indent=2),
        encoding="utf-8",
    )
    print({"evaluation": str(output_path), "metrics": result.metrics})


@models_app.command("leaderboard")
def models_leaderboard(
    predictions: Path,
    model_name: str = "adhoc_model",
    run_id: str = "manual",
    output_dir: Path = Path("reports"),
):
    df = pd.read_csv(predictions)
    result = evaluate_predictions(df)
    leaderboard_path, model_card_path = write_leaderboard_artifacts(output_dir, model_name, run_id, result)
    print({"leaderboard": str(leaderboard_path), "model_card": str(model_card_path)})


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


@episodes_app.command("build")
def episodes_build(
    claims_path: Path,
    output_path: Path = Path("reports/episodes.csv"),
    archetype: str = "general",
    window_days: int = 90,
):
    claims = pd.read_csv(claims_path)
    episodes = build_bundled_episodes(claims, archetype=archetype, window_days=window_days)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    episodes.to_csv(output_path, index=False)
    print({"episodes": str(output_path), "rows": len(episodes), "archetype": archetype})


@episodes_app.command("score")
def episodes_score(
    episodes_path: Path,
    output_path: Path = Path("reports/episode_scores.csv"),
    diagnosis_code_col: str | None = None,
    procedure_code_col: str | None = None,
):
    episodes_df = pd.read_csv(episodes_path)
    scored = score_episode_risk(
        episodes_df,
        diagnosis_code_col=diagnosis_code_col,
        procedure_code_col=procedure_code_col,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_path, index=False)
    print({"episode_scores": str(output_path), "rows": len(scored)})


@journey_app.command("merge")
def journey_merge(
    medical_path: Path,
    output_path: Path = Path("reports/journey_unified.csv"),
    pharmacy_path: Path | None = None,
):
    medical = pd.read_csv(medical_path)
    pharmacy = pd.read_csv(pharmacy_path) if pharmacy_path is not None else None
    unified = merge_medical_and_pharmacy_claims(medical, pharmacy)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    unified.to_csv(output_path, index=False)
    print({"journey_unified": str(output_path), "rows": len(unified)})


@journey_app.command("monthly-features")
def journey_monthly_features(
    claims_path: Path,
    output_path: Path = Path("reports/journey_member_month_utilization.csv"),
):
    claims = pd.read_csv(claims_path)
    features = monthly_utilization_features(claims)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False)
    print({"monthly_utilization": str(output_path), "rows": len(features)})


@policy_app.command("simulate")
def policy_simulate(
    scores_path: Path,
    budget: int = 100,
    output_path: Path = Path("reports/policy_simulation.json"),
):
    scores = pd.read_csv(scores_path)
    metrics = simulate_policy(scores, budget=budget)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print({"policy_metrics": metrics, "output": str(output_path)})


@benchmarks_app.command("pmpm")
def benchmarks_pmpm(
    settings_path: Path = Path("config/settings.yaml"),
    output_path: Path = Path("reports/benchmark_pmpm.csv"),
    database_url: str | None = None,
):
    settings = load_settings(settings_path)
    engine = create_db_engine(get_database_url(database_url or settings.database_url))
    pmpm_df = calculate_pmpm(engine)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pmpm_df.to_csv(output_path, index=False)
    print({"benchmark_pmpm": str(output_path), "rows": len(pmpm_df)})


@policy_app.command("scenario")
def policy_scenario(
    recommendations_path: Path,
    output_path: Path = Path("reports/policy_scenarios.json"),
):
    recs = pd.read_csv(recommendations_path)
    scenarios = run_policy_scenarios(recs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(scenarios, indent=2), encoding="utf-8")
    print({"policy_scenarios": scenarios, "output": str(output_path)})


@policy_app.command("enforce")
def policy_enforce(
    recommendations_path: Path,
    output_path: Path = Path("reports/recommendations_policy_enforced.csv"),
    shared_savings_rate: float = 0.5,
    downside_cap: float = 50000.0,
    outreach_budget: int = 100,
):
    recs = pd.read_csv(recommendations_path)
    constrained = apply_contract_constraints(
        recs,
        InsurancePolicyConfig(
            shared_savings_rate=float(shared_savings_rate),
            downside_cap=float(downside_cap),
            outreach_budget=int(outreach_budget),
        ),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    constrained.to_csv(output_path, index=False)
    print({"policy_enforced_output": str(output_path), "rows": len(constrained)})


@agents_app.command("run")
def agents_run(
    scores_path: Path,
    output_path: Path = Path("reports/agent_recommendations.csv"),
    audit_path: Path = Path("reports/agent_audit.csv"),
    contract_path: Path = Path("reports/agent_handoff_contract.json"),
    run_id: str = "manual",
    contract_id: str = "DEMO",
):
    scored = pd.read_csv(scores_path)
    recs = run_agentic_pipeline(scored)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    recs.to_csv(output_path, index=False)
    audit = generate_audit_log(recs, {"alerts": "none"}, audit_path)
    contract = build_handoff_contract(
        "agents_run_output",
        recs,
        run_id=run_id,
        contract_id=contract_id,
        policy_version="insurance_v1",
        upstream_stage_ids=["cohort_detection", "risk_triage", "care_gap", "contract_impact"],
        quality_status="ok",
    )
    write_contract(contract, contract_path)
    print(
        {
            "recommendations": str(output_path),
            "audit": str(audit_path),
            "contract": str(contract_path),
            "rows": len(recs),
            "audit_rows": len(audit),
        }
    )


@careops_app.command("run")
def careops_run(
    scores_path: Path,
    output_path: Path = Path("reports/agent_recommendations.csv"),
    audit_path: Path = Path("reports/agent_audit.csv"),
    contract_path: Path = Path("reports/agent_handoff_contract.json"),
    run_id: str = "manual",
    contract_id: str = "DEMO",
):
    agents_run(
        scores_path=scores_path,
        output_path=output_path,
        audit_path=audit_path,
        contract_path=contract_path,
        run_id=run_id,
        contract_id=contract_id,
    )


@agents_app.command("validate-contract")
def agents_validate_contract(contract_path: Path):
    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    contract = AgentHandoffContract(
        stage=str(payload["stage"]),
        payload=list(payload["payload"]),
        version=str(payload["version"]),
        run_id=str(payload.get("run_id", "manual")),
        contract_id=str(payload.get("contract_id", "DEMO")),
        policy_version=str(payload.get("policy_version", "v1")),
        upstream_stage_ids=list(payload.get("upstream_stage_ids", [])),
        quality_status=str(payload.get("quality_status", "ok")),
        generated_at=str(payload.get("generated_at", "")),
    )
    validate_handoff_contract(contract)
    print({"contract": str(contract_path), "valid": True})


@agents_app.command("evaluate")
def agents_evaluate(
    agent_output_path: Path,
    baseline_output_path: Path,
    budget: int = 100,
    output_path: Path = Path("reports/agent_eval.json"),
):
    agent_df = pd.read_csv(agent_output_path)
    baseline_df = pd.read_csv(baseline_output_path)
    metrics = evaluate_agent_strategy(agent_df, baseline_df, budget=budget)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print({"agent_eval": str(output_path), "metrics": metrics})


@agents_app.command("llm-draft")
def agents_llm_draft(
    member_payload_path: Path,
    model_cards_path: Path = Path("MODEL_CARDS.md"),
    output_path: Path = Path("reports/llm_optional_draft.json"),
):
    member_payload = json.loads(member_payload_path.read_text(encoding="utf-8"))
    context = retrieve_reference_context(
        model_cards_path=model_cards_path,
        benchmark_summary="Benchmark trend and PMPM context",
        contract_rules="Recommendation-only policy; no autonomous clinical action",
    )
    drafted = deterministic_postprocess(context + json.dumps(member_payload))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(drafted, indent=2), encoding="utf-8")
    print({"llm_optional_draft": str(output_path)})
