## Optional Databricks Track

This project is local-first and cloud-agnostic. This folder provides optional templates
for teams that want Databricks-compatible conventions without changing core code.

### Suggested layering

- `bronze`: raw ingested claims and eligibility extracts
- `silver`: cleaned conformed member-month and attribution tables
- `gold`: model-ready features, predictions, and contract analytics outputs

### MLflow compatibility

- Use the run metadata template in `mlflow_run_tags.example.json`.
- You can emit the same fields from local runs, then forward to Databricks MLflow later.

### Agent orchestration lineage

- Track `agent_policy_name`, `guardrail_profile`, and `contract_version` tags.
- Persist recommendation outputs and audit logs as gold-layer artifacts.
- Maintain contract validation status in run tags for reproducibility.

### Scalable simulation guidance

- Run agent-vs-baseline evaluations as batch jobs over historical scoring snapshots.
- Store cohort partitions by month and contract for replayable simulation windows.
- Track policy scenario runs (`optimistic`, `base`, `stress`) as separate experiment groups.
- Persist stage-level contracts to support replay and post-mortem auditability.

### Delta conventions

- Use partitioning by `month` where practical.
- Prefer append-only writes for lineage and reproducibility.
