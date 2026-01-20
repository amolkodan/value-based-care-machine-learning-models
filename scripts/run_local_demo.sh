#!/usr/bin/env bash
set -euo pipefail

if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

export DATABASE_URL="${DATABASE_URL:-postgresql+psycopg2://vbc:vbc@localhost:5432/vbc}"

carevalue-ml db init
carevalue-ml data generate --output data/generated --member-count 600 --months 24 --start-month 2023-01-01
carevalue-ml data load --input-dir data/generated
carevalue-ml features build
carevalue-ml models train

RISK_ARTIFACT=$(ls -1 models/risk_model_*.joblib | tail -n 1)
COST_ARTIFACT=$(ls -1 models/cost_model_*.joblib | tail -n 1)

carevalue-ml models score --artifact "$RISK_ARTIFACT" --model-name risk_high_cost --run-id demo_risk
carevalue-ml models score --artifact "$COST_ARTIFACT" --model-name cost_forecast --run-id demo_cost
carevalue-ml report summary
