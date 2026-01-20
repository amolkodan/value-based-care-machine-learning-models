# carevalue-claims-ml

Machine learning and analytics repository for value based care built on top of a medical claims backbone. It extends a standard claims and eligibility data model with repeatable pipelines for attribution, benchmark calculation, risk modeling, and cost forecasting.

## What this repo includes

- Postgres reference schema compatible with common claims extracts
- Synthetic data generator and sample CSVs for local runs
- Feature engineering for member-month features
- ML models:
  - Risk score model (classification) for prospective high-cost risk
  - Cost forecast model (regression) for prospective allowed amount
- Attribution utilities (primary clinician attribution based on visit volume)
- Benchmarks (PMPM, trend, contract target) and performance scoring
- CLI for end to end runs (init, load, features, train, score, report)
- Tests and GitHub Actions CI

## Quickstart

### 1) Start Postgres
```bash
docker compose up -d
```

### 2) Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
```

### 3) Run the local demo
```bash
./scripts/run_local_demo.sh
```

The demo will:
- Initialize the database schema
- Generate synthetic claims and eligibility data
- Load data into Postgres
- Build member-months
- Create ML features and labels
- Train models
- Score members and produce a report

## CLI

After installation:
```bash
carevalue-ml --help
```

## License

MIT
