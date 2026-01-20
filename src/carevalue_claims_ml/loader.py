from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine


def load_csv_to_table(engine: Engine, csv_path: Path, table_fqn: str) -> int:
    df = pd.read_csv(csv_path)
    schema, table = table_fqn.split(".")
    with engine.begin() as connection:
        connection.execute(text(f"truncate table {table_fqn} cascade"))
        df.to_sql(table, con=connection, schema=schema, if_exists="append", index=False, method="multi")
    return len(df)


def load_generated_folder(engine: Engine, input_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    counts["members"] = load_csv_to_table(engine, input_dir / "members.csv", "vbc.members")
    counts["providers"] = load_csv_to_table(engine, input_dir / "providers.csv", "vbc.providers")
    counts["eligibility"] = load_csv_to_table(engine, input_dir / "eligibility.csv", "vbc.eligibility")
    counts["claims_header"] = load_csv_to_table(engine, input_dir / "claims_header.csv", "vbc.claims_header")
    counts["claims_line"] = load_csv_to_table(engine, input_dir / "claims_line.csv", "vbc.claims_line")
    counts["diagnosis"] = load_csv_to_table(engine, input_dir / "diagnosis.csv", "vbc.diagnosis")
    counts["benchmarks"] = load_csv_to_table(engine, input_dir / "benchmarks.csv", "vbc.benchmarks")
    return counts
