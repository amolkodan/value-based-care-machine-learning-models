from __future__ import annotations

from pathlib import Path

from sqlalchemy import text
from sqlalchemy.engine import Engine

from carevalue_claims_ml.db import execute_sql_file


def initialize_schema(engine: Engine, schema_path: Path) -> None:
    execute_sql_file(engine, schema_path.read_text(encoding="utf-8"))


def build_member_months(engine: Engine, sql_path: Path) -> None:
    with engine.begin() as connection:
        connection.execute(text("delete from vbc.member_months"))
        connection.execute(text(sql_path.read_text(encoding="utf-8")))
