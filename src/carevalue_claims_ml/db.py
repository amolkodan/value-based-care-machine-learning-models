from __future__ import annotations

import os
from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


def get_database_url(explicit_url: str | None = None) -> str:
    if explicit_url:
        return explicit_url
    env_url = os.getenv("DATABASE_URL")
    if not env_url:
        raise RuntimeError("DATABASE_URL is not set. Copy .env.example to .env and set it.")
    return env_url


def create_db_engine(database_url: str) -> Engine:
    return create_engine(database_url, pool_pre_ping=True, future=True)


@contextmanager
def db_connection(engine: Engine):
    with engine.connect() as connection:
        yield connection


def execute_sql_file(engine: Engine, sql_text: str) -> None:
    statements = [s.strip() for s in sql_text.split(";") if s.strip()]
    with engine.begin() as connection:
        for statement in statements:
            connection.execute(text(statement))
