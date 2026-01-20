from __future__ import annotations

from datetime import date
from pathlib import Path

from carevalue_claims_ml.data_generation import SyntheticDataConfig, add_months, generate_synthetic_dataset


def test_add_months():
    assert add_months(date(2023, 1, 1), 1) == date(2023, 2, 1)
    assert add_months(date(2023, 12, 1), 1) == date(2024, 1, 1)


def test_generate_synthetic_dataset(tmp_path: Path):
    cfg = SyntheticDataConfig(member_count=10, months=3, start_month=date(2023, 1, 1), random_seed=1)
    generate_synthetic_dataset(tmp_path, cfg)
    assert (tmp_path / "members.csv").exists()
    assert (tmp_path / "claims_header.csv").exists()
