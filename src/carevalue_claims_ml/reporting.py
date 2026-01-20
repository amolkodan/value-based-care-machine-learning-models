from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_summary_report(output_dir: Path, pmpm_df: pd.DataFrame, savings_df: pd.DataFrame) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "summary_report.csv"
    merged = pmpm_df.merge(savings_df, on="month", how="left")
    merged.to_csv(path, index=False)
    return path
