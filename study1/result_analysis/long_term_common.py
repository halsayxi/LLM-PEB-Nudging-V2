"""Shared model and CSV helpers for Study 1 long-term analyses."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_NAMES = [
    "gpt-3.5-turbo-0125",
    "deepseek-v3",
    "claude-haiku-4-5-20251001",
    "gpt-4o-2024-11-20",
    "deepseek-v4-flash",
    "gpt-5.4",
]

MODEL_PREFIXES = {
    "gpt-3.5-turbo-0125": "gpt35",
    "deepseek-v3": "deepseekv3",
    "claude-haiku-4-5-20251001": "claudehaiku45",
    "gpt-4o-2024-11-20": "gpt4o",
    "deepseek-v4-flash": "deepseekv4flash",
    "gpt-5.4": "gpt54",
}

EXPECTED_AGENTS_PER_GROUP = 50


def add_analysis_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model_name",
        nargs="+",
        default=["all"],
        help="One or more model names, or 'all' (default).",
    )
    parser.add_argument(
        "--results_root",
        type=Path,
        default=None,
        help="Root containing <model_name>_res directories.",
    )
    parser.add_argument(
        "--csv_path",
        type=Path,
        default=None,
        help="Combined Study 1 long-term CSV to update.",
    )


def resolve_model_names(values: list[str]) -> list[str]:
    if not values or values == ["all"]:
        return MODEL_NAMES.copy()
    if "all" in values:
        raise ValueError("Use --model_name all alone, or list explicit models.")
    unknown = [value for value in values if value not in MODEL_NAMES]
    if unknown:
        raise ValueError(f"Unsupported model(s): {unknown}")
    return list(dict.fromkeys(values))


def model_columns(model_name: str) -> tuple[str, str, str]:
    prefix = MODEL_PREFIXES[model_name]
    return (
        f"{prefix}_llm_ATE",
        f"{prefix}_llm_Robust_SE",
        f"{prefix}_llm_p_value",
    )


def validate_group_results(group_dirs: list[Path]) -> None:
    """Require exactly 50 numeric per-agent JSON files in every analyzed group."""
    errors = []
    for group_dir in group_dirs:
        if not group_dir.is_dir():
            errors.append(f"missing directory: {group_dir}")
            continue
        files = [
            path
            for path in group_dir.glob("*.json")
            if path.is_file() and path.stem.isdigit()
        ]
        if len(files) != EXPECTED_AGENTS_PER_GROUP:
            errors.append(
                f"{group_dir}: expected {EXPECTED_AGENTS_PER_GROUP} agent files, "
                f"found {len(files)}"
            )
    if errors:
        raise FileNotFoundError("Incomplete long-term results:\n- " + "\n- ".join(errors))


def update_combined_csv(
    csv_path: Path,
    model_name: str,
    rows: list[dict[str, float | str]],
) -> None:
    """Upsert one model's results without overwriting other models' columns."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    ate_col, se_col, p_col = model_columns(model_name)
    for column in (ate_col, se_col, p_col):
        if column not in df.columns:
            df[column] = np.nan

    for row in rows:
        mask = df["study_id"].astype(str) == str(row["study_id"])
        if not mask.any():
            print(f"Warning: study_id {row['study_id']} not found in CSV.")
            continue
        df.loc[mask, ate_col] = row["llm_ATE"]
        df.loc[mask, se_col] = row["llm_Robust_SE"]
        df.loc[mask, p_col] = row["llm_p_value"]

    df.to_csv(csv_path, index=False)
    print(f"Updated {model_name} columns in: {csv_path}")
