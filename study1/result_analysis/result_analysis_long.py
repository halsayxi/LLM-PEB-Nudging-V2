"""Summarize replication success for all Study 1 long-term models."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from long_term_common import (
    MODEL_PREFIXES,
    add_analysis_arguments,
    model_columns,
    resolve_model_names,
)


def read_csv_robust(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "latin1"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def parse_llm_significance(value) -> bool:
    """Study 1 long-term analyses use p < .10 as significant."""
    if pd.isna(value):
        return False
    try:
        text = str(value).replace("<", "").replace("=", "").replace(">", "")
        return float(text) < 0.1
    except (TypeError, ValueError):
        return False


def effect_sign(value) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0
    if np.isnan(number) or number == 0:
        return 0
    return 1 if number > 0 else -1


def analyze_model(df: pd.DataFrame, model_name: str) -> list[str]:
    ate_col, _, p_col = model_columns(model_name)
    missing = [column for column in (ate_col, p_col) if column not in df.columns]
    if missing:
        raise ValueError(
            f"Missing result columns for {model_name}: {missing}. "
            "Run all three process_result_longterm*.py scripts first."
        )

    result = df.copy()
    result["human_sig"] = result["significance_binary"].astype(int).eq(1)
    result["llm_sig"] = result[p_col].apply(parse_llm_significance)
    result["human_dir"] = result["ATE"].apply(effect_sign)
    result["llm_dir"] = result[ate_col].apply(effect_sign)
    result["same_dir"] = (result["human_dir"] * result["llm_dir"]) > 0

    # Missing model results must not be counted as successful non-significant results.
    result["has_llm_result"] = (
        pd.to_numeric(result[ate_col], errors="coerce").notna()
        & pd.to_numeric(result[p_col], errors="coerce").notna()
    )
    result["replicated"] = result["has_llm_result"] & (
        (result["human_sig"] & result["llm_sig"] & result["same_dir"])
        | (~result["human_sig"] & ~result["llm_sig"])
    )

    correlation_data = result[["ATE", ate_col]].apply(
        pd.to_numeric, errors="coerce"
    ).dropna()
    if len(correlation_data) >= 2:
        correlation, correlation_p = stats.pearsonr(
            correlation_data["ATE"], correlation_data[ate_col]
        )
    else:
        correlation, correlation_p = np.nan, np.nan

    completed = int(result["has_llm_result"].sum())
    successes = int(result["replicated"].sum())
    total = len(result)
    failed_ids = result.loc[
        result["has_llm_result"] & ~result["replicated"], "study_id"
    ].astype(str).tolist()
    missing_ids = result.loc[
        ~result["has_llm_result"], "study_id"
    ].astype(str).tolist()

    success_line = (
        f"Replication success: {successes}/{completed} ({successes / completed:.0%})"
        if completed
        else "Replication success: N/A"
    )
    return [
        f"=== {model_name} ({MODEL_PREFIXES[model_name]}) ===",
        f"Completed studies: {completed}/{total}",
        f"Pearson r (human ATE vs. LLM ATE): {correlation:.2f}, p = {correlation_p:.3g}",
        success_line,
        f"Failed studies: {', '.join(failed_ids) if failed_ids else 'None'}",
        f"Missing studies: {', '.join(missing_ids) if missing_ids else 'None'}",
    ]


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    add_analysis_arguments(parser)
    parser.add_argument(
        "--output_txt",
        type=Path,
        default=script_dir / "study1_long.txt",
    )
    args = parser.parse_args()

    csv_path = (args.csv_path or script_dir / "../data/study_1_long.csv").resolve()
    df = read_csv_robust(csv_path)
    required = {"study_id", "ATE", "significance_binary"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    output_lines = ["=== Benchmarking LLMs vs. Long-run Field Experiments ==="]
    for model_name in resolve_model_names(args.model_name):
        output_lines.extend(["", *analyze_model(df, model_name)])

    output_text = "\n".join(output_lines)
    print(output_text)
    output_path = args.output_txt.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output_text, encoding="utf-8")
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
