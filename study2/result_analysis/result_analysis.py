from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


STUDY2_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = STUDY2_DIR / "data"
OUTPUT_PATH = Path(__file__).resolve().parent / "study2.txt"
NUM_DAYS = 14

INTERVENTION_GROUPS = [
    "information_high",
    "information_medium",
    "information_low",
    "assistant_high",
    "assistant_medium",
    "assistant_low",
    "structure_high",
    "structure_medium",
    "structure_low",
]

REQUIRED_COLUMNS = {
    "model",
    "group",
    "day",
    "human_d",
    "human_ci_lower",
    "human_ci_upper",
    "llm_d",
}


def get_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Analyze one model. By default, analyze every daily effect-size CSV.",
    )
    return parser


def find_input_paths(model_name: str | None) -> list[Path]:
    if model_name is not None:
        path = DATA_DIR / f"human_llm_daily_effect_sizes_{model_name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Effect-size file not found: {path}")
        return [path]

    paths = sorted(DATA_DIR.glob("human_llm_daily_effect_sizes_*.csv"))
    if not paths:
        raise FileNotFoundError(
            f"No human_llm_daily_effect_sizes_*.csv files found in {DATA_DIR}."
        )
    return paths


def load_effect_sizes(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, encoding="utf-8-sig")
    missing_columns = REQUIRED_COLUMNS - set(data.columns)
    if missing_columns:
        raise ValueError(f"{path} is missing columns: {sorted(missing_columns)}")

    data["day"] = pd.to_numeric(data["day"], errors="raise").astype(int)
    for column in ["human_d", "human_ci_lower", "human_ci_upper", "llm_d"]:
        data[column] = pd.to_numeric(data[column], errors="raise")
    return data


def safe_pearson(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    """Calculate Pearson's r using finite paired observations only."""
    paired = pd.DataFrame({"x": x, "y": y}).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    n_pairs = len(paired)
    if (
        n_pairs < 2
        or paired["x"].nunique() < 2
        or paired["y"].nunique() < 2
    ):
        return float("nan"), float("nan"), n_pairs

    correlation, correlation_p = stats.pearsonr(paired["x"], paired["y"])
    return float(correlation), float(correlation_p), n_pairs


def analyze_model(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    model_values = data["model"].dropna().astype(str).unique()
    if len(model_values) != 1:
        raise ValueError(f"Expected one model per input file, found {model_values}.")
    model_name = model_values[0]

    rows: list[dict[str, object]] = []
    for group in INTERVENTION_GROUPS:
        group_data = data.loc[data["group"] == group].sort_values("day")
        expected_days = list(range(1, NUM_DAYS + 1))
        if group_data["day"].tolist() != expected_days:
            raise ValueError(
                f"Expected days 1-{NUM_DAYS} for model={model_name}, group={group}."
            )

        correlation, correlation_p, correlation_n = safe_pearson(
            group_data["human_d"], group_data["llm_d"]
        )
        day1 = group_data.loc[group_data["day"] == 1].iloc[0]
        day14 = group_data.loc[group_data["day"] == NUM_DAYS].iloc[0]

        short_success = bool(
            day1["human_ci_lower"] <= day1["llm_d"] <= day1["human_ci_upper"]
        )
        long_success = bool(
            day14["human_ci_lower"]
            <= day14["llm_d"]
            <= day14["human_ci_upper"]
        )

        rows.append(
            {
                "model": model_name,
                "group": group,
                "trajectory_r": float(correlation),
                "trajectory_p": float(correlation_p),
                "trajectory_n": correlation_n,
                "short_term_success": short_success,
                "short_term_success_rate": float(short_success),
                "long_term_success": long_success,
                "long_term_success_rate": float(long_success),
            }
        )

    group_results = pd.DataFrame(rows)
    overall_r, overall_p, overall_n = safe_pearson(
        data["human_d"], data["llm_d"]
    )
    overall = {
        "pearson_r": overall_r,
        "pearson_p": overall_p,
        "pearson_n": float(overall_n),
        "short_term_replication_rate": float(
            group_results["short_term_success"].mean()
        ),
        "long_term_replication_rate": float(
            group_results["long_term_success"].mean()
        ),
    }
    return group_results, overall


def format_float(value: float) -> str:
    return "NA" if not np.isfinite(value) else f"{value:.4f}"


def build_report(results: list[tuple[pd.DataFrame, dict[str, float]]]) -> str:
    lines = ["Study 2: Human vs. LLM replication analysis"]

    for group_results, overall in results:
        model_name = str(group_results["model"].iloc[0])
        lines.extend(["", f"Model: {model_name}", ""])

        for row in group_results.itertuples(index=False):
            lines.append(f"Group: {row.group}")
            lines.append(
                "  Effect-size trajectory correlation: "
                f"r = {format_float(row.trajectory_r)}, "
                f"p = {format_float(row.trajectory_p)}, "
                f"n = {row.trajectory_n}"
            )
            lines.append(
                "  Short-term replication success (Day 1): "
                f"{'success' if row.short_term_success else 'failure'} "
                f"({row.short_term_success_rate:.0%})"
            )
            lines.append(
                "  Long-term replication success (Day 14): "
                f"{'success' if row.long_term_success else 'failure'} "
                f"({row.long_term_success_rate:.0%})"
            )

        short_count = int(group_results["short_term_success"].sum())
        long_count = int(group_results["long_term_success"].sum())
        lines.extend(
            [
                "",
                "Overall human-LLM Pearson correlation: "
                f"r = {format_float(overall['pearson_r'])}, "
                f"p = {format_float(overall['pearson_p'])}, "
                f"n = {int(overall['pearson_n'])}",
                "Overall short-term replication rate: "
                f"{short_count}/9 ({overall['short_term_replication_rate']:.2%})",
                "Overall long-term replication rate: "
                f"{long_count}/9 ({overall['long_term_replication_rate']:.2%})",
            ]
        )

    lines.extend(
        [
            "",
            "Model-level Pearson summary",
            "model | pearson_r | pearson_p | valid_pairs",
        ]
    )
    for group_results, overall in results:
        model_name = str(group_results["model"].iloc[0])
        lines.append(
            f"{model_name} | {format_float(overall['pearson_r'])} | "
            f"{format_float(overall['pearson_p'])} | "
            f"{int(overall['pearson_n'])}"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    args = get_argument_parser().parse_args()
    results = [analyze_model(load_effect_sizes(path)) for path in find_input_paths(args.model_name)]
    report = build_report(results)
    OUTPUT_PATH.write_text(report, encoding="utf-8")
    print(report, end="")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
