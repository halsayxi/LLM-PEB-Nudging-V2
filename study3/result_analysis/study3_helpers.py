from pathlib import Path
from typing import List, Tuple

import pandas as pd


MISMATCH_COLOR = "#A9A9A9"
CATEGORY_ORDER = ["information", "structure", "assistance"]

CATEGORY_PALETTE = {
    "information": "#0072B2",
    "structure": "#009E73",
    "assistance": "#D55E00",
}

TECHNIQUE_PALETTE = {
    "commitment": "#0072B2",
    "composition": "#56B4E9",
    "consequence": "#E69F00",
    "default": "#009E73",
    "effort": "#CC79A7",
    "reminder": "#D55E00",
    "social_reference": "#F0E442",
    "translation": "#7A68A6",
    "visibility": "#1B9E77",
}

TECHNIQUE_ORDER = list(TECHNIQUE_PALETTE.keys())


def local_path(filename: str) -> Path:
    return Path(__file__).resolve().parent / filename


def replication_success(
    simulated_d: float,
    ci_lower: float,
    ci_upper: float,
) -> bool:
    d = pd.to_numeric(simulated_d, errors="coerce")
    lower = pd.to_numeric(ci_lower, errors="coerce")
    upper = pd.to_numeric(ci_upper, errors="coerce")
    if pd.isna(d) or pd.isna(lower) or pd.isna(upper):
        return False
    return bool((d >= lower) and (d <= upper))


def prepare_day30_study_data() -> pd.DataFrame:
    rep = pd.read_csv(local_path("../data/nudge-replication.csv"))
    day30 = pd.read_csv(local_path("../data/long_round_average.csv")).rename(
        columns={"Unnamed: 0": "study_id"}
    )

    rep_subset = rep[
        [
            "study_id",
            "effect_size",
            "ci_lower",
            "ci_upper",
            "intervention_category",
            "intervention_technique",
        ]
    ]
    day30_subset = day30[["study_id", "d_round_1", "d_round_30"]].rename(
        columns={"d_round_1": "llm_effect_size", "d_round_30": "llm_d_day30"}
    )

    merged = rep_subset.merge(day30_subset, on="study_id", how="inner")

    merged["short_success"] = merged.apply(
        lambda row: replication_success(
            row["llm_effect_size"], row["ci_lower"], row["ci_upper"]
        ),
        axis=1,
    )
    merged["long_success"] = merged.apply(
        lambda row: replication_success(
            row["llm_d_day30"],
            row["ci_lower"],
            row["ci_upper"],
        ),
        axis=1,
    )
    merged["long_fail"] = merged["short_success"] & ~merged["long_success"]
    merged["short_color"] = merged.apply(
        lambda row: TECHNIQUE_PALETTE.get(row["intervention_technique"], "#4C4C4C")
        if row["short_success"]
        else MISMATCH_COLOR,
        axis=1,
    )
    merged["long_color"] = merged.apply(
        lambda row: TECHNIQUE_PALETTE.get(row["intervention_technique"], "#4C4C4C")
        if row["long_success"]
        else MISMATCH_COLOR,
        axis=1,
    )

    merged = merged.sort_values("effect_size", ascending=False).reset_index(drop=True)
    merged["study_label"] = merged.index + 1
    return merged


def split_short_term_success_groups(day30_df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    short_term_success = day30_df[day30_df["short_success"]].copy()
    success_ids = short_term_success.loc[short_term_success["long_success"], "study_id"].tolist()
    fail_ids = short_term_success.loc[short_term_success["long_fail"], "study_id"].tolist()
    return success_ids, fail_ids


def ordered_technique_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["intervention_category", "intervention_technique"], as_index=False)
        .size()
        .rename(columns={"size": "n"})
    )
    category_rank = {name: idx for idx, name in enumerate(CATEGORY_ORDER)}
    technique_rank = {name: idx for idx, name in enumerate(TECHNIQUE_ORDER)}
    summary["category_rank"] = summary["intervention_category"].map(category_rank).fillna(999)
    summary["technique_rank"] = (
        summary["intervention_technique"].map(technique_rank).fillna(999)
    )
    summary = summary.sort_values(
        ["category_rank", "n", "technique_rank", "intervention_technique"],
        ascending=[True, False, True, True],
    ).reset_index(drop=True)
    return summary.drop(columns=["category_rank", "technique_rank"])


def format_technique_label(technique: str) -> str:
    return technique.replace("_", " ").title()
