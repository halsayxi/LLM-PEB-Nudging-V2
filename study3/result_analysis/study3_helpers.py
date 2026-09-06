from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

MODEL_PREFIX = "deepseekv4flash_llm"
MODEL_LABEL = "DeepSeek-V4-Flash-Think"
MISMATCH_COLOR = "#A9A9A9"
CATEGORY_ORDER = ["information", "structure", "assistance"]
CATEGORY_PALETTE = {"information": "#0072B2", "structure": "#009E73", "assistance": "#D55E00"}
TECHNIQUE_PALETTE = {
    "commitment": "#0072B2", "composition": "#56B4E9", "consequence": "#E69F00",
    "default": "#009E73", "effort": "#CC79A7", "reminder": "#D55E00",
    "social_reference": "#F0E442", "translation": "#7A68A6", "visibility": "#1B9E77",
}
TECHNIQUE_ORDER = list(TECHNIQUE_PALETTE)


def local_path(filename: str) -> Path:
    return Path(__file__).resolve().parent / filename


def replication_success(human_d: float, agent_ci_lower: float, agent_ci_upper: float) -> bool:
    """Success means that the agent 95% CI contains the human effect size."""
    human = pd.to_numeric(human_d, errors="coerce")
    lower = pd.to_numeric(agent_ci_lower, errors="coerce")
    upper = pd.to_numeric(agent_ci_upper, errors="coerce")
    if pd.isna(human) or pd.isna(lower) or pd.isna(upper):
        return False
    return bool(lower <= human <= upper)


def agent_effect_ci(effect_size, n_control, n_intervention, z: float = 1.96):
    d = np.asarray(effect_size, dtype=float)
    n_c = np.asarray(n_control, dtype=float)
    n_i = np.asarray(n_intervention, dtype=float)
    variance = (n_c + n_i) / (n_c * n_i) + d**2 / (2.0 * (n_c + n_i))
    half_width = z * np.sqrt(variance)
    return d - half_width, d + half_width


def add_agent_ci(df: pd.DataFrame, effect_col: str, prefix: str) -> pd.DataFrame:
    out = df.copy()
    lower, upper = agent_effect_ci(out[effect_col], out["n_control"], out["n_intervention"])
    out[f"{prefix}_ci_lower"] = lower
    out[f"{prefix}_ci_upper"] = upper
    return out


def prepare_day30_study_data() -> pd.DataFrame:
    rep = pd.read_csv(local_path("../data/nudge-replication.csv"))
    day30 = pd.read_csv(local_path("../data/long_round_average.csv"))
    rep_subset = rep[
        ["study_id", "effect_size", "ci_lower", "ci_upper", "n_control", "n_intervention",
         "intervention_category", "intervention_technique",
         f"{MODEL_PREFIX}_effect_size", f"{MODEL_PREFIX}_ci_lower", f"{MODEL_PREFIX}_ci_upper"]
    ].rename(columns={
        "effect_size": "human_effect_size",
        f"{MODEL_PREFIX}_effect_size": "agent_effect_size",
        f"{MODEL_PREFIX}_ci_lower": "agent_ci_lower",
        f"{MODEL_PREFIX}_ci_upper": "agent_ci_upper",
    })
    day30_subset = day30[["study_id", "d_round_30"]].rename(columns={"d_round_30": "agent_d_day30"})
    merged = add_agent_ci(rep_subset.merge(day30_subset, on="study_id", how="inner"),
                          "agent_d_day30", "day30")
    merged["short_success"] = merged.apply(
        lambda r: replication_success(r["human_effect_size"], r["agent_ci_lower"], r["agent_ci_upper"]), axis=1
    )
    merged["long_success"] = merged.apply(
        lambda r: replication_success(r["human_effect_size"], r["day30_ci_lower"], r["day30_ci_upper"]), axis=1
    )
    merged["long_fail"] = merged["short_success"] & ~merged["long_success"]
    merged["short_color"] = merged.apply(
        lambda r: TECHNIQUE_PALETTE.get(r["intervention_technique"], "#4C4C4C")
        if r["short_success"] else MISMATCH_COLOR, axis=1
    )
    merged["long_color"] = merged.apply(
        lambda r: TECHNIQUE_PALETTE.get(r["intervention_technique"], "#4C4C4C")
        if r["long_success"] else MISMATCH_COLOR, axis=1
    )
    merged["effect_size"] = merged["human_effect_size"]
    merged["llm_effect_size"] = merged["agent_effect_size"]
    merged["llm_d_day30"] = merged["agent_d_day30"]
    merged = merged.sort_values("human_effect_size", ascending=False).reset_index(drop=True)
    merged["study_label"] = merged.index + 1
    return merged


def split_short_term_success_groups(day30_df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    short = day30_df[day30_df["short_success"]]
    return (short.loc[short["long_success"], "study_id"].tolist(),
            short.loc[short["long_fail"], "study_id"].tolist())


def success_mask_for_effect(df: pd.DataFrame, effect_col: str) -> pd.Series:
    lower, upper = agent_effect_ci(df[effect_col], df["n_control"], df["n_intervention"])
    human = pd.to_numeric(df["human_effect_size"], errors="coerce").to_numpy(float)
    return pd.Series((lower <= human) & (human <= upper), index=df.index)


def ordered_technique_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = df.groupby(["intervention_category", "intervention_technique"], as_index=False).size()
    summary = summary.rename(columns={"size": "n"})
    c_rank = {name: idx for idx, name in enumerate(CATEGORY_ORDER)}
    t_rank = {name: idx for idx, name in enumerate(TECHNIQUE_ORDER)}
    summary["category_rank"] = summary["intervention_category"].map(c_rank).fillna(999)
    summary["technique_rank"] = summary["intervention_technique"].map(t_rank).fillna(999)
    return summary.sort_values(["category_rank", "n", "technique_rank", "intervention_technique"],
                               ascending=[True, False, True, True]).drop(
        columns=["category_rank", "technique_rank"]).reset_index(drop=True)


def format_technique_label(technique: str) -> str:
    return technique.replace("_", " ").title()


def save_figure(fig, stem: str):
    fig.savefig(local_path(f"{stem}.svg"), bbox_inches="tight")
    pdf_dir = local_path("tmp/pdfs")
    pdf_dir.mkdir(parents=True, exist_ok=True)
    # Preserve the exact panel canvas so it maps one-to-one onto the composite grid.
    fig.savefig(pdf_dir / f"{stem}.pdf")
    fig.savefig(local_path(f"{stem}.png"), dpi=300, bbox_inches="tight")



