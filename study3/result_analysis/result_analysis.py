import sys

import numpy as np
import pandas as pd
from scipy import stats
from scipy.integrate import simpson
from scipy.optimize import curve_fit

from study3_helpers import (
    format_technique_label,
    local_path,
    prepare_day30_study_data,
    replication_success,
    split_short_term_success_groups,
)


sys.stdout = open("study3.txt", "w", encoding="utf-8")


def exp_decay(t, asymptote, y0, decay_rate):
    return asymptote + (y0 - asymptote) * np.exp(-decay_rate * t)


def p_text(p_value):
    if pd.isna(p_value):
        return "NA"
    if p_value < 0.0001:
        return "< .0001"
    if p_value < 0.001:
        return "< .001"
    return f"= {p_value:.4f}"


def fit_piecewise_segments(day_values, mean_curve, interval):
    starts = list(range(int(day_values[0]), int(day_values[-1]) + 1, interval))
    rows = []

    for segment_index, start_day in enumerate(starts, start=1):
        mask = (day_values >= start_day) & (day_values < start_day + interval)
        segment_days = day_values[mask]
        segment_curve = mean_curve[mask]

        if segment_days.size < 3 or np.isnan(segment_curve).any():
            continue

        params, _ = curve_fit(
            lambda t, asymptote, y0, decay_rate: asymptote
            + (y0 - asymptote) * np.exp(-decay_rate * (t - t[0])),
            segment_days,
            segment_curve,
            p0=[segment_curve[-1], segment_curve[0], 0.12],
            bounds=([-5.0, -5.0, 0.0], [5.0, 5.0, 5.0]),
            maxfev=20000,
        )
        fitted = params[0] + (params[1] - params[0]) * np.exp(
            -params[2] * (segment_days - segment_days[0])
        )
        auc = simpson(fitted, x=segment_days)
        rows.append(
            {
                "round": segment_index,
                "start_day": int(start_day),
                "MDE": auc / len(segment_days),
                "y0": params[1],
                "A": params[0],
                "PR": params[0] / params[1] if params[1] != 0 else np.nan,
            }
        )

    return pd.DataFrame(rows)


def technique_rate_table(df, outcome_col):
    table = (
        df.groupby("intervention_technique")[outcome_col]
        .agg(outcome_n="sum", total_n="count")
        .reset_index()
    )
    table["rate_%"] = (table["outcome_n"] / table["total_n"] * 100).round(1)
    table["technique_label"] = table["intervention_technique"].map(format_technique_label)
    return table.sort_values(["rate_%", "total_n", "technique_label"], ascending=[False, False, True])


def recovery_by_technique(recovery_mask):
    technique_series = study_df.set_index("study_id").loc[
        recovery_mask.index, "intervention_technique"
    ]
    rows = []
    for technique in sorted(technique_series.unique()):
        mask = technique_series == technique
        total_n = int(mask.sum())
        recovered_n = int(recovery_mask[mask].sum())
        recovered_pct = round(100 * recovered_n / total_n, 1) if total_n > 0 else np.nan
        rows.append(
            {
                "technique": technique,
                "Technique": format_technique_label(technique),
                "N": total_n,
                "Recovered (n)": recovered_n,
                "Recovered (%)": recovered_pct,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["Recovered (%)", "N", "Technique"], ascending=[False, False, True]
    )


study_df = prepare_day30_study_data()
sim = pd.read_csv(local_path("../data/longitudinal-simulation.csv"))

meta_cols = [
    "study_id",
    "intervention_category",
    "intervention_technique",
    "effect_size",
    "ci_lower",
    "ci_upper",
    "llm_effect_size",
    "llm_d_day30",
    "short_success",
    "long_success",
    "long_fail",
]
sim = sim.merge(study_df[meta_cols], on="study_id", how="inner")

d_cols = sorted(
    [column for column in sim.columns if column.startswith("d_round_")],
    key=lambda column: int(column.split("_")[-1]),
)
days = np.array([int(column.split("_")[-1]) for column in d_cols], dtype=int)

single = sim[sim["cycle"] == "single nudge"].copy()
single = single.set_index("study_id").loc[study_df["study_id"]].reset_index()

success_ids, fail_ids = split_short_term_success_groups(study_df)
print("=== Effect-lost set used across Figure 2b-2d ===")
print(fail_ids)

# (A) Overall Day 30 change among included studies
short_h = pd.to_numeric(single["llm_effect_size"], errors="coerce")
day30_h = pd.to_numeric(single["d_round_30"], errors="coerce")
delta_h = day30_h - short_h
mask_pair = short_h.notna() & day30_h.notna()
t_a, p_a = stats.ttest_rel(short_h[mask_pair], day30_h[mask_pair], nan_policy="omit")

a_summary = pd.DataFrame(
    {
        "Included studies (n)": [int(mask_pair.sum())],
        "Short-term mean h": [short_h[mask_pair].mean()],
        "Short-term SD h": [short_h[mask_pair].std(ddof=1)],
        "Day 30 mean h": [day30_h[mask_pair].mean()],
        "Day 30 SD h": [day30_h[mask_pair].std(ddof=1)],
        "Mean delta h": [delta_h[mask_pair].mean()],
        "t (paired)": [t_a],
        "df": [int(mask_pair.sum() - 1)],
        "p": [p_a],
    }
).round(4)

print("\n=== (A) Overall Day 30 change among included studies ===")
print(a_summary.to_string(index=False))
print(
    f"Interpretation: short-term mean h = {short_h[mask_pair].mean():.3f}, "
    f"Day 30 mean h = {day30_h[mask_pair].mean():.3f}, "
    f"mean delta h = {delta_h[mask_pair].mean():+.3f}, "
    f"t({int(mask_pair.sum()-1)}) = {t_a:.3f}, p {p_text(p_a)}"
)

# (B) Day 30 success / failure among short-term successful studies
succ_single = single[single["short_success"]].copy()
succ_single["effect_lost"] = ~succ_single["long_success"]

b_overview = pd.DataFrame(
    {
        "Short-term successful (n)": [int(succ_single.shape[0])],
        "Day 30 successful (n)": [int(succ_single["long_success"].sum())],
        "Effect lost (n)": [int(succ_single["effect_lost"].sum())],
        "Effect lost rate (%)": [
            round(100 * succ_single["effect_lost"].mean(), 1) if not succ_single.empty else np.nan
        ],
    }
)

print("\n=== (B) Day 30 outcome among short-term successful studies ===")
print(b_overview.to_string(index=False))

failure_by_technique = technique_rate_table(succ_single, "effect_lost").rename(
    columns={
        "outcome_n": "effect_lost_n",
        "total_n": "short_term_success_n",
        "rate_%": "effect_lost_rate_%",
    }
)

print("\n=== (B-1) Effect-lost rate by intervention technique ===")
print(failure_by_technique.to_string(index=False))

# (C) Exponential decay fit for the effect-lost set by technique
fit_rows = []
fail_single = single[single["study_id"].isin(fail_ids)].copy()
for technique, group_df in fail_single.groupby("intervention_technique"):
    mean_trajectory = group_df[d_cols].astype(float).mean(axis=0).to_numpy()
    t = np.array([day - 1 for day in days], dtype=float)
    valid = ~np.isnan(mean_trajectory)
    t_fit = t[valid]
    y_fit = mean_trajectory[valid]

    if y_fit.size < 3:
        continue

    initial_y0 = float(group_df["d_round_1"].astype(float).mean())
    initial_asymptote = float(
        group_df[[column for column in d_cols if int(column.split("_")[-1]) >= 25]]
        .astype(float)
        .mean(axis=1)
        .mean()
    )

    params, _ = curve_fit(
        exp_decay,
        t_fit,
        y_fit,
        p0=[initial_asymptote, initial_y0, 0.15],
        bounds=([-5.0, -5.0, 0.0], [5.0, 5.0, 5.0]),
        maxfev=10000,
    )
    fitted = exp_decay(t_fit, *params)
    ss_res = np.sum((y_fit - fitted) ** 2)
    ss_tot = np.sum((y_fit - y_fit.mean()) ** 2)
    r_squared = np.nan if ss_tot == 0 else 1 - ss_res / ss_tot

    fit_rows.append(
        {
            "intervention_technique": technique,
            "Technique": format_technique_label(technique),
            "n": int(len(group_df)),
            "day1_mean_h": float(group_df["d_round_1"].astype(float).mean()),
            "day30_mean_h": float(group_df["d_round_30"].astype(float).mean()),
            "y0_hat": params[1],
            "asymptote_hat": params[0],
            "decay_rate_hat": params[2],
            "R2": r_squared,
        }
    )

decay_fits = pd.DataFrame(fit_rows).round(4)
print("\n=== (C) Exponential decay fits for the effect-lost set ===")
print(decay_fits.to_string(index=False))

# (D) Repetition vs single-shot on the effect-lost set
# The source data uses legacy cycle names "3 repeated nudge" and "5 repeated nudge";
# in the paper and figures we report these schedules by their actual reinforcement intervals.
rep10 = sim[sim["cycle"] == "3 repeated nudge"].copy().set_index("study_id")
rep6 = sim[sim["cycle"] == "5 repeated nudge"].copy().set_index("study_id")
single_fail = fail_single.set_index("study_id")
ci_lower_map = study_df.set_index("study_id")["ci_lower"]
ci_upper_map = study_df.set_index("study_id")["ci_upper"]

ids_common_10 = sorted(set(single_fail.index).intersection(rep10.index))
ids_common_6 = sorted(set(single_fail.index).intersection(rep6.index))

single_10 = single_fail.loc[ids_common_10]
rep10_m = rep10.loc[ids_common_10]
single_6 = single_fail.loc[ids_common_6]
rep6_m = rep6.loc[ids_common_6]

day30_single_10 = pd.to_numeric(single_10["d_round_30"], errors="coerce")
day30_rep10 = pd.to_numeric(rep10_m["d_round_30"], errors="coerce")
mask10 = day30_single_10.notna() & day30_rep10.notna()
t_d30_10, p_d30_10 = stats.ttest_rel(
    day30_rep10[mask10], day30_single_10[mask10], nan_policy="omit"
)

day30_single_6 = pd.to_numeric(single_6["d_round_30"], errors="coerce")
day30_rep6 = pd.to_numeric(rep6_m["d_round_30"], errors="coerce")
mask6 = day30_single_6.notna() & day30_rep6.notna()
t_d30_6, p_d30_6 = stats.ttest_rel(
    day30_rep6[mask6], day30_single_6[mask6], nan_policy="omit"
)

d_summary = pd.DataFrame(
    {
        "Comparison": ["10-day vs single", "6-day vs single"],
        "N paired": [int(mask10.sum()), int(mask6.sum())],
        "Mean Day 30 h (single)": [
            day30_single_10[mask10].mean(),
            day30_single_6[mask6].mean(),
        ],
        "Mean Day 30 h (repeated)": [
            day30_rep10[mask10].mean(),
            day30_rep6[mask6].mean(),
        ],
        "Delta Day 30 h": [
            day30_rep10[mask10].mean() - day30_single_10[mask10].mean(),
            day30_rep6[mask6].mean() - day30_single_6[mask6].mean(),
        ],
        "t": [t_d30_10, t_d30_6],
        "df": [int(mask10.sum() - 1), int(mask6.sum() - 1)],
        "p": [p_d30_10, p_d30_6],
    }
).round(4)

print("\n=== (D) Day 30 improvement with repeated nudges on the effect-lost set ===")
print(d_summary.to_string(index=False))


def recovery_table(repeated_df):
    recovered = repeated_df.apply(
        lambda row: replication_success(
            pd.to_numeric(row["d_round_30"], errors="coerce"),
            ci_lower_map.loc[row.name],
            ci_upper_map.loc[row.name],
        ),
        axis=1,
    )
    total_n = int(recovered.shape[0])
    recovered_n = int(recovered.sum())
    recovered_pct = round(100 * recovered.mean(), 1) if total_n > 0 else np.nan
    return recovered, total_n, recovered_n, recovered_pct


rec10, n10, r10, r10_pct = recovery_table(rep10_m)
rec6, n6, r6, r6_pct = recovery_table(rep6_m)

rec_overall = pd.DataFrame(
    [
        {"Cycle": "10-day", "N": n10, "Recovered (n)": r10, "Recovered (%)": r10_pct},
        {"Cycle": "6-day", "N": n6, "Recovered (n)": r6, "Recovered (%)": r6_pct},
    ]
)

print("\n=== (D-1) Recovery at Day 30 among prior failures ===")
print(rec_overall.to_string(index=False))


print("\n=== (D-2) Recovery by technique (10-day) ===")
print(recovery_by_technique(rec10).to_string(index=False))
print("\n=== (D-3) Recovery by technique (6-day) ===")
print(recovery_by_technique(rec6).to_string(index=False))

# (E) Piecewise schedule summaries for Figure 2d
schedule_configs = [
    {"label": "30-day", "csv_name": "../data/summary_all_p_d_30_freq.csv", "interval": 30},
    {"label": "10-day", "csv_name": "../data/summary_all_p_d_10_freq.csv", "interval": 10},
    {"label": "6-day", "csv_name": "../data/summary_all_p_d_6_freq.csv", "interval": 6},
]

piecewise_rows = []
segment_rows = []
for config in schedule_configs:
    df_schedule = pd.read_csv(local_path(config["csv_name"])).rename(
        columns={"Unnamed: 0": "study_id"}
    )
    df_schedule = df_schedule[df_schedule["study_id"].isin(fail_ids)].copy()
    schedule_d_cols = sorted(
        [column for column in df_schedule.columns if column.startswith("d_round_")],
        key=lambda column: int(column.split("_")[-1]),
    )
    schedule_days = np.array([int(column.split("_")[-1]) for column in schedule_d_cols], dtype=int)
    mean_curve = df_schedule[schedule_d_cols].to_numpy(dtype=float).mean(axis=0)
    segment_df = fit_piecewise_segments(schedule_days, mean_curve, config["interval"])
    segment_df["schedule"] = config["label"]
    segment_rows.append(segment_df)

    row = {
        "schedule": config["label"],
        "n_studies": int(len(df_schedule)),
        "n_segments": int(len(segment_df)),
        "day1_mean_h": float(mean_curve[0]),
        "day30_mean_h": float(mean_curve[-1]),
        "mean_MDE": float(segment_df["MDE"].mean()),
        "mean_PR": float(segment_df["PR"].mean()),
        "mean_y0": float(segment_df["y0"].mean()),
        "mean_A": float(segment_df["A"].mean()),
    }
    if not segment_df.empty:
        row.update(
            {
                "first_y0": float(segment_df.iloc[0]["y0"]),
                "last_y0": float(segment_df.iloc[-1]["y0"]),
                "first_MDE": float(segment_df.iloc[0]["MDE"]),
                "last_MDE": float(segment_df.iloc[-1]["MDE"]),
                "first_PR": float(segment_df.iloc[0]["PR"]),
                "last_PR": float(segment_df.iloc[-1]["PR"]),
            }
        )
    piecewise_rows.append(row)

piecewise_summary = pd.DataFrame(piecewise_rows).round(4)
piecewise_segments = pd.concat(segment_rows, ignore_index=True).round(4)

print("\n=== (E) Piecewise schedule summaries for Figure 2d ===")
print(piecewise_summary.to_string(index=False))
print("\n=== (E-1) Segment-level schedule parameters ===")
print(piecewise_segments.to_string(index=False))
