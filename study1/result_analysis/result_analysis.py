import pandas as pd
import numpy as np
from scipy import stats
from collections import OrderedDict


def read_csv_robust(path):
    for enc in ["utf-8-sig", "utf-8", "gbk", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


df = read_csv_robust("../data/nudge-replication.csv")

# 95% two-sided confidence interval
def success_mask(df, model_prefix):
    d_col = f"{model_prefix}_llm_effect_size"
    p_col = f"{model_prefix}_llm_p_value"

    d = pd.to_numeric(df[d_col], errors="coerce")
    p = pd.to_numeric(df[p_col], errors="coerce")
    ci_lower = pd.to_numeric(df["ci_lower"], errors="coerce")
    ci_upper = pd.to_numeric(df["ci_upper"], errors="coerce")

    success = (d >= ci_lower) & (d <= ci_upper)

    return success, d, p


def add(text=""):
    output_lines.append(str(text))


def safe_mean(series):
    series = pd.to_numeric(series, errors="coerce")
    return series.mean() if series.notna().any() else np.nan


def safe_sd(series):
    series = pd.to_numeric(series, errors="coerce")
    return series.std() if series.notna().sum() >= 2 else np.nan


def safe_min(series):
    series = pd.to_numeric(series, errors="coerce")
    return series.min() if series.notna().any() else np.nan


def safe_max(series):
    series = pd.to_numeric(series, errors="coerce")
    return series.max() if series.notna().any() else np.nan


def weighted_mean(values, variances):
    values = pd.to_numeric(values, errors="coerce")
    variances = pd.to_numeric(variances, errors="coerce")
    mask = values.notna() & variances.notna() & (variances > 0)
    if not mask.any():
        return np.nan
    weights = 1 / variances[mask]
    return float(np.sum(weights * values[mask]) / np.sum(weights))


def mad(values):
    values = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if values.empty:
        return np.nan
    return float(np.median(np.abs(values - np.median(values))))


def chi_square_summary(ct):
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return np.nan, np.nan, np.nan, np.nan
    chi2, p, dof, expected = stats.chi2_contingency(ct)
    min_expected = expected.min() if expected.size else np.nan
    return chi2, p, dof, min_expected


def technique_summary(base_df, success_col=None, llm_d_col=None):
    rows = []
    cols = ["intervention_category", "intervention_technique", "effect_size"]
    if success_col is not None:
        cols.append(success_col)
    if llm_d_col is not None:
        cols.append(llm_d_col)

    tmp = base_df[cols].copy()
    tmp["intervention_category"] = tmp["intervention_category"].fillna("missing")
    tmp["intervention_technique"] = tmp["intervention_technique"].fillna("missing")

    for (category, technique), g in tmp.groupby(
        ["intervention_category", "intervention_technique"], dropna=False
    ):
        human_d = pd.to_numeric(g["effect_size"], errors="coerce")
        row = {
            "Category": category,
            "Technique": technique,
            "N": int(len(g)),
            "Mean human effect size": round(safe_mean(human_d), 3),
            "SD human effect size": round(safe_sd(human_d), 3),
            "Min human effect size": round(safe_min(human_d), 3),
            "Max human effect size": round(safe_max(human_d), 3),
        }

        if success_col is not None:
            success = g[success_col].astype(bool)
            success_n = int(success.sum())
            row["Success (n)"] = success_n
            row["Success rate (%)"] = round(100 * success_n / len(g), 1)

        if llm_d_col is not None:
            llm_d = pd.to_numeric(g[llm_d_col], errors="coerce")
            mask = human_d.notna() & llm_d.notna()
            row["Mean LLM effect size"] = round(safe_mean(llm_d), 3)
            row["Mean signed effect-size gap"] = (
                round((llm_d[mask] - human_d[mask]).mean(), 3)
                if mask.any()
                else np.nan
            )
            row["Mean absolute effect-size gap"] = (
                round((llm_d[mask] - human_d[mask]).abs().mean(), 3)
                if mask.any()
                else np.nan
            )

        rows.append(row)

    out = pd.DataFrame(rows)
    out = out.sort_values(["N", "Category", "Technique"], ascending=[False, True, True])
    return out


def category_technique_counts(base_df):
    rows = []
    tmp = base_df[["intervention_category", "intervention_technique"]].copy()
    tmp["intervention_category"] = tmp["intervention_category"].fillna("missing")
    tmp["intervention_technique"] = tmp["intervention_technique"].fillna("missing")

    for (category, technique), g in tmp.groupby(
        ["intervention_category", "intervention_technique"], dropna=False
    ):
        rows.append(
            {
                "Category": category,
                "Technique": technique,
                "N": int(len(g)),
            }
        )

    out = pd.DataFrame(rows)
    out = out.sort_values(["Category", "N", "Technique"], ascending=[True, False, True])
    return out


def llm_technique_heterogeneity(base_df, model_map):
    rows = []
    for mkey, mname in model_map.items():
        d_col = f"{mkey}_llm_effect_size"
        tmp = base_df[["intervention_category", "intervention_technique", d_col]].copy()
        tmp[d_col] = pd.to_numeric(tmp[d_col], errors="coerce")
        tmp = tmp.dropna(subset=[d_col])
        for (category, technique), g in tmp.groupby(
            ["intervention_category", "intervention_technique"], dropna=False
        ):
            rows.append(
                {
                    "Model": mname,
                    "Category": category,
                    "Technique": technique,
                    "N": int(len(g)),
                    "Mean LLM effect size": round(g[d_col].mean(), 3),
                    "SD LLM effect size": round(g[d_col].std(), 3) if len(g) >= 2 else np.nan,
                    "Min LLM effect size": round(g[d_col].min(), 3),
                    "Max LLM effect size": round(g[d_col].max(), 3),
                    "Range LLM effect size": round(g[d_col].max() - g[d_col].min(), 3),
                }
            )
    out = pd.DataFrame(rows)
    out = out.sort_values(["Category", "Technique", "Model"])
    return out


def pooled_llm_technique_heterogeneity(base_df, model_map):
    rows = []
    for _, row in base_df.iterrows():
        for mkey, mname in model_map.items():
            value = pd.to_numeric(row.get(f"{mkey}_llm_effect_size"), errors="coerce")
            if pd.notna(value):
                rows.append(
                    {
                        "Category": row["intervention_category"],
                        "Technique": row["intervention_technique"],
                        "Model": mname,
                        "LLM effect size": float(value),
                    }
                )
    long_df = pd.DataFrame(rows)
    out = (
        long_df.groupby(["Category", "Technique"], dropna=False)
        .agg(
            N_estimates=("LLM effect size", "count"),
            Mean_LLM_effect_size=("LLM effect size", "mean"),
            SD_LLM_effect_size=("LLM effect size", "std"),
            Min_LLM_effect_size=("LLM effect size", "min"),
            Max_LLM_effect_size=("LLM effect size", "max"),
        )
        .reset_index()
    )
    out["Range_LLM_effect_size"] = out["Max_LLM_effect_size"] - out["Min_LLM_effect_size"]
    for col in [
        "Mean_LLM_effect_size",
        "SD_LLM_effect_size",
        "Min_LLM_effect_size",
        "Max_LLM_effect_size",
        "Range_LLM_effect_size",
    ]:
        out[col] = out[col].round(3)
    out = out.sort_values(["Category", "Technique"])
    return out


def figure_technique_distribution_summary(base_df, model_map):
    chatting_model_cols = [
        "gpt35_llm_effect_size",
        "gpt4o_llm_effect_size",
        "claudehaiku45_llm_effect_size",
        "deepseekv3_llm_effect_size",
    ]
    reasoning_model_cols = [
        "deepseekv4flash_llm_effect_size",
        "gpt54_llm_effect_size",
    ]
    rows = []
    for _, row in base_df.iterrows():
        human_value = pd.to_numeric(row.get("effect_size"), errors="coerce")
        if pd.notna(human_value):
            rows.append(
                {
                    "Source": "Human experiments",
                    "Category": row["intervention_category"],
                    "Technique": row["intervention_technique"],
                    "Effect size": float(human_value),
                }
            )

        llm_groups = {
            "Non-reasoning LLMs": pd.to_numeric(
                row[chatting_model_cols], errors="coerce"
            ).mean(),
            "Reasoning LLMs": pd.to_numeric(
                row[reasoning_model_cols], errors="coerce"
            ).mean(),
        }
        for source, llm_value in llm_groups.items():
            if pd.notna(llm_value):
                rows.append(
                    {
                        "Source": source,
                        "Category": row["intervention_category"],
                        "Technique": row["intervention_technique"],
                        "Effect size": float(llm_value),
                    }
                )

    long_df = pd.DataFrame(rows)
    out = (
        long_df.groupby(["Source", "Category", "Technique"], dropna=False)
        .agg(
            N=("Effect size", "count"),
            Mean_effect_size=("Effect size", "mean"),
            Median_effect_size=("Effect size", "median"),
            SD_effect_size=("Effect size", "std"),
            Min_effect_size=("Effect size", "min"),
            Max_effect_size=("Effect size", "max"),
        )
        .reset_index()
    )
    for col in [
        "Mean_effect_size",
        "Median_effect_size",
        "SD_effect_size",
        "Min_effect_size",
        "Max_effect_size",
    ]:
        out[col] = out[col].round(3)
    out["Source"] = pd.Categorical(
        out["Source"],
        categories=["Human experiments", "Non-reasoning LLMs", "Reasoning LLMs"],
        ordered=True,
    )
    out = out.sort_values(["Source", "Category", "Technique"])
    return out


def figure_between_technique_distribution_summary(fig_dist_df):
    rows = []
    for source, g in fig_dist_df.groupby("Source", observed=False):
        means = pd.to_numeric(g["Mean_effect_size"], errors="coerce").dropna()
        medians = pd.to_numeric(g["Median_effect_size"], errors="coerce").dropna()
        rows.append(
            {
                "Source": source,
                "N techniques": int(len(g)),
                "Mean-of-means": round(means.mean(), 3),
                "SD of technique means": round(means.std(), 3),
                "IQR of technique means": round(
                    means.quantile(0.75) - means.quantile(0.25), 3
                ),
                "MAD of technique means": round(mad(means), 3),
                "IQR of technique medians": round(
                    medians.quantile(0.75) - medians.quantile(0.25), 3
                ),
            }
        )
    return pd.DataFrame(rows)


def model_type_effect_summary(base_df):
    group_cols = OrderedDict(
        [
            (
                "Non-reasoning LLMs",
                [
                    "gpt35_llm_effect_size",
                    "gpt4o_llm_effect_size",
                    "claudehaiku45_llm_effect_size",
                    "deepseekv3_llm_effect_size",
                ],
            ),
            (
                "Reasoning LLMs",
                [
                    "deepseekv4flash_llm_effect_size",
                    "gpt54_llm_effect_size",
                ],
            ),
        ]
    )

    d_h = pd.to_numeric(base_df["effect_size"], errors="coerce")
    rows = []
    for group, cols in group_cols.items():
        d_group = base_df[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        mask = d_h.notna() & d_group.notna()
        r, p_corr = stats.pearsonr(d_h[mask], d_group[mask])
        t, p_t = stats.ttest_rel(d_group[mask], d_h[mask], nan_policy="omit")
        rows.append(
            {
                "Model group": group,
                "N": int(mask.sum()),
                "Mean simulated effect size": round(d_group[mask].mean(), 3),
                "Mean human effect size": round(d_h[mask].mean(), 3),
                "Pearson r": round(r, 3),
                "Pearson p": p_corr,
                "Mean difference (group - human)": round((d_group[mask] - d_h[mask]).mean(), 3),
                "Paired t": round(t, 3),
                "Paired p": p_t,
            }
        )
    return pd.DataFrame(rows)


def model_type_success_comparison(base_df):
    success_df = pd.DataFrame(index=base_df.index)
    for mkey in models:
        success_df[mkey] = success_mask(base_df, mkey)[0].astype(float)

    non_reasoning_rate = success_df[
        ["gpt35", "gpt4o", "claudehaiku45", "deepseekv3"]
    ].mean(axis=1)
    reasoning_rate = success_df[["deepseekv4flash", "gpt54"]].mean(axis=1)
    diff = non_reasoning_rate - reasoning_rate

    t, p_t = stats.ttest_rel(non_reasoning_rate, reasoning_rate, nan_policy="omit")
    try:
        w, p_w = stats.wilcoxon(non_reasoning_rate, reasoning_rate, zero_method="wilcox")
    except ValueError:
        w, p_w = np.nan, np.nan

    return pd.DataFrame(
        [
            {
                "N studies": int(diff.notna().sum()),
                "Mean non-reasoning success": round(non_reasoning_rate.mean(), 3),
                "Mean reasoning success": round(reasoning_rate.mean(), 3),
                "Mean paired difference": round(diff.mean(), 3),
                "Paired t": round(t, 3),
                "Paired p": p_t,
                "Wilcoxon W": w,
                "Wilcoxon p": p_w,
            }
        ]
    )


def technique_heterogeneity_tests(fig_long_df):
    rows = []
    for source, g in fig_long_df.groupby("Source", observed=False):
        groups = [
            vals["Effect size"].dropna().to_numpy()
            for _, vals in g.groupby("Technique", observed=False)
            if vals["Effect size"].notna().sum() > 0
        ]
        if len(groups) >= 2:
            h, p = stats.kruskal(*groups)
        else:
            h, p = np.nan, np.nan
        rows.append(
            {
                "Source": source,
                "N techniques": len(groups),
                "Kruskal H": round(h, 3) if pd.notna(h) else np.nan,
                "p": p,
            }
        )
    return pd.DataFrame(rows)


def model_type_technique_correlations(fig_dist_df):
    means = fig_dist_df[["Source", "Technique", "Mean_effect_size"]].copy()
    human = means[means["Source"] == "Human experiments"][
        ["Technique", "Mean_effect_size"]
    ].rename(columns={"Mean_effect_size": "Human mean effect size"})
    rows = []
    for source in ["Non-reasoning LLMs", "Reasoning LLMs"]:
        model = means[means["Source"] == source][["Technique", "Mean_effect_size"]].rename(
            columns={"Mean_effect_size": "Model-group mean effect size"}
        )
        merged = human.merge(model, on="Technique", how="inner").dropna()
        r, p = stats.pearsonr(
            merged["Human mean effect size"], merged["Model-group mean effect size"]
        )
        rho, sp = stats.spearmanr(
            merged["Human mean effect size"], merged["Model-group mean effect size"]
        )
        rows.append(
            {
                "Model group": source,
                "N techniques": int(len(merged)),
                "Pearson r": round(r, 3),
                "Pearson p": p,
                "Spearman rho": round(rho, 3),
                "Spearman p": sp,
            }
        )
    return pd.DataFrame(rows)


def figure_distribution_long(base_df):
    chatting_model_cols = [
        "gpt35_llm_effect_size",
        "gpt4o_llm_effect_size",
        "claudehaiku45_llm_effect_size",
        "deepseekv3_llm_effect_size",
    ]
    reasoning_model_cols = [
        "deepseekv4flash_llm_effect_size",
        "gpt54_llm_effect_size",
    ]
    rows = []
    for _, row in base_df.iterrows():
        values = {
            "Human experiments": pd.to_numeric(row.get("effect_size"), errors="coerce"),
            "Non-reasoning LLMs": pd.to_numeric(
                row[chatting_model_cols], errors="coerce"
            ).mean(),
            "Reasoning LLMs": pd.to_numeric(
                row[reasoning_model_cols], errors="coerce"
            ).mean(),
        }
        for source, value in values.items():
            if pd.notna(value):
                rows.append(
                    {
                        "Source": source,
                        "Category": row["intervention_category"],
                        "Technique": row["intervention_technique"],
                        "Effect size": float(value),
                    }
                )
    return pd.DataFrame(rows)


def figure_distribution_long_with_variance(base_df):
    non_reasoning_effect_cols = [
        "gpt35_llm_effect_size",
        "gpt4o_llm_effect_size",
        "claudehaiku45_llm_effect_size",
        "deepseekv3_llm_effect_size",
    ]
    non_reasoning_variance_cols = [
        "gpt35_llm_variance_d",
        "gpt4o_llm_variance_d",
        "claudehaiku45_llm_variance_d",
        "deepseekv3_llm_variance_d",
    ]
    reasoning_effect_cols = [
        "deepseekv4flash_llm_effect_size",
        "gpt54_llm_effect_size",
    ]
    reasoning_variance_cols = [
        "deepseekv4flash_llm_variance_d",
        "gpt54_llm_variance_d",
    ]

    rows = []
    for _, row in base_df.iterrows():
        human_effect = pd.to_numeric(row.get("effect_size"), errors="coerce")
        human_var = pd.to_numeric(row.get("variance_d"), errors="coerce")
        if pd.notna(human_effect) and pd.notna(human_var):
            rows.append(
                {
                    "Source": "Human experiments",
                    "Category": row["intervention_category"],
                    "Technique": row["intervention_technique"],
                    "Effect size": float(human_effect),
                    "Variance": float(human_var),
                }
            )

        llm_groups = {
            "Non-reasoning LLMs": (non_reasoning_effect_cols, non_reasoning_variance_cols),
            "Reasoning LLMs": (reasoning_effect_cols, reasoning_variance_cols),
        }
        for source, (effect_cols, variance_cols) in llm_groups.items():
            effects = pd.to_numeric(row[effect_cols], errors="coerce").to_numpy()
            variances = pd.to_numeric(row[variance_cols], errors="coerce").to_numpy()
            mask = pd.notna(effects) & pd.notna(variances) & (variances > 0)
            if mask.any():
                group_effect = effects[mask].mean()
                # Variance of an average, assuming model estimates are independent.
                group_variance = variances[mask].sum() / (mask.sum() ** 2)
                rows.append(
                    {
                        "Source": source,
                        "Category": row["intervention_category"],
                        "Technique": row["intervention_technique"],
                        "Effect size": float(group_effect),
                        "Variance": float(group_variance),
                    }
                )
    return pd.DataFrame(rows)


def weighted_technique_heterogeneity_summary(long_df):
    technique_rows = []
    summary_rows = []
    for source, source_df in long_df.groupby("Source", dropna=False):
        technique_means = []
        technique_weights = []
        for (category, technique), g in source_df.groupby(
            ["Category", "Technique"], dropna=False
        ):
            values = pd.to_numeric(g["Effect size"], errors="coerce")
            variances = pd.to_numeric(g["Variance"], errors="coerce")
            mean_effect = weighted_mean(values, variances)
            valid_var = variances.dropna()
            total_weight = float((1 / valid_var[valid_var > 0]).sum())
            technique_rows.append(
                {
                    "Source": source,
                    "Category": category,
                    "Technique": technique,
                    "N": int(values.notna().sum()),
                    "Weighted mean effect size": round(mean_effect, 3),
                    "Total inverse-variance weight": round(total_weight, 3),
                }
            )
            if pd.notna(mean_effect) and total_weight > 0:
                technique_means.append(mean_effect)
                technique_weights.append(total_weight)

        means = pd.Series(technique_means, dtype=float)
        weights = pd.Series(technique_weights, dtype=float)
        if len(means) >= 2:
            weighted_overall = float(np.sum(weights * means) / np.sum(weights))
            ss_total = float(np.sum(weights * (means - weighted_overall) ** 2))
            # With one row per technique mean, the model sum of squares equals total
            # dispersion among technique means; eta^2 is reported against study-level
            # weighted dispersion below.
            iqr = float(means.quantile(0.75) - means.quantile(0.25))
            summary_rows.append(
                {
                    "Source": source,
                    "N techniques": int(len(means)),
                    "Weighted mean across techniques": round(weighted_overall, 3),
                    "SD of weighted technique means": round(means.std(), 3),
                    "IQR of weighted technique means": round(iqr, 3),
                    "MAD of weighted technique means": round(mad(means), 3),
                    "Weighted dispersion among technique means": round(ss_total, 3),
                }
            )

    return pd.DataFrame(technique_rows), pd.DataFrame(summary_rows)


def weighted_variance_explained_by_technique(long_df):
    rows = []
    for source, source_df in long_df.groupby("Source", dropna=False):
        values = pd.to_numeric(source_df["Effect size"], errors="coerce")
        variances = pd.to_numeric(source_df["Variance"], errors="coerce")
        mask = values.notna() & variances.notna() & (variances > 0)
        tmp = source_df.loc[mask].copy()
        tmp["Effect size"] = values[mask]
        tmp["Weight"] = 1 / variances[mask]
        if tmp.empty:
            continue
        overall = float(np.sum(tmp["Weight"] * tmp["Effect size"]) / np.sum(tmp["Weight"]))
        ss_total = float(np.sum(tmp["Weight"] * (tmp["Effect size"] - overall) ** 2))
        ss_between = 0.0
        for _, g in tmp.groupby("Technique", dropna=False):
            group_mean = float(np.sum(g["Weight"] * g["Effect size"]) / np.sum(g["Weight"]))
            ss_between += float(np.sum(g["Weight"]) * (group_mean - overall) ** 2)
        eta2 = ss_between / ss_total if ss_total > 0 else np.nan
        rows.append(
            {
                "Source": source,
                "N observations": int(len(tmp)),
                "N techniques": int(tmp["Technique"].nunique(dropna=False)),
                "Weighted eta^2 for technique": round(eta2, 3) if pd.notna(eta2) else np.nan,
                "Weighted between-technique SS": round(ss_between, 3),
                "Weighted total SS": round(ss_total, 3),
            }
        )
    return pd.DataFrame(rows)


def technique_mean_by_source(base_df, model_map):
    rows = []
    human_tmp = base_df[["intervention_category", "intervention_technique", "effect_size"]].copy()
    human_tmp["effect_size"] = pd.to_numeric(human_tmp["effect_size"], errors="coerce")
    for (category, technique), g in human_tmp.groupby(
        ["intervention_category", "intervention_technique"], dropna=False
    ):
        rows.append(
            {
                "Source": "Human",
                "Category": category,
                "Technique": technique,
                "N": int(g["effect_size"].notna().sum()),
                "Mean effect size": round(g["effect_size"].mean(), 3),
            }
        )

    for mkey, mname in model_map.items():
        d_col = f"{mkey}_llm_effect_size"
        tmp = base_df[["intervention_category", "intervention_technique", d_col]].copy()
        tmp[d_col] = pd.to_numeric(tmp[d_col], errors="coerce")
        for (category, technique), g in tmp.groupby(
            ["intervention_category", "intervention_technique"], dropna=False
        ):
            rows.append(
                {
                    "Source": mname,
                    "Category": category,
                    "Technique": technique,
                    "N": int(g[d_col].notna().sum()),
                    "Mean effect size": round(g[d_col].mean(), 3),
                }
            )
    out = pd.DataFrame(rows)
    out = out.sort_values(["Source", "Category", "Technique"])
    return out


def between_technique_heterogeneity(tech_mean_df):
    rows = []
    for source, g in tech_mean_df.groupby("Source", dropna=False):
        values = pd.to_numeric(g["Mean effect size"], errors="coerce").dropna()
        rows.append(
            {
                "Source": source,
                "N techniques": int(values.shape[0]),
                "Mean across techniques": round(values.mean(), 3),
                "SD across techniques": round(values.std(), 3),
                "Min technique mean": round(values.min(), 3),
                "Max technique mean": round(values.max(), 3),
                "Range across techniques": round(values.max() - values.min(), 3),
            }
        )
    out = pd.DataFrame(rows)
    source_order = ["Human"] + list(models.values()) if "models" in globals() else None
    if source_order is not None:
        out["Source"] = pd.Categorical(out["Source"], categories=source_order, ordered=True)
        out = out.sort_values("Source")
    return out


def model_human_technique_correlations(tech_mean_df, model_map):
    human = tech_mean_df[tech_mean_df["Source"] == "Human"][
        ["Technique", "Mean effect size"]
    ].rename(columns={"Mean effect size": "Human mean effect size"})
    rows = []
    for mname in model_map.values():
        model = tech_mean_df[tech_mean_df["Source"] == mname][
            ["Technique", "Mean effect size"]
        ].rename(columns={"Mean effect size": "Model mean effect size"})
        merged = human.merge(model, on="Technique", how="inner").dropna()
        if len(merged) >= 3:
            r, p = stats.pearsonr(
                merged["Human mean effect size"], merged["Model mean effect size"]
            )
            rho, sp = stats.spearmanr(
                merged["Human mean effect size"], merged["Model mean effect size"]
            )
        else:
            r, p, rho, sp = np.nan, np.nan, np.nan, np.nan
        rows.append(
            {
                "Model": mname,
                "N techniques": int(len(merged)),
                "Pearson r": round(r, 3) if pd.notna(r) else np.nan,
                "Pearson p": p,
                "Spearman rho": round(rho, 3) if pd.notna(rho) else np.nan,
                "Spearman p": sp,
            }
        )
    return pd.DataFrame(rows)


models = OrderedDict(
    [
        ("gpt35", "GPT-3.5"),
        ("gpt4o", "GPT-4o"),
        ("claudehaiku45", "Claude-4.5-Haiku"),
        ("deepseekv3", "DeepSeek-V3"),
        ("deepseekv4flash", "DeepSeek-V4-Flash-Think"),
        ("gpt54", "GPT-5.4-Think"),
    ]
)

model_groups = {
    "GPT-3.5": "Non-reasoning",
    "GPT-4o": "Non-reasoning",
    "Claude-4.5-Haiku": "Non-reasoning",
    "DeepSeek-V3": "Non-reasoning",
    "DeepSeek-V4-Flash-Think": "Reasoning",
    "GPT-5.4-Think": "Reasoning",
}

output_lines = []

# 1. Success summary
success_summary = []
for mkey, mname in models.items():
    succ, d, p = success_mask(df, mkey)
    n_total = succ.notna().sum()
    n_succ = succ.sum()
    rate = 100 * n_succ / n_total if n_total > 0 else np.nan
    success_summary.append(
        {
            "Model": mname,
            "N (evaluable)": int(n_total),
            "Success (n)": int(n_succ),
            "Success rate (%)": round(rate, 1),
        }
    )

success_df = pd.DataFrame(success_summary)
add("=== Overall Success Summary ===")
add(success_df.to_string(index=False))
add()

add("=== Success / Failure Study IDs by Model ===")

for mkey, mname in models.items():
    succ, d, p = success_mask(df, mkey)

    # Only include evaluable studies
    valid = succ.notna()

    success_ids = df.loc[valid & succ, "study_id"].tolist()
    failure_ids = df.loc[valid & ~succ, "study_id"].tolist()

    add(f"--- {mname} ---")
    add(f"Success ({len(success_ids)}):")
    add(", ".join(map(str, success_ids)))

    add(f"Failure ({len(failure_ids)}):")
    add(", ".join(map(str, failure_ids)))
    add()

group_success_df = success_df.copy()
group_success_df["Model group"] = group_success_df["Model"].map(model_groups)
group_success_df = (
    group_success_df.groupby("Model group", dropna=False)[
        ["Success rate (%)", "Success (n)", "N (evaluable)"]
    ]
    .mean(numeric_only=True)
    .reset_index()
)
add("=== Reasoning vs Non-reasoning Summary ===")
add(group_success_df.to_string(index=False))
add()

model_type_success_df = model_type_success_comparison(df)
add("=== Paired success comparison: non-reasoning vs reasoning model groups ===")
add(
    "For each study, success was averaged within non-reasoning and reasoning model groups, "
    "then compared using paired tests."
)
add(model_type_success_df.to_string(index=False))
add()

# 2. Success rate by effect_label
label_rate_rows = []
for mkey, mname in models.items():
    succ, d, p = success_mask(df, mkey)
    effect_label = pd.to_numeric(df["effect_label"], errors="coerce")

    for label in [-1, 0, 1]:
        mask = effect_label == label
        n_total = mask.sum()
        n_succ = succ[mask].sum()
        rate = 100 * n_succ / n_total if n_total > 0 else np.nan

        label_rate_rows.append(
            {
                "Model": mname,
                "effect_label": label,
                "N": int(n_total),
                "Success (n)": int(n_succ),
                "Success rate (%)": round(rate, 1) if pd.notna(rate) else np.nan,
            }
        )

label_rate_df = pd.DataFrame(label_rate_rows)
add("=== Success Rate by effect_label ===")
add(label_rate_df.to_string(index=False))
add()

label_distribution = (
    pd.to_numeric(df["effect_label"], errors="coerce")
    .value_counts(dropna=False)
    .sort_index()
    .rename_axis("effect_label")
    .reset_index(name="N")
)
add("=== Human effect_label distribution ===")
add(label_distribution.to_string(index=False))
add()

tech_label_df = (
    df.groupby(["intervention_category", "intervention_technique", "effect_label"], dropna=False)
    .size()
    .reset_index(name="N")
    .sort_values(
        ["intervention_category", "intervention_technique", "effect_label"],
        ascending=[True, True, True],
    )
)
add("=== effect_label distribution by intervention_technique ===")
add(
    "This table shows whether broad categories contain mixes of positive, null, and negative effects."
)
add(tech_label_df.to_string(index=False))
add()

group_label_df = label_rate_df.copy()
group_label_df["Model group"] = group_label_df["Model"].map(model_groups)
group_label_df = (
    group_label_df.groupby(["Model group", "effect_label"], dropna=False)[
        ["Success rate (%)", "Success (n)", "N"]
    ]
    .mean(numeric_only=True)
    .reset_index()
)
add("=== Mean Success Rate by effect_label and model group ===")
add(group_label_df.to_string(index=False))
add()

# 3. Pearson correlation with human effect size
cor_rows = []
for mkey, mname in models.items():
    d_h = pd.to_numeric(df["effect_size"], errors="coerce")
    d_m = pd.to_numeric(df[f"{mkey}_llm_effect_size"], errors="coerce")
    mask = d_h.notna() & d_m.notna()
    if mask.sum() >= 3:
        r, p = stats.pearsonr(d_h[mask], d_m[mask])
    else:
        r, p = (np.nan, np.nan)
    cor_rows.append({"Model": mname, "r": r, "p": p, "N": int(mask.sum())})

cor_df = pd.DataFrame(cor_rows)
add("=== Pearson Correlation: LLM effect size vs Human effect size ===")
add(cor_df.to_string(index=False))
add()

# 4. Paired t test
paired_rows = []
for mkey, mname in models.items():
    d_h = pd.to_numeric(df["effect_size"], errors="coerce")
    d_m = pd.to_numeric(df[f"{mkey}_llm_effect_size"], errors="coerce")
    mask = d_h.notna() & d_m.notna()
    if mask.sum() >= 3:
        t, p = stats.ttest_rel(d_m[mask], d_h[mask], nan_policy="omit")
        delta = (d_m[mask] - d_h[mask]).mean()
    else:
        t, p, delta = (np.nan, np.nan, np.nan)
    paired_rows.append(
        {
            "Model": mname,
            "Mean difference (LLM - Human)": delta,
            "t": t,
            "p": p,
            "N": int(mask.sum()),
        }
    )

paired_df = pd.DataFrame(paired_rows)
add("=== Paired t-test: LLM effect size vs Human effect size ===")
add(paired_df.to_string(index=False))
add()

model_type_effect_df = model_type_effect_summary(df)
add("=== Model-type averaged effect correspondence with human effects ===")
add(
    "Model outputs are averaged within each study separately for non-reasoning and reasoning models."
)
add(model_type_effect_df.to_string(index=False))
add()

# 5. Human effect heterogeneity by intervention technique
technique_count_df = category_technique_counts(df)
add("=== intervention_technique composition within broad categories ===")
add("This table lists the technique subcategories represented in each broad category.")
add(technique_count_df.to_string(index=False))
add()

heterogeneity_df = technique_summary(df)
add("=== Human Effect Heterogeneity by intervention_technique ===")
add("Note: this section is descriptive and meant to show within-category heterogeneity.")
add(heterogeneity_df.to_string(index=False))
add()

# 6. Success rate by intervention_technique for each model
tech_rate_rows = []
for mkey, mname in models.items():
    succ, d, p = success_mask(df, mkey)
    tmp = df[["intervention_category", "intervention_technique"]].copy()
    tmp["success"] = succ.values
    tmp["intervention_category"] = tmp["intervention_category"].fillna("missing")
    tmp["intervention_technique"] = tmp["intervention_technique"].fillna("missing")

    for (category, technique), g in tmp.groupby(
        ["intervention_category", "intervention_technique"], dropna=False
    ):
        n_total = int(len(g))
        n_succ = int(g["success"].sum())
        tech_rate_rows.append(
            {
                "Model": mname,
                "Category": category,
                "Technique": technique,
                "N": n_total,
                "Success (n)": n_succ,
                "Success rate (%)": round(100 * n_succ / n_total, 1),
            }
        )

tech_rate_df = pd.DataFrame(tech_rate_rows)
tech_rate_df = tech_rate_df.sort_values(
    ["Category", "Technique", "Model"], ascending=[True, True, True]
)
add("=== Success Rate by intervention_technique ===")
add("Note: techniques with very small N should be interpreted cautiously.")
add(tech_rate_df.to_string(index=False))
add()

# 6b. Figure 1d: between-technique heterogeneity in humans and simulations
fig_dist_df = figure_technique_distribution_summary(df, models)
add("=== Figure 1d distribution summary: human experiments vs LLM model-type averages ===")
add(
    "This table matches the boxplot in Fig. 1d: human effect sizes, non-reasoning-model "
    "mean simulated effect sizes, and reasoning-model mean simulated effect sizes "
    "are summarized within each intervention technique."
)
add(fig_dist_df.to_string(index=False))
add()

fig_between_dist_df = figure_between_technique_distribution_summary(fig_dist_df)
add("=== Figure 1d distribution summary: heterogeneity across intervention techniques ===")
add(
    "These statistics summarize how strongly the technique-level distributions differ "
    "within human experiments, non-reasoning-model simulations, and reasoning-model simulations using robust dispersion indices."
)
add(fig_between_dist_df.to_string(index=False))
add()

fig_long_df = figure_distribution_long(df)
fig_heterogeneity_tests = technique_heterogeneity_tests(fig_long_df)
add("=== Figure 1d Kruskal-Wallis tests across intervention techniques ===")
add(
    "Kruskal-Wallis tests ask whether effect-size distributions differ across the nine intervention techniques."
)
add(fig_heterogeneity_tests.to_string(index=False))
add()

weighted_fig_long_df = figure_distribution_long_with_variance(df)
weighted_technique_df, weighted_dispersion_df = weighted_technique_heterogeneity_summary(
    weighted_fig_long_df
)
add("=== Figure 1d inverse-variance weighted technique means ===")
add(
    "Effect sizes are summarized within each intervention technique using inverse-variance weights."
)
add(weighted_technique_df.to_string(index=False))
add()

add("=== Figure 1d inverse-variance weighted heterogeneity summary ===")
add(
    "Heterogeneity across techniques is summarized with SD, IQR, and MAD of weighted technique means, rather than simple ranges."
)
add(weighted_dispersion_df.to_string(index=False))
add()

weighted_eta_df = weighted_variance_explained_by_technique(weighted_fig_long_df)
add("=== Figure 1d weighted variance explained by intervention technique ===")
add(
    "Weighted eta^2 estimates the proportion of study-level weighted dispersion attributable to intervention technique."
)
add(weighted_eta_df.to_string(index=False))
add()

fig_model_type_corr_df = model_type_technique_correlations(fig_dist_df)
add("=== Figure 1d model-type correspondence with human technique-level heterogeneity ===")
add(
    "Correlations compare the nine human technique-level means with the corresponding model-type means."
)
add(fig_model_type_corr_df.to_string(index=False))
add()

tech_mean_source_df = technique_mean_by_source(df, models)
add("=== Figure 1d summary: technique-level mean effects by source ===")
add(
    "This table compares human and simulated mean effect sizes across the same nine intervention techniques."
)
add(tech_mean_source_df.to_string(index=False))
add()

tech_corr_df = model_human_technique_correlations(tech_mean_source_df, models)
add("=== Figure 1d summary: model-human correspondence in technique-level heterogeneity ===")
add(
    "Correlations compare each model's nine technique-level means with the human technique-level means."
)
add(tech_corr_df.to_string(index=False))
add()

# 7. DeepSeek-V4-Flash-Think subgroup analysis
succ_mask_deepseekv4flash, d_deepseekv4flash, p_deepseekv4flash = success_mask(df, "deepseekv4flash")
valid = succ_mask_deepseekv4flash.notna()
df_valid = df.loc[valid].copy()
df_valid["deepseekv4flash_success"] = succ_mask_deepseekv4flash[valid].values

n_succ = pd.to_numeric(
    df_valid.loc[df_valid["deepseekv4flash_success"], "n_comparison"], errors="coerce"
).dropna()
n_fail = pd.to_numeric(
    df_valid.loc[~df_valid["deepseekv4flash_success"], "n_comparison"], errors="coerce"
).dropna()

t_stat, t_p = stats.ttest_ind(n_succ, n_fail, equal_var=False)

add("=== DeepSeek-V4-Flash-Think subgroup analysis ===")
add(
    f"n_comparison: success mean = {n_succ.mean()}, "
    f"fail mean = {n_fail.mean()}, "
    f"t = {t_stat}, p = {t_p}"
)
add()

# intervention_category
ct_cat = pd.crosstab(df_valid["deepseekv4flash_success"], df_valid["intervention_category"])
chi2_cat, p_cat, dof_cat, min_exp_cat = chi_square_summary(ct_cat)

add("Intervention category chi-square:")
add(
    f"chi2 = {chi2_cat}, dof = {dof_cat}, p = {p_cat}, "
    f"min expected cell = {min_exp_cat}"
)
add(ct_cat.to_string())
add()

# type_experiment
ct_type = pd.crosstab(df_valid["deepseekv4flash_success"], df_valid["type_experiment"])
chi2_type, p_type, dof_type, min_exp_type = chi_square_summary(ct_type)

add("Type of experiment chi-square:")
add(
    f"chi2 = {chi2_type}, dof = {dof_type}, p = {p_type}, "
    f"min expected cell = {min_exp_type}"
)
add(ct_type.to_string())
add()

# intervention_technique
ct_tech = pd.crosstab(df_valid["deepseekv4flash_success"], df_valid["intervention_technique"])
chi2_tech, p_tech, dof_tech, min_exp_tech = chi_square_summary(ct_tech)

add("Intervention technique chi-square:")
add(
    f"chi2 = {chi2_tech}, dof = {dof_tech}, p = {p_tech}, "
    f"min expected cell = {min_exp_tech}"
)
add(ct_tech.to_string())
add()

if True in ct_cat.index:
    rates_by_cat = (ct_cat.loc[True] / ct_cat.sum(axis=0) * 100).round(1)
    add("Success rates by category (%):")
    add(rates_by_cat.to_string())
    add()

if True in ct_tech.index:
    rates_by_tech = (ct_tech.loc[True] / ct_tech.sum(axis=0) * 100).round(1)
    add("Success rates by intervention technique (%):")
    add(rates_by_tech.sort_values(ascending=False).to_string())
    add()

if False in ct_type.index:
    fail_type = ct_type.loc[False].sort_values(ascending=False)
    add("Failures by type of experiment:")
    add(fail_type.to_string())
    add()

if False in ct_tech.index:
    fail_tech = ct_tech.loc[False].sort_values(ascending=False)
    add("Failures by intervention technique:")
    add(fail_tech.to_string())
    add()

deepseekv4flash_technique_df = technique_summary(
    df_valid, success_col="deepseekv4flash_success", llm_d_col="deepseekv4flash_llm_effect_size"
)
add("DeepSeek-V4-Flash-Think technique-level descriptive summary:")
add(
    "Note: this table is useful for reviewer-facing interpretation because it shows "
    "which techniques have larger/smaller human effects and where DeepSeek-V4-Flash-Think diverges."
)
add(deepseekv4flash_technique_df.to_string(index=False))
add()

# write to txt
output_text = "\n".join(output_lines)

with open("study1.txt", "w", encoding="utf-8") as f:
    f.write(output_text)

print(output_text)
print("\nResults have been saved to study1.txt")
