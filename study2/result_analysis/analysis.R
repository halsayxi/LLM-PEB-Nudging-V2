required_packages <- c(
  "dplyr",
  "tidyr",
  "ggplot2",
  "ggnewscale",
  "readr",
  "patchwork",
  "ggbeeswarm",
  "lme4",
  "lmerTest",
  "emmeans",
  "car",
  "afex",
  "broom",
  "minpack.lm",
  "sandwich",
  "stringr",
  "purrr"
)

missing_packages <- setdiff(required_packages, rownames(installed.packages()))
if (length(missing_packages) > 0) {
  stop(
    paste0("Missing packages: ", paste(missing_packages, collapse = ", ")),
    call. = FALSE
  )
}

suppressWarnings(suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(ggnewscale)
  library(readr)
  library(patchwork)
  library(ggbeeswarm)
  library(lme4)
  library(lmerTest)
  library(emmeans)
  library(car)
  library(afex)
  library(broom)
  library(minpack.lm)
  library(sandwich)
  library(stringr)
  library(purrr)
}))

options(width = 200, scipen = 999)
emm_options(
  rg.limit = 50000,
  lmerTest.limit = 50000,
  pbkrtest.limit = 50000,
  disable.pbkrtest = TRUE
)
afex_options(type = 3)

if (!interactive()) {
  grDevices::pdf(file = NULL)
}

if (!exists("bootstrap_n")) {
  bootstrap_n <- 5000
}

get_script_dir <- function() {
  file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(file_arg) > 0) {
    return(dirname(normalizePath(sub("^--file=", "", file_arg[1]), winslash = "/", mustWork = FALSE)))
  }
  normalizePath(getwd(), winslash = "/", mustWork = FALSE)
}

clean_csv_names <- function(data) {
  names(data) <- sub("^\ufeff", "", names(data))
  data
}

format_p_value <- function(x) {
  ifelse(
    is.na(x),
    NA_character_,
    ifelse(x < 0.001, "<.001", formatC(round(x, 3), format = "f", digits = 3))
  )
}

format_numeric_column <- function(x, digits = 2, is_p = FALSE) {
  if (is_p) {
    return(format_p_value(x))
  }
  rounded_x <- round(x, digits)
  rounded_x[abs(rounded_x) < (10^(-digits) / 2)] <- 0
  ifelse(
    is.na(x),
    NA_character_,
    formatC(rounded_x, format = "f", digits = digits)
  )
}

format_table <- function(data, digits = 2) {
  output <- as.data.frame(data, check.names = FALSE, stringsAsFactors = FALSE)
  for (column_name in names(output)) {
    column <- output[[column_name]]
    is_p <- grepl("(^p$|^p\\.|^p_|_p$|p.value|p_value|Pr\\()", column_name)
    if (is.numeric(column)) {
      output[[column_name]] <- format_numeric_column(column, digits = digits, is_p = is_p)
    } else if (is.factor(column)) {
      output[[column_name]] <- as.character(column)
    }
  }
  output
}

print_subsection <- function(title) {
  cat("\n\n", title, "\n", sep = "")
  cat(strrep("-", nchar(title)), "\n", sep = "")
}

print_table <- function(data, title = NULL, digits = 2) {
  if (!is.null(title) && nzchar(title)) {
    cat("\n", title, "\n", sep = "")
    cat(strrep("=", nchar(title)), "\n", sep = "")
  }
  print(format_table(data, digits = digits), row.names = FALSE, right = FALSE)
  invisible(data)
}

save_png <- function(plot_obj, filename, width, height, dpi = 320) {
  output_path <- file.path(output_dir, filename)
  suppressWarnings(suppressMessages(
    ggplot2::ggsave(
      filename = output_path,
      plot = plot_obj,
      width = width,
      height = height,
      units = "in",
      dpi = dpi,
      bg = "white"
    )
  ))
  invisible(output_path)
}

group_levels <- c(
  "control",
  "information_low", "information_medium", "information_high",
  "structure_low", "structure_medium", "structure_high",
  "assistant_low", "assistant_medium", "assistant_high"
)

group_display_labels <- c(
  "control" = "Control",
  "information_low" = "Information low",
  "information_medium" = "Information medium",
  "information_high" = "Information high",
  "structure_low" = "Structure low",
  "structure_medium" = "Structure medium",
  "structure_high" = "Structure high",
  "assistant_low" = "Assistant low",
  "assistant_medium" = "Assistant medium",
  "assistant_high" = "Assistant high"
)

group_facet_levels <- c(
  "assistant_high", "structure_high", "information_high",
  "assistant_medium", "structure_medium", "information_medium",
  "assistant_low", "structure_low", "information_low"
)

group_facet_labels <- c(
  "assistant_high" = "Assistant\nHigh",
  "structure_high" = "Structure\nHigh",
  "information_high" = "Information\nHigh",
  "assistant_medium" = "Assistant\nMedium",
  "structure_medium" = "Structure\nMedium",
  "information_medium" = "Information\nMedium",
  "assistant_low" = "Assistant\nLow",
  "structure_low" = "Structure\nLow",
  "information_low" = "Information\nLow"
)

sex_levels <- c("Male", "Female")
ethnicity_levels <- c("White", "Black", "Asian", "Mixed", "Other")
occupation_levels <- c(
  "Student",
  "Government/Public sector",
  "Private sector",
  "Self-employed/Freelancer",
  "Other"
)
income_levels <- c(
  "<10,000",
  "10,001-20,000",
  "20,001-30,000",
  "30,001-40,000",
  "40,001-60,000",
  ">60,000"
)
education_levels <- c(
  "Primary or below",
  "Junior high",
  "Senior high",
  "College diploma",
  "Bachelor",
  "Master",
  "Doctoral"
)

demographic_covariates <- c("Age", "Sex", "ethnicity", "occupation", "income", "education")

intensity_palette <- c(
  "Low" = "#E4D591",
  "Medium" = "#CE8A37",
  "High" = "#B2513B",
  "Control" = "#687061"
)

type_palette <- c(
  "Information" = "#B02226",
  "Structure" = "#F0A12C",
  "Assistant" = "#009E73",
  "Control" = "#5C6770"
)

feedback_palette <- c(
  "Acceptance" = "#C5D6F0",
  "Reactance" = "#A9CA70",
  "Fatigue" = "#F18C54"
)

feedback_metric_min <- 1
feedback_metric_max <- 7

label_group <- function(x) {
  dplyr::recode(as.character(x), !!!group_display_labels, .default = as.character(x))
}

prettify_group_text <- function(x) {
  group_keys <- names(group_display_labels)[order(nchar(names(group_display_labels)), decreasing = TRUE)]
  output <- as.character(x)
  for (group_key in group_keys) {
    output <- stringr::str_replace_all(
      output,
      stringr::fixed(group_key),
      group_display_labels[[group_key]]
    )
  }
  output
}

label_nudge_type <- function(x) {
  dplyr::recode(
    as.character(x),
    "information" = "Information",
    "structure" = "Structure",
    "assistant" = "Assistant",
    .default = as.character(x)
  )
}

label_frequency_condition <- function(x) {
  dplyr::recode(
    as.character(x),
    "control" = "Control",
    "low" = "Low",
    "medium" = "Medium",
    "high" = "High",
    .default = as.character(x)
  )
}

label_frequency_contrast <- function(x) {
  dplyr::recode(
    as.character(x),
    "low vs control" = "Low vs Control",
    "medium vs control" = "Medium vs Control",
    "high vs control" = "High vs Control",
    "medium vs low" = "Medium vs Low",
    "high vs low" = "High vs Low",
    "high vs medium" = "High vs Medium",
    .default = as.character(x)
  )
}

label_structure_state <- function(x) {
  dplyr::recode(
    as.character(x),
    "control" = "Control",
    "structure_off" = "Structure Off",
    "structure_active" = "Structure Active",
    .default = as.character(x)
  )
}

label_structure_contrast <- function(x) {
  dplyr::recode(
    as.character(x),
    "structure off vs control" = "Structure Off vs Control",
    "structure active vs control" = "Structure Active vs Control",
    "structure active vs structure off" = "Structure Active vs Structure Off",
    .default = as.character(x)
  )
}

build_outcome_plot_df <- function(data, pre_col, post_col, lower_bound, upper_bound) {
  facet_levels <- unname(group_facet_labels[group_facet_levels])

  data %>%
    filter(as.character(group) %in% group_facet_levels) %>%
    filter(!is.na(ID), !is.na(.data[[pre_col]]), !is.na(.data[[post_col]])) %>%
    mutate(
      "{pre_col}" := pmin(pmax(.data[[pre_col]], lower_bound), upper_bound),
      "{post_col}" := pmin(pmax(.data[[post_col]], lower_bound), upper_bound),
      group = factor(as.character(group), levels = group_facet_levels, labels = facet_levels),
      sign_label = if_else(.data[[post_col]] > .data[[pre_col]], "increase", "decrease")
    )
}

build_feedback_band_df <- function(feedback_df, band_low, band_high) {
  facet_levels <- unname(group_facet_labels[group_facet_levels])
  scale_fac <- (band_high - band_low) / (feedback_metric_max - feedback_metric_min)

  feedback_df %>%
    filter(as.character(group) %in% group_facet_levels) %>%
    mutate(group = factor(as.character(group), levels = group_facet_levels, labels = facet_levels)) %>%
    select(group, nudge_acceptance_post, nudge_reactance_post, nudge_fatigue_post) %>%
    pivot_longer(
      cols = c(nudge_acceptance_post, nudge_reactance_post, nudge_fatigue_post),
      names_to = "Metric",
      values_to = "Score"
    ) %>%
    mutate(
      Metric = dplyr::recode(
        Metric,
        "nudge_acceptance_post" = "Acceptance",
        "nudge_reactance_post" = "Reactance",
        "nudge_fatigue_post" = "Fatigue"
      ),
      Metric = factor(Metric, levels = c("Acceptance", "Reactance", "Fatigue"))
    ) %>%
    group_by(group, Metric) %>%
    summarise(
      Mean = mean(Score, na.rm = TRUE),
      SD = sd(Score, na.rm = TRUE),
      N = sum(!is.na(Score)),
      SE = if_else(N > 0, SD / sqrt(N), NA_real_),
      .groups = "drop"
    ) %>%
    mutate(
      mean_scaled = band_low + (Mean - feedback_metric_min) * scale_fac,
      se_scaled = SE * scale_fac,
      ymin_scaled = band_low,
      ymax_scaled = mean_scaled,
      idx = as.integer(Metric),
      xmin = 1.125 - 0.42 / 2 + (idx - 1) * (0.42 / 3),
      xmax = xmin + (0.42 / 3),
      x_ctr = (xmin + xmax) / 2
    )
}

build_prepost_feedback_plot <- function(
  analysis_df,
  pre_col,
  post_col,
  y_label,
  feedback_source_df = analysis_df,
  y_display_limits = NULL,
  legend_mode = c("inside", "top", "none")
) {
  legend_mode <- match.arg(legend_mode)

  plot_df <- analysis_df %>%
    select(ID, group, sign_label, all_of(pre_col), all_of(post_col)) %>%
    pivot_longer(
      cols = all_of(c(pre_col, post_col)),
      names_to = "condition",
      values_to = "ScoreValue"
    ) %>%
    mutate(
      condition = factor(condition, levels = c(pre_col, post_col), labels = c("pre", "post")),
      xpos = if_else(condition == "pre", 1, 1.25)
    )

  set.seed(42)
  plot_df <- plot_df %>%
    group_by(ID) %>%
    mutate(x_offset = runif(1, -0.05, 0.05)) %>%
    ungroup() %>%
    mutate(xpos_j = xpos + x_offset)

  y_min <- min(plot_df$ScoreValue, na.rm = TRUE)
  y_max <- max(plot_df$ScoreValue, na.rm = TRUE)
  y_span <- y_max - y_min
  if (!is.finite(y_span) || y_span <= 0) {
    y_span <- 1
  }
  y_star <- y_max + 0.03 * y_span

  stats_df <- analysis_df %>%
    group_by(group) %>%
    summarise(
      p_value = tryCatch(
        t.test(.data[[pre_col]], .data[[post_col]], paired = TRUE)$p.value,
        error = function(e) NA_real_
      ),
      .groups = "drop"
    ) %>%
    mutate(
      sig = dplyr::case_when(
        is.na(p_value) ~ "",
        p_value < 0.001 ~ "***",
        p_value < 0.01 ~ "**",
        p_value < 0.05 ~ "*",
        TRUE ~ ""
      ),
      xpos = 1.125,
      ypos = y_star
    )

  p_main <- ggplot(plot_df, aes(x = xpos_j, y = ScoreValue)) +
    geom_line(
      aes(group = ID, color = sign_label),
      alpha = 0.30,
      show.legend = FALSE,
      na.rm = TRUE
    ) +
    scale_color_manual(values = c(increase = "#F48892", decrease = "grey50"), guide = "none") +
    ggnewscale::new_scale_color() +
    geom_boxplot(
      data = dplyr::filter(plot_df, condition == "pre"),
      aes(fill = condition),
      position = position_nudge(x = -0.15),
      width = 0.10,
      outlier.shape = NA,
      alpha = 0.88,
      colour = "#FDAE61",
      linewidth = 0.40,
      show.legend = TRUE,
      na.rm = TRUE
    ) +
    geom_point(aes(color = condition), size = 1, alpha = 0.60, na.rm = TRUE) +
    geom_boxplot(
      data = dplyr::filter(plot_df, condition == "post"),
      aes(x = xpos, y = ScoreValue, fill = condition),
      position = position_nudge(x = 0.15),
      width = 0.10,
      outlier.shape = NA,
      alpha = 0.88,
      colour = "#313695",
      linewidth = 0.40,
      na.rm = TRUE
    ) +
    geom_point(aes(color = condition), size = 1, alpha = 0.60, na.rm = TRUE) +
    scale_colour_manual(
      name = NULL,
      values = c(pre = "#FDAE61", post = "#313695"),
      breaks = c("pre", "post")
    ) +
    scale_fill_manual(
      name = NULL,
      values = c(pre = "#FDAE61", post = "#313695"),
      breaks = c("pre", "post")
    ) +
    facet_wrap(~ group, nrow = 1, strip.position = "bottom") +
    scale_x_continuous(breaks = NULL) +
    labs(y = y_label) +
    theme_minimal(base_size = 15) +
    theme(
      axis.title.x = element_blank(),
      axis.title.y = element_text(size = 12, face = "bold"),
      panel.grid.major.x = element_blank(),
      strip.background = element_blank(),
      strip.placement = "outside",
      strip.text = element_text(size = 11, face = "bold"),
      legend.direction = "horizontal",
      legend.justification = c("right", "top"),
      legend.background = element_rect(fill = scales::alpha("white", 0.75), colour = NA),
      legend.key.size = grid::unit(0.30, "cm"),
      legend.text = element_text(size = 12),
      legend.margin = margin(t = 0.10, r = 0.10, b = 0.10, l = 0.10, unit = "cm")
    )

  band_low <- y_min
  band_high <- y_min + 0.15 * y_span
  bars_df <- build_feedback_band_df(feedback_source_df, band_low, band_high)

  last_group <- tail(levels(plot_df$group), 1)
  x_axis <- 1.5
  tick_len <- 0.012
  scale_fac <- (band_high - band_low) / (feedback_metric_max - feedback_metric_min)

  axis_spine <- tibble::tibble(group = factor(last_group, levels = levels(plot_df$group)))
  axis_ticks <- tibble::tibble(
    group = factor(last_group, levels = levels(plot_df$group)),
    y = band_low + ((feedback_metric_min:feedback_metric_max) - feedback_metric_min) * scale_fac,
    lab = as.character(feedback_metric_min:feedback_metric_max)
  )

  p_all <- p_main +
    geom_text(
      data = stats_df,
      aes(x = xpos, y = ypos, label = sig),
      inherit.aes = FALSE,
      size = 5,
      vjust = 0,
      na.rm = TRUE
    ) +
    expand_limits(y = y_star) +
    ggnewscale::new_scale_fill() +
    geom_rect(
      data = bars_df,
      aes(
        xmin = xmin, xmax = xmax,
        ymin = ymin_scaled, ymax = ymax_scaled,
        fill = Metric
      ),
      inherit.aes = FALSE,
      color = NA,
      show.legend = FALSE,
      na.rm = TRUE
    ) +
    geom_errorbar(
      data = bars_df,
      aes(
        x = x_ctr,
        ymin = pmax(mean_scaled - se_scaled, band_low),
        ymax = pmin(mean_scaled + se_scaled, band_high)
      ),
      inherit.aes = FALSE,
      width = (0.42 / 3) * 0.28,
      linewidth = 0.45,
      show.legend = FALSE,
      na.rm = TRUE
    ) +
    geom_segment(
      data = axis_spine,
      aes(x = x_axis - 0.05, xend = x_axis - 0.05, y = band_low, yend = band_high),
      inherit.aes = FALSE,
      linewidth = 0.45,
      na.rm = TRUE
    ) +
    geom_segment(
      data = axis_ticks,
      aes(x = x_axis - tick_len - 0.05, xend = x_axis - 0.05, y = y, yend = y),
      inherit.aes = FALSE,
      linewidth = 0.45,
      na.rm = TRUE
    ) +
    geom_text(
      data = axis_ticks,
      aes(x = x_axis + 0.012, y = y, label = lab),
      inherit.aes = FALSE,
      size = 2.7,
      hjust = 0,
      vjust = 0.35,
      na.rm = TRUE
    ) +
    scale_fill_manual(values = feedback_palette, guide = "none") +
    coord_cartesian(clip = "off")

  if (!is.null(y_display_limits)) {
    p_all <- p_all + coord_cartesian(ylim = y_display_limits, clip = "off")
  }

  if (legend_mode == "inside") {
    p_all <- p_all +
      theme(
        legend.position = c(0.985, 0.985),
        legend.justification = c("right", "top")
      )
  } else if (legend_mode == "top") {
    p_all <- p_all +
      theme(
        legend.position = "top",
        legend.justification = "center"
      )
  } else {
    p_all <- p_all +
      theme(legend.position = "none")
  }

  p_all
}

coerce_labeled_factor <- function(x, labels, var_name) {
  x_chr <- trimws(as.character(x))
  x_chr[x_chr %in% c("", "NA")] <- NA_character_
  code_map <- setNames(labels, as.character(seq_along(labels)))
  x_recoded <- dplyr::recode(x_chr, !!!code_map, .default = x_chr)
  unexpected_values <- sort(unique(x_recoded[!is.na(x_recoded) & !x_recoded %in% labels]))
  if (length(unexpected_values) > 0) {
    warning(
      sprintf(
        "Unexpected values in %s were converted to NA: %s",
        var_name,
        paste(unexpected_values, collapse = ", ")
      )
    )
  }
  factor(x_recoded, levels = labels)
}

add_mean_nudge_rt <- function(data, set_structure_zero = FALSE) {
  rt_cols <- grep(
    "^(information_nudge_rt_ms_day|assistant_commitment_rt_ms_day)[0-9]+$",
    names(data),
    value = TRUE
  )

  if (length(rt_cols) > 0) {
    mean_nudge_rt <- rowMeans(log1p(data[, rt_cols, drop = FALSE]), na.rm = TRUE)
    mean_nudge_rt[is.nan(mean_nudge_rt)] <- NA_real_
  } else {
    mean_nudge_rt <- rep(NA_real_, nrow(data))
  }

  if (set_structure_zero && "nudgeType" %in% names(data)) {
    mean_nudge_rt[data$nudgeType == "structure"] <- 0
  }

  data$mean_nudge_rt <- mean_nudge_rt
  data
}

calc_fit_r2 <- function(obs, fit_aug) {
  if (is.null(fit_aug) || nrow(fit_aug) == 0 || all(is.na(fit_aug$.fitted))) {
    return(NA_real_)
  }
  stats::cor(obs, fit_aug$.fitted, use = "complete.obs")^2
}

calc_auc <- function(x, y) {
  if (length(x) < 2 || length(y) < 2 || all(is.na(y))) {
    return(NA_real_)
  }
  ord <- order(x)
  x2 <- x[ord]
  y2 <- y[ord]
  sum(diff(x2) * (head(y2, -1) + tail(y2, -1)) / 2)
}

calc_partial_r2_from_t <- function(t_value, df_residual) {
  ifelse(
    is.finite(t_value) & is.finite(df_residual) & df_residual > 0,
    (t_value^2) / (t_value^2 + df_residual),
    NA_real_
  )
}

calc_partial_eta_sq_from_f <- function(f_value, df1, df2) {
  ifelse(
    is.finite(f_value) & is.finite(df1) & is.finite(df2) & df1 > 0 & df2 > 0,
    (f_value * df1) / ((f_value * df1) + df2),
    NA_real_
  )
}

label_analysis_var <- function(var_name) {
  dplyr::recode(
    var_name,
    "delta_attitude" = "Delta attitude",
    "delta_motivation" = "Delta motivation",
    "delta_self_efficacy" = "Delta self-efficacy",
    "nudge_acceptance_post" = "Nudge acceptance",
    "nudge_fatigue_post" = "Nudge fatigue",
    "nudge_reactance_post" = "Nudge reactance",
    .default = var_name
  )
}

donation_get_prompt_days <- function(intensity) {
  dplyr::case_when(
    intensity == "low" ~ list(c(1L)),
    intensity == "medium" ~ list(c(1L, 4L, 7L, 10L, 13L)),
    intensity == "high" ~ list(1:14),
    TRUE ~ list(integer(0))
  )
}

build_day_long <- function(wide_df) {
  wide_df %>%
    mutate(
      ID = factor(ID),
      nudgeType = factor(nudgeType, levels = c("information", "structure", "assistant", "control")),
      nudgeIntensity = factor(nudgeIntensity, levels = c("control", "low", "medium", "high")),
      frequency4 = factor(
        ifelse(as.character(group) == "control", "control", as.character(nudgeIntensity)),
        levels = c("control", "low", "medium", "high")
      )
    ) %>%
    pivot_longer(
      cols = matches("^(eco_choices|daily_donation_cents|n_trials|information_nudge_rt_ms|assistant_commitment_rt_ms)_day\\d+$"),
      names_to = c(".value", "day"),
      names_pattern = "(.*)_day(\\d+)$"
    ) %>%
    mutate(
      day = as.integer(day),
      eco_share = eco_choices / n_trials,
      donation_cents = daily_donation_cents
    ) %>%
    rowwise() %>%
    mutate(
      prompt_days = donation_get_prompt_days(as.character(nudgeIntensity)),
      nudge_active = if_else(as.character(group) == "control", FALSE, day %in% prompt_days),
      structure_state = dplyr::case_when(
        as.character(group) == "control" ~ "control",
        as.character(nudgeType) == "structure" & nudge_active ~ "structure_active",
        as.character(nudgeType) == "structure" & !nudge_active ~ "structure_off",
        TRUE ~ NA_character_
      ),
      event_window = dplyr::case_when(
        as.character(nudgeIntensity) == "medium" & day %in% c(4L, 7L, 10L, 13L) ~ "NudgeDay",
        as.character(nudgeIntensity) == "medium" & day %in% c(2L, 5L, 8L, 11L, 14L) ~ "PostDay1",
        as.character(nudgeIntensity) == "medium" & day %in% c(3L, 6L, 9L, 12L) ~ "PostDay2",
        TRUE ~ NA_character_
      )
    ) %>%
    ungroup() %>%
    mutate(
      structure_state = factor(structure_state, levels = c("control", "structure_off", "structure_active")),
      event_window = factor(event_window, levels = c("NudgeDay", "PostDay1", "PostDay2")),
      any_donation = as.integer(!is.na(donation_cents) & donation_cents > 0),
      positive_log_donation = if_else(donation_cents > 0, log(donation_cents), NA_real_)
    ) %>%
    select(-prompt_days)
}

prepare_simulation_validity_wide <- function(data, source_name) {
  data <- clean_csv_names(as.data.frame(data, check.names = FALSE))

  required_cols <- c("ID", "group")
  missing_required <- setdiff(required_cols, names(data))
  if (length(missing_required) > 0) {
    stop(
      paste0(source_name, " is missing required columns: ", paste(missing_required, collapse = ", ")),
      call. = FALSE
    )
  }

  data <- data %>%
    mutate(
      ID = as.character(ID),
      group = as.character(group)
    ) %>%
    select(-any_of(c("nudgeType", "nudgeIntensity"))) %>%
    mutate(
      nudgeType = dplyr::case_when(
        group == "control" ~ "control",
        grepl("_", group) ~ sub("_(low|medium|high)$", "", group),
        TRUE ~ NA_character_
      ),
      nudgeIntensity = dplyr::case_when(
        group == "control" ~ "control",
        grepl("_", group) ~ sub("^.*_", "", group),
        TRUE ~ NA_character_
      )
    )

  for (day in 1:14) {
    choice_col <- paste0("eco_choices_day", day)
    trial_col <- paste0("n_trials_day", day)
    share_col <- paste0("eco_share_day", day)

    if (choice_col %in% names(data)) {
      data[[choice_col]] <- suppressWarnings(as.numeric(data[[choice_col]]))
    }
    if (trial_col %in% names(data)) {
      data[[trial_col]] <- suppressWarnings(as.numeric(data[[trial_col]]))
    }
    if (share_col %in% names(data)) {
      data[[share_col]] <- suppressWarnings(as.numeric(data[[share_col]]))
    } else if (choice_col %in% names(data) && trial_col %in% names(data)) {
      data[[share_col]] <- data[[choice_col]] / ifelse(data[[trial_col]] == 0, NA_real_, data[[trial_col]])
    } else if (choice_col %in% names(data)) {
      data[[share_col]] <- data[[choice_col]] / 3
    }
  }

  missing_share_cols <- setdiff(paste0("eco_share_day", 1:14), names(data))
  if (length(missing_share_cols) > 0) {
    stop(
      paste0(source_name, " is missing daily eco-share columns: ", paste(missing_share_cols, collapse = ", ")),
      call. = FALSE
    )
  }

  data
}

read_simulation_validity_wide <- function(path, source_name) {
  if (!file.exists(path)) {
    stop(paste0(source_name, " was not found: ", path), call. = FALSE)
  }

  readr::read_csv(path, show_col_types = FALSE, locale = readr::locale(encoding = "UTF-8")) %>%
    prepare_simulation_validity_wide(source_name = source_name)
}

simulation_validity_to_long <- function(data, source_name) {
  share_cols <- paste0("eco_share_day", 1:14)
  choice_cols <- paste0("eco_choices_day", 1:14)

  long_data <- data %>%
    select(ID, group, nudgeType, nudgeIntensity, any_of(choice_cols), all_of(share_cols)) %>%
    pivot_longer(
      cols = matches("^eco_(choices|share)_day\\d+$"),
      names_to = c(".value", "day"),
      names_pattern = "^(eco_(?:choices|share))_day(\\d+)$"
    )

  if (!("eco_choices" %in% names(long_data))) {
    long_data$eco_choices <- NA_real_
  }

  long_data %>%
    mutate(
      day = as.integer(day),
      eco_choices = suppressWarnings(as.numeric(eco_choices)),
      eco_share = suppressWarnings(as.numeric(eco_share)),
      source = source_name
    )
}

simulation_validity_pair <- function(human, simulation) {
  data.frame(human = as.numeric(human), simulation = as.numeric(simulation)) %>%
    filter(is.finite(human), is.finite(simulation))
}

simulation_validity_ccc <- function(human, simulation) {
  pair <- simulation_validity_pair(human, simulation)
  if (nrow(pair) < 3) return(NA_real_)

  mean_h <- mean(pair$human)
  mean_s <- mean(pair$simulation)
  var_h <- mean((pair$human - mean_h)^2)
  var_s <- mean((pair$simulation - mean_s)^2)
  cov_hs <- mean((pair$human - mean_h) * (pair$simulation - mean_s))
  denom <- var_h + var_s + (mean_h - mean_s)^2
  if (denom == 0) return(NA_real_)

  2 * cov_hs / denom
}

simulation_validity_cor <- function(human, simulation, metric = c("r", "p")) {
  metric <- match.arg(metric)
  pair <- simulation_validity_pair(human, simulation)
  if (nrow(pair) < 3 || sd(pair$human) == 0 || sd(pair$simulation) == 0) {
    return(NA_real_)
  }

  cor_out <- suppressWarnings(stats::cor.test(pair$human, pair$simulation, method = "pearson"))
  if (metric == "r") {
    return(unname(cor_out$estimate))
  }
  cor_out$p.value
}

simulation_validity_paired_stats <- function(human, simulation) {
  pair <- simulation_validity_pair(human, simulation)
  n <- nrow(pair)
  if (n == 0) {
    return(tibble::tibble(
      N = 0,
      `Human M` = NA_real_,
      `Simulation M` = NA_real_,
      `Delta M` = NA_real_,
      MAE = NA_real_,
      r = NA_real_,
      p = NA_real_,
      CCC = NA_real_
    ))
  }

  tibble::tibble(
    N = n,
    `Human M` = mean(pair$human),
    `Simulation M` = mean(pair$simulation),
    `Delta M` = mean(pair$simulation - pair$human),
    MAE = mean(abs(pair$simulation - pair$human)),
    r = simulation_validity_cor(pair$human, pair$simulation, "r"),
    p = simulation_validity_cor(pair$human, pair$simulation, "p"),
    CCC = simulation_validity_ccc(pair$human, pair$simulation)
  )
}

pair_simulation_validity_data <- function(human_wide, simulation_wide) {
  human_long <- simulation_validity_to_long(human_wide, "human") %>%
    rename(
      group_human = group,
      nudgeType_human = nudgeType,
      nudgeIntensity_human = nudgeIntensity,
      eco_choices_human = eco_choices,
      eco_share_human = eco_share
    )

  simulation_long <- simulation_validity_to_long(simulation_wide, "simulation") %>%
    rename(
      group_simulation = group,
      nudgeType_simulation = nudgeType,
      nudgeIntensity_simulation = nudgeIntensity,
      eco_choices_simulation = eco_choices,
      eco_share_simulation = eco_share
    )

  paired <- full_join(human_long, simulation_long, by = c("ID", "day"))

  unmatched_ids <- paired %>%
    filter(is.na(group_human) | is.na(group_simulation)) %>%
    distinct(ID)

  if (nrow(unmatched_ids) > 0) {
    stop(
      paste0(
        "Human and simulation data cannot be fully paired by ID/day. Example IDs: ",
        paste(head(unmatched_ids$ID, 5), collapse = ", ")
      ),
      call. = FALSE
    )
  }

  group_mismatches <- paired %>%
    filter(group_human != group_simulation) %>%
    distinct(ID, group_human, group_simulation)

  if (nrow(group_mismatches) > 0) {
    stop(
      paste0(
        "The same ID has different groups in human and simulation data. Example IDs: ",
        paste(head(group_mismatches$ID, 5), collapse = ", ")
      ),
      call. = FALSE
    )
  }

  paired %>%
    mutate(
      group = factor(group_human, levels = group_levels),
      nudgeType = nudgeType_human,
      nudgeIntensity = nudgeIntensity_human
    )
}

make_simulation_group_day <- function(paired) {
  paired %>%
    group_by(group, day) %>%
    summarise(
      human_n = sum(!is.na(eco_share_human)),
      simulation_n = sum(!is.na(eco_share_simulation)),
      human_eco_share = mean(eco_share_human, na.rm = TRUE),
      simulation_eco_share = mean(eco_share_simulation, na.rm = TRUE),
      human_eco_choices = mean(eco_choices_human, na.rm = TRUE),
      simulation_eco_choices = mean(eco_choices_simulation, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(
      group = factor(as.character(group), levels = group_levels),
      Group = label_group(group),
      gap = simulation_eco_share - human_eco_share
    ) %>%
    arrange(group, day)
}

make_simulation_treatment_effects <- function(group_day) {
  control <- group_day %>%
    filter(as.character(group) == "control") %>%
    select(
      day,
      human_control_eco_share = human_eco_share,
      simulation_control_eco_share = simulation_eco_share
    )

  group_day %>%
    filter(as.character(group) != "control") %>%
    left_join(control, by = "day") %>%
    mutate(
      human_effect = human_eco_share - human_control_eco_share,
      simulation_effect = simulation_eco_share - simulation_control_eco_share
    )
}

make_simulation_validation_summary <- function(group_day) {
  simulation_validity_paired_stats(group_day$human_eco_share, group_day$simulation_eco_share) %>%
    mutate(
      Analysis = "Group-by-day means",
      Unit = "10 groups x 14 days"
    ) %>%
    select(Analysis, Unit, N, `Human M`, `Simulation M`, `Delta M`, MAE, r, p)
}

make_simulation_condition_table <- function(group_day) {
  group_day %>%
    group_by(group, Group) %>%
    summarise(
      Days = sum(is.finite(human_eco_share) & is.finite(simulation_eco_share)),
      `Human 14-day M` = mean(human_eco_share, na.rm = TRUE),
      `Simulation 14-day M` = mean(simulation_eco_share, na.rm = TRUE),
      `Delta M` = mean(simulation_eco_share - human_eco_share, na.rm = TRUE),
      MAE = mean(abs(simulation_eco_share - human_eco_share), na.rm = TRUE),
      r = simulation_validity_cor(human_eco_share, simulation_eco_share, "r"),
      p = simulation_validity_cor(human_eco_share, simulation_eco_share, "p"),
      .groups = "drop"
    ) %>%
    arrange(group) %>%
    select(Group, Days, `Human 14-day M`, `Simulation 14-day M`, `Delta M`, MAE, r, p)
}

build_simulation_main_trend_plot <- function(simulation_wide) {
  simulation_long <- simulation_validity_to_long(simulation_wide, "simulation") %>%
    filter(!is.na(eco_share)) %>%
    mutate(
      nudgeType = factor(nudgeType, levels = c("information", "structure", "assistant", "control")),
      nudgeIntensity = factor(nudgeIntensity, levels = c("control", "low", "medium", "high"))
    )

  plot_total_df <- simulation_long %>%
    group_by(nudgeIntensity, day) %>%
    summarise(mean_eco_share = mean(eco_share, na.rm = TRUE), .groups = "drop") %>%
    mutate(panel = "Total")

  plot_treat_df <- simulation_long %>%
    filter(as.character(nudgeType) %in% c("information", "structure", "assistant")) %>%
    group_by(nudgeType, nudgeIntensity, day) %>%
    summarise(mean_eco_share = mean(eco_share, na.rm = TRUE), .groups = "drop") %>%
    mutate(panel = label_nudge_type(nudgeType))

  plot_control_df <- simulation_long %>%
    filter(as.character(nudgeType) == "control") %>%
    group_by(day) %>%
    summarise(mean_eco_share = mean(eco_share, na.rm = TRUE), .groups = "drop") %>%
    mutate(nudgeIntensity = factor("control", levels = c("control", "low", "medium", "high")))

  plot_control_expanded_df <- tidyr::crossing(
    panel = c("Information", "Structure", "Assistant"),
    plot_control_df
  )

  simulation_plot_df <- bind_rows(
    plot_total_df,
    plot_treat_df,
    plot_control_expanded_df
  ) %>%
    mutate(
      panel = factor(panel, levels = c("Total", "Information", "Structure", "Assistant")),
      nudgeIntensity = factor(
        as.character(nudgeIntensity),
        levels = c("low", "medium", "high", "control"),
        labels = c("Low", "Medium", "High", "Control")
      )
    )

  ggplot(
    simulation_plot_df,
    aes(x = day, y = mean_eco_share, color = nudgeIntensity, group = nudgeIntensity)
  ) +
    geom_smooth(
      method = "loess",
      formula = y ~ x,
      span = 0.6,
      se = TRUE,
      color = NA,
      fill = "grey85"
    ) +
    stat_smooth(
      method = "loess",
      formula = y ~ x,
      span = 0.3,
      linewidth = 1.1,
      se = FALSE
    ) +
    geom_point(size = 1.6, alpha = 0.6) +
    facet_grid(. ~ panel) +
    scale_color_manual(values = intensity_palette) +
    scale_x_continuous(breaks = c(1, 4, 7, 10, 14), minor_breaks = 1:14) +
    scale_y_continuous(breaks = seq(0, 1, by = 0.2)) +
    coord_cartesian(ylim = c(0, 1.1)) +
    labs(
      x = "Day",
      y = "Simulated pro-environmental choice",
      color = NULL
    ) +
    guides(color = guide_legend(nrow = 2, byrow = TRUE)) +
    theme_bw(base_size = 15) +
    theme(
      panel.grid.minor = element_line(color = "grey92", linewidth = 0.3),
      panel.grid.major = element_line(color = "grey88", linewidth = 0.4),
      strip.background = element_blank(),
      strip.text = element_text(size = 12, face = "bold"),
      axis.title = element_text(face = "bold"),
      aspect.ratio = 1,
      legend.position = c(0.01, 0.98),
      legend.justification = c("left", "top"),
      legend.direction = "horizontal",
      legend.background = element_rect(fill = scales::alpha("white", 0.7), color = NA),
      legend.key.size = grid::unit(0.4, "cm"),
      legend.text = element_text(size = 9)
    )
}

build_simulation_gap_plot <- function(simulation_group_day) {
  plot_text_size <- 12
  plot_title_size <- 15
  plot_axis_text_size <- 12.5
  plot_colorbar_text_size <- 12
  pt_to_mm <- function(x) x / 2.845276

  type_levels <- c("information", "structure", "assistant")
  type_labels <- c(
    "information" = "Information",
    "structure" = "Structure",
    "assistant" = "Assistant"
  )
  type_colors <- c(
    "Information" = "#B02226",
    "Structure" = "#F0A12C",
    "Assistant" = "#009E73"
  )
  intensity_levels <- c("high", "medium", "low")
  intensity_labels <- c(
    "high" = "High",
    "medium" = "Medium",
    "low" = "Low"
  )
  strip_colors <- c(
    "High" = "#D65245",
    "Medium" = "#B4563D",
    "Low" = "#C98F92"
  )

  gap_daily <- simulation_group_day %>%
    mutate(
      nudgeType = dplyr::case_when(
        as.character(group) == "control" ~ "control",
        grepl("_", as.character(group)) ~ sub("_(low|medium|high)$", "", as.character(group)),
        TRUE ~ NA_character_
      ),
      nudgeIntensity = dplyr::case_when(
        as.character(group) == "control" ~ "control",
        grepl("_", as.character(group)) ~ sub("^.*_", "", as.character(group)),
        TRUE ~ NA_character_
      )
    ) %>%
    filter(
      nudgeType %in% type_levels,
      nudgeIntensity %in% intensity_levels
    ) %>%
    transmute(
      day = day,
      nudgeType = factor(type_labels[nudgeType], levels = unname(type_labels[type_levels])),
      nudgeIntensity = factor(intensity_labels[nudgeIntensity], levels = unname(intensity_labels[intensity_levels])),
      type_index = as.integer(factor(type_labels[nudgeType], levels = unname(type_labels[type_levels]))),
      gap_mean = gap
    )

  max_day <- max(gap_daily$day, na.rm = TRUE)
  gap_limit <- max(abs(gap_daily$gap_mean), na.rm = TRUE)
  gap_limit <- ceiling(gap_limit * 20) / 20
  if (!is.finite(gap_limit) || gap_limit == 0) {
    gap_limit <- 0.05
  }

  header_bars <- tidyr::crossing(
    nudgeIntensity = factor(unname(intensity_labels[intensity_levels]), levels = unname(intensity_labels[intensity_levels])),
    nudgeType = factor(unname(type_labels[type_levels]), levels = unname(type_labels[type_levels]))
  ) %>%
    mutate(type_index = as.integer(nudgeType))

  strip_df <- tibble::tibble(
    nudgeIntensity = factor(unname(intensity_labels[intensity_levels]), levels = unname(intensity_labels[intensity_levels])),
    strip_label = as.character(nudgeIntensity),
    strip_color = strip_colors[strip_label]
  )

  legend_df <- tibble::tibble(
    nudgeType = factor(unname(type_labels[type_levels]), levels = unname(type_labels[type_levels])),
    x = c(0.45, 3.35, 6.05),
    label_x = x + 0.32
  )

  legend_plot <- ggplot(legend_df) +
    geom_point(
      aes(x = x, y = 1, fill = nudgeType),
      shape = 22,
      size = 3.1,
      color = "black",
      stroke = 0.45
    ) +
    geom_text(
      aes(x = label_x, y = 1, label = nudgeType),
      hjust = 0,
      size = pt_to_mm(plot_title_size),
      family = ""
    ) +
    scale_fill_manual(values = type_colors) +
    coord_cartesian(xlim = c(0.15, 7.9), ylim = c(0.75, 1.25), clip = "off") +
    theme_void() +
    theme(legend.position = "none", plot.margin = margin(0, 5, -3, 5))

  gap_plot <- ggplot(gap_daily, aes(x = type_index, y = day)) +
    geom_rect(
      data = header_bars,
      aes(
        xmin = type_index - 0.48,
        xmax = type_index + 0.48,
        ymin = -0.75,
        ymax = -0.30,
        fill = nudgeType
      ),
      inherit.aes = FALSE,
      color = "black",
      linewidth = 0.45
    ) +
    scale_fill_manual(values = type_colors, guide = "none") +
    ggnewscale::new_scale_fill() +
    geom_rect(
      data = strip_df,
      aes(xmin = 0.5, xmax = 3.5, ymin = -0.18, ymax = 0.55, fill = strip_color),
      inherit.aes = FALSE,
      color = NA
    ) +
    scale_fill_identity(guide = "none") +
    geom_text(
      data = strip_df,
      aes(x = 2, y = 0.18, label = strip_label),
      inherit.aes = FALSE,
      color = "white",
      fontface = "bold",
      size = pt_to_mm(plot_title_size)
    ) +
    ggnewscale::new_scale_fill() +
    geom_point(
      aes(fill = gap_mean),
      shape = 21,
      size = 4.2,
      color = "black",
      stroke = 0.45
    ) +
    facet_grid(. ~ nudgeIntensity) +
    scale_x_continuous(
      limits = c(0.5, 3.5),
      breaks = NULL,
      expand = c(0, 0)
    ) +
    scale_y_reverse(
      limits = c(max_day + 0.5, -0.75),
      breaks = 1:max_day,
      expand = c(0, 0)
    ) +
    scale_fill_gradient2(
      low = "#2F4EA2",
      mid = "#F1EFE4",
      high = "#E23D2B",
      midpoint = 0,
      limits = c(-gap_limit, gap_limit),
      breaks = pretty(c(-gap_limit, gap_limit), n = 5),
      name = NULL
    ) +
    coord_cartesian(clip = "off") +
    labs(x = NULL, y = NULL) +
    theme_classic(base_size = plot_text_size) +
    theme(
      axis.text.y = element_text(color = "black", face = "bold", size = plot_axis_text_size),
      axis.ticks = element_blank(),
      axis.line = element_blank(),
      panel.border = element_rect(color = "black", fill = NA, linewidth = 0.45),
      panel.spacing.x = grid::unit(0.38, "cm"),
      strip.background = element_blank(),
      strip.text = element_blank(),
      legend.position = "bottom",
      legend.direction = "horizontal",
      legend.justification = "center",
      legend.box.margin = margin(0, 0, 0, 0),
      legend.text = element_text(size = plot_colorbar_text_size),
      legend.key.height = grid::unit(0.30, "cm"),
      legend.key.width = grid::unit(1.65, "cm"),
      plot.margin = margin(0, 2, 2, 2)
    ) +
    guides(
      fill = guide_colorbar(
        title.position = "top",
        title.hjust = 0.5,
        direction = "horizontal",
        barwidth = grid::unit(5.8, "cm"),
        barheight = grid::unit(0.32, "cm"),
        frame.colour = "black",
        ticks.colour = "black"
      )
    )

  legend_plot / gap_plot + patchwork::plot_layout(heights = c(0.58, 8.5))
}

donation_extract_model_fit <- function(model) {
  if (is.list(model) && !is.null(model$fit)) {
    return(model$fit)
  }
  model
}

donation_extract_model_vcov <- function(model) {
  if (is.list(model) && !is.null(model$vcov)) {
    return(model$vcov)
  }
  stats::vcov(model)
}

donation_linear_combo_model <- function(model, weights, label, exponentiate = FALSE) {
  beta <- stats::coef(donation_extract_model_fit(model))
  vc <- donation_extract_model_vcov(model)
  l_vec <- rep(0, length(beta))
  names(l_vec) <- names(beta)
  missing_terms <- setdiff(names(weights), names(beta))

  if (length(missing_terms) > 0) {
    stop(sprintf("Missing model terms: %s", paste(missing_terms, collapse = ", ")), call. = FALSE)
  }

  l_vec[names(weights)] <- weights
  est <- sum(l_vec * beta)
  se <- as.numeric(sqrt(t(l_vec) %*% vc %*% l_vec))
  z_value <- est / se
  p_value <- 2 * stats::pnorm(abs(z_value), lower.tail = FALSE)
  ci_low <- est - stats::qnorm(0.975) * se
  ci_high <- est + stats::qnorm(0.975) * se

  if (isTRUE(exponentiate)) {
    return(
      tibble::tibble(
        contrast = label,
        estimate = exp(est),
        conf_low = exp(ci_low),
        conf_high = exp(ci_high),
        p_value = p_value
      )
    )
  }

  tibble::tibble(
    contrast = label,
    estimate = est,
    conf_low = ci_low,
    conf_high = ci_high,
    p_value = p_value
  )
}

donation_fit_poisson_model <- function(data, formula) {
  fit <- stats::glm(
    formula = formula,
    data = data,
    family = poisson(link = "log")
  )
  list(
    fit = fit,
    vcov = sandwich::vcovCL(fit, cluster = data$ID, type = "HC1")
  )
}

donation_fit_logistic_model <- function(data, formula) {
  fit <- stats::glm(
    formula = formula,
    data = data,
    family = binomial(link = "logit")
  )
  list(
    fit = fit,
    vcov = sandwich::vcovCL(fit, cluster = data$ID, type = "HC1")
  )
}

donation_fit_positive_amount_model <- function(data, formula) {
  fit <- stats::lm(
    formula = formula,
    data = data
  )
  list(
    fit = fit,
    vcov = sandwich::vcovCL(fit, cluster = data$ID, type = "HC1")
  )
}

build_donation_hurdle_plot_data <- function(day_long_data) {
  day_long_data %>%
    filter(!is.na(donation_cents)) %>%
    group_by(ID, frequency4) %>%
    summarise(
      any_donation_probability = mean(any_donation, na.rm = TRUE),
      positive_donation_amount = dplyr::if_else(
        any(donation_cents > 0, na.rm = TRUE),
        mean(donation_cents[donation_cents > 0], na.rm = TRUE),
        NA_real_
      ),
      .groups = "drop"
    ) %>%
    pivot_longer(
      cols = c(any_donation_probability, positive_donation_amount),
      names_to = "panel",
      values_to = "value"
    ) %>%
    mutate(
      panel = factor(
        panel,
        levels = c("any_donation_probability", "positive_donation_amount"),
        labels = c(
          "Probability of any donation",
          "Positive donation amount\nconditional on donation"
        )
      ),
      frequency_label = factor(
        frequency4,
        levels = c("control", "low", "medium", "high"),
        labels = c("Control", "Low", "Medium", "High")
      )
    )
}

run_focus_main_effect <- function(data, predictor_var, predictor_label, baseline_var_map) {
  baseline_var <- unname(baseline_var_map[[predictor_var]])
  rhs_terms <- c("predictor_z", "group", demographic_covariates)
  select_vars <- c("group", "eco_choice_change", "predictor_value", demographic_covariates)

  if (!is.na(baseline_var) && nzchar(baseline_var)) {
    rhs_terms <- c(rhs_terms, baseline_var)
    select_vars <- c(select_vars, baseline_var)
  }

  analysis_df <- data %>%
    transmute(
      group = droplevels(group),
      eco_choice_change = eco_choice_change,
      predictor_value = .data[[predictor_var]],
      Age = Age,
      Sex = Sex,
      ethnicity = ethnicity,
      occupation = occupation,
      income = income,
      education = education,
      envAttitude_pre = envAttitude_pre,
      envMotivation_pre = envMotivation_pre,
      envSelfEfficacy_pre = envSelfEfficacy_pre
    ) %>%
    select(all_of(unique(select_vars))) %>%
    tidyr::drop_na()

  if (
    nrow(analysis_df) < 15 ||
      n_distinct(analysis_df$group) < 2 ||
      n_distinct(analysis_df$eco_choice_change) < 2 ||
      n_distinct(analysis_df$predictor_value) < 2
  ) {
    return(
      tibble::tibble(
        Predictor = predictor_label,
        beta = NA_real_,
        SE = NA_real_,
        t = NA_real_,
        p = NA_real_,
        `Lower 95% CI` = NA_real_,
        `Upper 95% CI` = NA_real_
      )
    )
  }

  analysis_df <- analysis_df %>%
    mutate(
      eco_choice_change_z = as.numeric(scale(eco_choice_change)),
      predictor_z = as.numeric(scale(predictor_value))
    )

  fit <- lm(
    stats::reformulate(rhs_terms, response = "eco_choice_change_z"),
    data = analysis_df
  )

  fit_row <- broom::tidy(fit, conf.int = TRUE) %>%
    filter(term == "predictor_z")

  tibble::tibble(
    Predictor = predictor_label,
    beta = fit_row$estimate,
    SE = fit_row$std.error,
    t = fit_row$statistic,
    p = fit_row$p.value,
    `Lower 95% CI` = fit_row$conf.low,
    `Upper 95% CI` = fit_row$conf.high
  )
}

run_focus_moderation <- function(data, predictor_var, moderator_var, predictor_label, moderator_label, baseline_var_map) {
  baseline_var <- unname(baseline_var_map[[predictor_var]])
  rhs_terms <- c("predictor_z * moderator_z", "group", demographic_covariates)
  select_vars <- c("group", "eco_choice_change", "predictor_value", "moderator_value", demographic_covariates)

  if (!is.na(baseline_var) && nzchar(baseline_var)) {
    rhs_terms <- c(rhs_terms, baseline_var)
    select_vars <- c(select_vars, baseline_var)
  }

  analysis_df <- data %>%
    transmute(
      group = droplevels(group),
      eco_choice_change = eco_choice_change,
      predictor_value = .data[[predictor_var]],
      moderator_value = .data[[moderator_var]],
      Age = Age,
      Sex = Sex,
      ethnicity = ethnicity,
      occupation = occupation,
      income = income,
      education = education,
      envAttitude_pre = envAttitude_pre,
      envMotivation_pre = envMotivation_pre,
      envSelfEfficacy_pre = envSelfEfficacy_pre
    ) %>%
    select(all_of(unique(select_vars))) %>%
    tidyr::drop_na()

  if (
    nrow(analysis_df) < 15 ||
      n_distinct(analysis_df$group) < 2 ||
      n_distinct(analysis_df$eco_choice_change) < 2 ||
      n_distinct(analysis_df$predictor_value) < 2 ||
      n_distinct(analysis_df$moderator_value) < 2
  ) {
    return(
      tibble::tibble(
        Term = paste(predictor_label, "x", moderator_label),
        beta = NA_real_,
        SE = NA_real_,
        t = NA_real_,
        p = NA_real_,
        `Lower 95% CI` = NA_real_,
        `Upper 95% CI` = NA_real_
      )
    )
  }

  analysis_df <- analysis_df %>%
    mutate(
      eco_choice_change_z = as.numeric(scale(eco_choice_change)),
      predictor_z = as.numeric(scale(predictor_value)),
      moderator_z = as.numeric(scale(moderator_value))
    )

  fit <- lm(
    stats::reformulate(rhs_terms, response = "eco_choice_change_z"),
    data = analysis_df
  )

  interaction_row <- broom::tidy(fit, conf.int = TRUE) %>%
    filter(term %in% c("predictor_z:moderator_z", "moderator_z:predictor_z"))

  tibble::tibble(
    Term = paste(predictor_label, "x", moderator_label),
    beta = interaction_row$estimate[1],
    SE = interaction_row$std.error[1],
    t = interaction_row$statistic[1],
    p = interaction_row$p.value[1],
    `Lower 95% CI` = interaction_row$conf.low[1],
    `Upper 95% CI` = interaction_row$conf.high[1]
  )
}

run_focus_rt_mediation <- function(data, mediator_var, mediator_label, baseline_var_map, n_boot = bootstrap_n) {
  baseline_var <- unname(baseline_var_map[[mediator_var]])
  select_vars <- c("group", "mean_nudge_rt", "mediator_value", "eco_choice_change", demographic_covariates)
  fit_a_rhs <- c("mean_nudge_rt", "group", demographic_covariates)
  fit_b_rhs <- c("mean_nudge_rt", "mediator_z", "group", demographic_covariates)
  fit_total_rhs <- c("mean_nudge_rt", "group", demographic_covariates)

  if (!is.na(baseline_var) && nzchar(baseline_var)) {
    select_vars <- c(select_vars, baseline_var)
    fit_a_rhs <- c(fit_a_rhs, baseline_var)
    fit_b_rhs <- c(fit_b_rhs, baseline_var)
    fit_total_rhs <- c(fit_total_rhs, baseline_var)
  }

  analysis_df <- data %>%
    transmute(
      group = droplevels(group),
      mean_nudge_rt = mean_nudge_rt,
      mediator_value = .data[[mediator_var]],
      eco_choice_change = eco_choice_change,
      Age = Age,
      Sex = Sex,
      ethnicity = ethnicity,
      occupation = occupation,
      income = income,
      education = education,
      envAttitude_pre = envAttitude_pre,
      envMotivation_pre = envMotivation_pre,
      envSelfEfficacy_pre = envSelfEfficacy_pre
    ) %>%
    select(all_of(unique(select_vars))) %>%
    tidyr::drop_na()

  if (
    nrow(analysis_df) < 15 ||
      n_distinct(analysis_df$group) < 2 ||
      n_distinct(analysis_df$mean_nudge_rt) < 2 ||
      n_distinct(analysis_df$mediator_value) < 2 ||
      n_distinct(analysis_df$eco_choice_change) < 2
  ) {
    return(
      tibble::tibble(
        Mediator = mediator_label,
        `Path a` = NA_real_,
        `Path b` = NA_real_,
        `Indirect effect` = NA_real_,
        p = NA_real_,
        `Lower 95% CI` = NA_real_,
        `Upper 95% CI` = NA_real_,
        `Direct effect` = NA_real_,
        `Total effect` = NA_real_,
        note = "Model not fitted because of insufficient complete cases or variation."
      )
    )
  }

  analysis_df <- analysis_df %>%
    mutate(
      mean_nudge_rt = as.numeric(scale(mean_nudge_rt)),
      mediator_z = as.numeric(scale(mediator_value)),
      eco_choice_change_z = as.numeric(scale(eco_choice_change))
    )

  fit_a <- lm(
    stats::reformulate(fit_a_rhs, response = "mediator_z"),
    data = analysis_df
  )
  fit_b <- lm(
    stats::reformulate(fit_b_rhs, response = "eco_choice_change_z"),
    data = analysis_df
  )
  fit_total <- lm(
    stats::reformulate(fit_total_rhs, response = "eco_choice_change_z"),
    data = analysis_df
  )

  coef_a <- summary(fit_a)$coefficients
  coef_b <- summary(fit_b)$coefficients
  coef_total <- summary(fit_total)$coefficients
  indirect_effect <- unname(coef(fit_a)[["mean_nudge_rt"]] * coef(fit_b)[["mediator_z"]])

  set.seed(20260419 + sum(utf8ToInt(mediator_var)))
  boot_indirect <- replicate(n_boot, {
    boot_idx <- sample.int(nrow(analysis_df), replace = TRUE)
    boot_df <- analysis_df[boot_idx, , drop = FALSE]
    boot_df$group <- droplevels(boot_df$group)

    if (
      n_distinct(boot_df$group) < 2 ||
        n_distinct(boot_df$mean_nudge_rt) < 2 ||
        n_distinct(boot_df$mediator_z) < 2 ||
        n_distinct(boot_df$eco_choice_change_z) < 2
    ) {
      return(NA_real_)
    }

    fit_a_boot <- try(
      lm(
        stats::reformulate(fit_a_rhs, response = "mediator_z"),
        data = boot_df
      ),
      silent = TRUE
    )
    fit_b_boot <- try(
      lm(
        stats::reformulate(fit_b_rhs, response = "eco_choice_change_z"),
        data = boot_df
      ),
      silent = TRUE
    )

    if (inherits(fit_a_boot, "try-error") || inherits(fit_b_boot, "try-error")) {
      return(NA_real_)
    }

    unname(coef(fit_a_boot)[["mean_nudge_rt"]] * coef(fit_b_boot)[["mediator_z"]])
  })

  boot_indirect <- boot_indirect[is.finite(boot_indirect)]

  if (length(boot_indirect) < 100) {
    return(
      tibble::tibble(
        Mediator = mediator_label,
        `Path a` = coef_a["mean_nudge_rt", "Estimate"],
        `Path b` = coef_b["mediator_z", "Estimate"],
        `Indirect effect` = indirect_effect,
        p = NA_real_,
        `Lower 95% CI` = NA_real_,
        `Upper 95% CI` = NA_real_,
        `Direct effect` = coef_b["mean_nudge_rt", "Estimate"],
        `Total effect` = coef_total["mean_nudge_rt", "Estimate"],
        note = "Indirect-effect bootstrap CI was not computed because too few valid resamples were available."
      )
    )
  }

  indirect_ci <- quantile(boot_indirect, probs = c(0.025, 0.975), na.rm = TRUE)
  indirect_p <- 2 * min(mean(boot_indirect >= 0), mean(boot_indirect <= 0))

  tibble::tibble(
    Mediator = mediator_label,
    `Path a` = coef_a["mean_nudge_rt", "Estimate"],
    `Path b` = coef_b["mediator_z", "Estimate"],
    `Indirect effect` = indirect_effect,
    p = indirect_p,
    `Lower 95% CI` = unname(indirect_ci[1]),
    `Upper 95% CI` = unname(indirect_ci[2]),
    `Direct effect` = coef_b["mean_nudge_rt", "Estimate"],
    `Total effect` = coef_total["mean_nudge_rt", "Estimate"],
    note = ""
  )
}

run_standardized_rt_regression <- function(data, outcome_var, outcome_label, baseline_var_map) {
  baseline_var <- unname(baseline_var_map[[outcome_var]])
  rhs_terms <- c("mean_nudge_rt_z", demographic_covariates)
  select_vars <- c("mean_nudge_rt", "outcome_value", demographic_covariates)

  if (!is.na(baseline_var) && nzchar(baseline_var)) {
    rhs_terms <- c(rhs_terms, baseline_var)
    select_vars <- c(select_vars, baseline_var)
  }

  analysis_df <- data %>%
    transmute(
      mean_nudge_rt = mean_nudge_rt,
      outcome_value = .data[[outcome_var]],
      Age = Age,
      Sex = Sex,
      ethnicity = ethnicity,
      occupation = occupation,
      income = income,
      education = education,
      envAttitude_pre = envAttitude_pre,
      envMotivation_pre = envMotivation_pre,
      envSelfEfficacy_pre = envSelfEfficacy_pre
    ) %>%
    select(all_of(unique(select_vars))) %>%
    tidyr::drop_na()

  if (nrow(analysis_df) < 3 || n_distinct(analysis_df$mean_nudge_rt) < 2 || n_distinct(analysis_df$outcome_value) < 2) {
    return(
      tibble::tibble(
        Outcome = outcome_label,
        estimate = NA_real_,
        SE = NA_real_,
        t = NA_real_,
        p = NA_real_,
        `Lower 95% CI` = NA_real_,
        `Upper 95% CI` = NA_real_,
        note = "Model not fitted because of insufficient complete cases or variation."
      )
    )
  }

  analysis_df <- analysis_df %>%
    mutate(
      outcome_z = as.numeric(scale(outcome_value)),
      mean_nudge_rt_z = as.numeric(scale(mean_nudge_rt))
    )

  fit_rt <- lm(
    stats::reformulate(rhs_terms, response = "outcome_z"),
    data = analysis_df
  )
  fit_row <- broom::tidy(fit_rt, conf.int = TRUE) %>%
    filter(term == "mean_nudge_rt_z")

  tibble::tibble(
    Outcome = outcome_label,
    estimate = fit_row$estimate,
    SE = fit_row$std.error,
    t = fit_row$statistic,
    p = fit_row$p.value,
    `Lower 95% CI` = fit_row$conf.low,
    `Upper 95% CI` = fit_row$conf.high,
    note = ""
  )
}

script_dir <- get_script_dir()
data_dir <- normalizePath(file.path(script_dir, "..", "data"), winslash = "/", mustWork = FALSE)
human_data_path <- file.path(data_dir, "data_human_deidentified.csv")
simulation_data_path <- file.path(data_dir, "data_simulation_deidentified.csv")
output_dir <- file.path(script_dir, "output")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

if (!file.exists(human_data_path)) {
  stop(paste0("data_human_deidentified.csv was not found at: ", human_data_path), call. = FALSE)
}

if (!file.exists(simulation_data_path)) {
  stop(paste0("data_simulation_deidentified.csv was not found at: ", simulation_data_path), call. = FALSE)
}

df_raw <- as.data.frame(
  readr::read_csv(human_data_path, show_col_types = FALSE, locale = readr::locale(encoding = "UTF-8")),
  check.names = FALSE
) %>%
  clean_csv_names()

simulation_human_wide <- prepare_simulation_validity_wide(df_raw, "human data")
simulation_wide <- read_simulation_validity_wide(simulation_data_path, "simulation data")

required_columns <- c(
  "ID", "group", "Age", "Sex", "ethnicity", "occupation", "income", "education",
  "envAttitude_pre", "envAttitude_post",
  "envMotivation_pre", "envMotivation_post",
  "envSelfEfficacy_pre", "envSelfEfficacy_post",
  "nudge_acceptance_post", "nudge_reactance_post", "nudge_fatigue_post",
  "eco_choices_day1", "eco_choices_day14",
  "daily_donation_cents_day1", "daily_donation_cents_day14",
  "n_trials_day1", "n_trials_day14"
)

missing_columns <- setdiff(required_columns, names(df_raw))
if (length(missing_columns) > 0) {
  stop(
    paste0("Missing required columns: ", paste(missing_columns, collapse = ", ")),
    call. = FALSE
  )
}

df_base <- df_raw %>%
  mutate(
    across(
      matches("^(Age|env|nudge_acceptance_post|nudge_reactance_post|nudge_fatigue_post|eco_choices_day|daily_donation_cents_day|n_trials_day|eco_share_day|information_nudge_rt_ms_day|assistant_commitment_rt_ms_day)"),
      ~ suppressWarnings(as.numeric(.x))
    ),
    ID = factor(ID),
    group = factor(group, levels = group_levels)
  ) %>%
  select(-any_of(c("nudgeType", "nudgeIntensity")))

group_parts <- tibble::tibble(group_code = as.character(df_base$group)) %>%
  tidyr::separate(group_code, into = c("nudgeType", "nudgeIntensity"), sep = "_", fill = "right")

df <- bind_cols(df_base, group_parts) %>%
  mutate(
    nudgeIntensity = tidyr::replace_na(nudgeIntensity, "control"),
    Sex = factor(trimws(as.character(Sex)), levels = sex_levels),
    ethnicity = factor(trimws(as.character(ethnicity)), levels = ethnicity_levels),
    occupation = coerce_labeled_factor(occupation, occupation_levels, "occupation"),
    income = coerce_labeled_factor(income, income_levels, "income"),
    education = coerce_labeled_factor(education, education_levels, "education"),
    envAttitude_pre = pmin(pmax(envAttitude_pre, 1), 5),
    envAttitude_post = pmin(pmax(envAttitude_post, 1), 5),
    delta_attitude = envAttitude_post - envAttitude_pre,
    delta_motivation = envMotivation_post - envMotivation_pre,
    delta_self_efficacy = envSelfEfficacy_post - envSelfEfficacy_pre,
    eco_choice_change = eco_choices_day14 - eco_choices_day1
  )

day_long <- build_day_long(df) %>%
  filter(!is.na(n_trials), n_trials > 0, !is.na(eco_share))

donation_day_long <- day_long %>%
  filter(!is.na(donation_cents))

df_nudge <- df %>%
  filter(as.character(group) != "control") %>%
  mutate(
    nudgeType = factor(nudgeType, levels = c("information", "structure", "assistant")),
    nudgeIntensity = factor(nudgeIntensity, levels = c("low", "medium", "high"))
  )

sample_overview <- tibble::tibble(
  Participants = dplyr::n_distinct(df$ID),
  `Person-Days` = nrow(donation_day_long),
  `Total Choices` = sum(donation_day_long$n_trials, na.rm = TRUE),
  `Total Donations (Cents)` = sum(donation_day_long$donation_cents, na.rm = TRUE)
)

trajectory_fit <- lme4::lmer(
  eco_share ~ (day - 1) * group +
    Age + Sex + ethnicity + occupation + income + education +
    envAttitude_pre + envSelfEfficacy_pre + envMotivation_pre +
    (1 | ID),
  data = day_long,
  REML = FALSE,
  control = lme4::lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
)

trajectory_emm <- emmeans::emmeans(
  trajectory_fit,
  ~ group | day,
  at = list(day = c(1, 14)),
  weights = "proportional"
)

day1_vs_control <- as.data.frame(
  summary(
    contrast(trajectory_emm, method = "trt.vs.ctrl", ref = "control", by = "day", adjust = "BH"),
    infer = TRUE
  )
) %>%
  filter(day == 1) %>%
  transmute(
    Contrast = prettify_group_text(contrast),
    estimate = estimate,
    SE = SE,
    df = df,
    `Lower 95% CI` = lower.CL,
    `Upper 95% CI` = upper.CL,
    t = t.ratio,
    p = p.value
  )

day14_vs_control <- as.data.frame(
  summary(
    contrast(trajectory_emm, method = "trt.vs.ctrl", ref = "control", by = "day", adjust = "BH"),
    infer = TRUE
  )
) %>%
  filter(day == 14) %>%
  transmute(
    Contrast = prettify_group_text(contrast),
    estimate = estimate,
    SE = SE,
    df = df,
    `Lower 95% CI` = lower.CL,
    `Upper 95% CI` = upper.CL,
    t = t.ratio,
    p = p.value
  )

within_group_change <- as.data.frame(
  summary(
    contrast(
      emmeans::emmeans(
        trajectory_fit,
        ~ day | group,
        at = list(day = c(1, 14)),
        weights = "proportional"
      ),
      method = list("Day 14 - Day 1" = c(-1, 1)),
      by = "group",
      adjust = "BH"
    ),
    infer = TRUE
  )
) %>%
  transmute(
    Contrast = contrast,
    Group = label_group(group),
    estimate = estimate,
    SE = SE,
    df = df,
    `Lower 95% CI` = lower.CL,
    `Upper 95% CI` = upper.CL,
    t = t.ratio,
    p = p.value
  )

group_daily_fit <- day_long %>%
  group_by(group, day) %>%
  summarise(y = mean(eco_share, na.rm = TRUE), .groups = "drop") %>%
  arrange(group, day)

group_split_fit <- split(group_daily_fit, group_daily_fit$group)
fit_compare_list <- list()

for (group_name in names(group_split_fit)) {
  group_df <- group_split_fit[[group_name]]
  group_df <- group_df[!is.na(group_df$y), ]

  y0_start <- group_df$y[group_df$day == 1][1]
  if (is.na(y0_start)) {
    y0_start <- group_df$y[1]
  }

  asym_start <- min(group_df$y, na.rm = TRUE)
  if (!is.finite(asym_start)) {
    asym_start <- 0
  }

  fit_nl <- NULL
  for (k0 in c(0.05, 0.10, 0.20, 0.30, 0.50, 0.80)) {
    fit_nl <- tryCatch(
      minpack.lm::nlsLM(
        y ~ asym + (y0 - asym) * exp(-k * (day - 1)),
        data = group_df,
        start = list(y0 = y0_start, asym = asym_start, k = k0),
        lower = c(y0 = 0, asym = 0, k = 0.000001),
        upper = c(y0 = 1, asym = 1, k = 10),
        control = minpack.lm::nls.lm.control(maxiter = 500)
      ),
      error = function(e) NULL
    )
    if (!is.null(fit_nl)) {
      break
    }
  }

  fit_lin <- lm(y ~ day, data = group_df)
  pred_lin <- broom::augment(fit_lin, newdata = group_df)
  r2_lin <- calc_fit_r2(group_df$y, pred_lin)

  if (!is.null(fit_nl)) {
    pred_nl <- broom::augment(fit_nl, newdata = group_df)
    coef_nl <- coef(fit_nl)
    r2_nl <- calc_fit_r2(group_df$y, pred_nl)
    auc_nl <- calc_auc(pred_nl$day, pred_nl$.fitted)
    study_length <- max(group_df$day, na.rm = TRUE) - 1
    mde_nl <- dplyr::case_when(
      is.na(coef_nl["y0"]) | is.na(coef_nl["asym"]) | is.na(coef_nl["k"]) | study_length <= 0 ~ NA_real_,
      abs(coef_nl["k"]) < 0.00000001 ~ unname(coef_nl["y0"]),
      TRUE ~ unname(
        coef_nl["asym"] +
          ((coef_nl["y0"] - coef_nl["asym"]) / (coef_nl["k"] * study_length)) *
          (1 - exp(-coef_nl["k"] * study_length))
      )
    )
    pr_nl <- dplyr::case_when(
      is.na(coef_nl["y0"]) | is.na(coef_nl["asym"]) | coef_nl["y0"] == 0 ~ NA_real_,
      TRUE ~ unname(coef_nl["asym"] / coef_nl["y0"])
    )
  } else {
    coef_nl <- c(y0 = NA_real_, asym = NA_real_, k = NA_real_)
    r2_nl <- NA_real_
    auc_nl <- NA_real_
    mde_nl <- NA_real_
    pr_nl <- NA_real_
  }

  fit_compare_list[[group_name]] <- tibble::tibble(
    Group = label_group(group_name),
    y_0 = unname(coef_nl["y0"]),
    asym = unname(coef_nl["asym"]),
    k = unname(coef_nl["k"]),
    AUC = auc_nl,
    MDE = mde_nl,
    PR = pr_nl,
    R2_nl = r2_nl,
    R2_lin = r2_lin,
    `Delta R2` = ifelse(abs(r2_nl - r2_lin) < 0.0000001, 0, r2_nl - r2_lin)
  )
}

fit_compare_df <- bind_rows(fit_compare_list)

event_window_choice_df <- day_long %>%
  filter(as.character(nudgeIntensity) == "medium", as.character(nudgeType) %in% c("information", "structure", "assistant")) %>%
  filter(!is.na(event_window)) %>%
  mutate(nudgeType = factor(nudgeType, levels = c("information", "structure", "assistant")))

event_window_choice_fit <- lme4::lmer(
  eco_share ~ event_window * nudgeType +
    Age + Sex + ethnicity + occupation + income + education +
    envAttitude_pre + envSelfEfficacy_pre + envMotivation_pre +
    (1 | ID),
  data = event_window_choice_df,
  REML = FALSE,
  control = lme4::lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
)

event_window_choice_contrasts <- as.data.frame(
  summary(
    contrast(
      emmeans(event_window_choice_fit, ~ event_window | nudgeType, weights = "proportional"),
      method = list(
        "NudgeDay - PostDay1" = c(1, -1, 0),
        "NudgeDay - PostDay2" = c(1, 0, -1)
      ),
      by = "nudgeType",
      adjust = "BH"
    ),
    infer = TRUE
  )
) %>%
  transmute(
    Contrast = contrast,
    `Nudge Type` = label_nudge_type(nudgeType),
    estimate = estimate,
    SE = SE,
    df = df,
    `Lower 95% CI` = lower.CL,
    `Upper 95% CI` = upper.CL,
    t = t.ratio,
    p = p.value
  )

donation_frequency_means <- donation_day_long %>%
  group_by(frequency4) %>%
  summarise(
    `Mean Daily Donation (Cents)` = mean(donation_cents, na.rm = TRUE),
    `Person-Days` = n(),
    .groups = "drop"
  ) %>%
  transmute(
    `Frequency Condition` = label_frequency_condition(frequency4),
    `Mean Daily Donation (Cents)` = `Mean Daily Donation (Cents)`,
    `Person-Days` = as.character(`Person-Days`)
  )

donation_frequency_fit <- donation_fit_poisson_model(
  donation_day_long,
  donation_cents ~ frequency4 + factor(day) + Age + Sex + ethnicity + occupation + income + education +
    envAttitude_pre + envSelfEfficacy_pre + envMotivation_pre
)

donation_frequency_rr <- bind_rows(
  donation_linear_combo_model(donation_frequency_fit, c("frequency4low" = 1), "low vs control", exponentiate = TRUE),
  donation_linear_combo_model(donation_frequency_fit, c("frequency4medium" = 1), "medium vs control", exponentiate = TRUE),
  donation_linear_combo_model(donation_frequency_fit, c("frequency4high" = 1), "high vs control", exponentiate = TRUE)
) %>%
  transmute(
    Contrast = label_frequency_contrast(contrast),
    `Rate Ratio` = estimate,
    `Lower 95% CI` = conf_low,
    `Upper 95% CI` = conf_high,
    p = p_value
  )

donation_frequency_pairwise <- bind_rows(
  donation_linear_combo_model(donation_frequency_fit, c("frequency4medium" = 1, "frequency4low" = -1), "medium vs low", exponentiate = TRUE),
  donation_linear_combo_model(donation_frequency_fit, c("frequency4high" = 1, "frequency4low" = -1), "high vs low", exponentiate = TRUE),
  donation_linear_combo_model(donation_frequency_fit, c("frequency4high" = 1, "frequency4medium" = -1), "high vs medium", exponentiate = TRUE)
) %>%
  transmute(
    Contrast = label_frequency_contrast(contrast),
    `Rate Ratio` = estimate,
    `Lower 95% CI` = conf_low,
    `Upper 95% CI` = conf_high,
    p = p_value
  )

structure_donation_df <- donation_day_long %>%
  filter(as.character(group) == "control" | as.character(nudgeType) == "structure") %>%
  filter(!is.na(structure_state))

donation_structure_means <- structure_donation_df %>%
  group_by(structure_state) %>%
  summarise(
    `Mean Daily Donation (Cents)` = mean(donation_cents, na.rm = TRUE),
    `Person-Days` = n(),
    .groups = "drop"
  ) %>%
  transmute(
    `Structural Cue State` = label_structure_state(structure_state),
    `Mean Daily Donation (Cents)` = `Mean Daily Donation (Cents)`,
    `Person-Days` = as.character(`Person-Days`)
  )

donation_structure_fit <- donation_fit_poisson_model(
  structure_donation_df,
  donation_cents ~ structure_state + factor(day) + Age + Sex + ethnicity + occupation + income + education +
    envAttitude_pre + envSelfEfficacy_pre + envMotivation_pre
)

donation_structure_rr <- bind_rows(
  donation_linear_combo_model(donation_structure_fit, c("structure_statestructure_off" = 1), "structure off vs control", exponentiate = TRUE),
  donation_linear_combo_model(donation_structure_fit, c("structure_statestructure_active" = 1), "structure active vs control", exponentiate = TRUE),
  donation_linear_combo_model(
    donation_structure_fit,
    c("structure_statestructure_active" = 1, "structure_statestructure_off" = -1),
    "structure active vs structure off",
    exponentiate = TRUE
  )
) %>%
  transmute(
    Contrast = label_structure_contrast(contrast),
    `Rate Ratio` = estimate,
    `Lower 95% CI` = conf_low,
    `Upper 95% CI` = conf_high,
    p = p_value
  )

donation_event_window_df <- donation_day_long %>%
  filter(as.character(nudgeIntensity) == "medium", as.character(nudgeType) %in% c("information", "structure", "assistant")) %>%
  filter(!is.na(event_window))

donation_event_window_fit <- lme4::lmer(
  donation_cents ~ event_window * nudgeType +
    Age + Sex + ethnicity + occupation + income + education +
    envAttitude_pre + envSelfEfficacy_pre + envMotivation_pre +
    (1 | ID),
  data = donation_event_window_df,
  REML = FALSE,
  control = lme4::lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
)

donation_event_window_emm <- emmeans::emmeans(donation_event_window_fit, ~ event_window | nudgeType)

donation_event_window_adjusted <- as.data.frame(
  summary(donation_event_window_emm, infer = TRUE)
)

donation_event_window_contrasts <- as.data.frame(
  summary(
    contrast(
      donation_event_window_emm,
      method = list(
        "Nudge Day vs Post Day 1" = c(1, -1, 0),
        "Nudge Day vs Post Day 2" = c(1, 0, -1)
      ),
      by = "nudgeType",
      adjust = "BH"
    ),
    infer = TRUE
  )
) %>%
  transmute(
    Contrast = contrast,
    `Nudge Type` = label_nudge_type(nudgeType),
    Estimate = estimate,
    SE = SE,
    df = df,
    `Lower 95% CI` = lower.CL,
    `Upper 95% CI` = upper.CL,
    t = t.ratio,
    p = p.value
  )

donation_any_fit <- donation_fit_logistic_model(
  donation_day_long,
  any_donation ~ frequency4 + factor(day) + Age + Sex + ethnicity + occupation + income + education +
    envAttitude_pre + envSelfEfficacy_pre + envMotivation_pre
)

donation_positive_fit <- donation_fit_positive_amount_model(
  donation_day_long %>% filter(!is.na(positive_log_donation)),
  positive_log_donation ~ frequency4 + factor(day) + Age + Sex + ethnicity + occupation + income + education +
    envAttitude_pre + envSelfEfficacy_pre + envMotivation_pre
)

donation_hurdle_binary <- bind_rows(
  donation_linear_combo_model(donation_any_fit, c("frequency4low" = 1), "low vs control", exponentiate = TRUE),
  donation_linear_combo_model(donation_any_fit, c("frequency4medium" = 1), "medium vs control", exponentiate = TRUE),
  donation_linear_combo_model(donation_any_fit, c("frequency4high" = 1), "high vs control", exponentiate = TRUE)
) %>%
  transmute(
    Contrast = label_frequency_contrast(contrast),
    OR = estimate,
    `Lower 95% CI` = conf_low,
    `Upper 95% CI` = conf_high,
    p = p_value
  )

donation_hurdle_positive <- bind_rows(
  donation_linear_combo_model(donation_positive_fit, c("frequency4low" = 1), "low vs control", exponentiate = TRUE),
  donation_linear_combo_model(donation_positive_fit, c("frequency4medium" = 1), "medium vs control", exponentiate = TRUE),
  donation_linear_combo_model(donation_positive_fit, c("frequency4high" = 1), "high vs control", exponentiate = TRUE)
) %>%
  transmute(
    Contrast = label_frequency_contrast(contrast),
    estimate = estimate,
    `Lower 95% CI` = conf_low,
    `Upper 95% CI` = conf_high,
    p = p_value
  )

attitude_long <- df %>%
  select(ID, group, envAttitude_pre, envAttitude_post) %>%
  pivot_longer(
    cols = c(envAttitude_pre, envAttitude_post),
    names_to = "time",
    values_to = "value"
  ) %>%
  mutate(time = factor(time, levels = c("envAttitude_pre", "envAttitude_post"), labels = c("Pre", "Post")))

motivation_long <- df %>%
  select(ID, group, envMotivation_pre, envMotivation_post) %>%
  pivot_longer(
    cols = c(envMotivation_pre, envMotivation_post),
    names_to = "time",
    values_to = "value"
  ) %>%
  mutate(time = factor(time, levels = c("envMotivation_pre", "envMotivation_post"), labels = c("Pre", "Post")))

self_efficacy_long <- df %>%
  select(ID, group, envSelfEfficacy_pre, envSelfEfficacy_post) %>%
  pivot_longer(
    cols = c(envSelfEfficacy_pre, envSelfEfficacy_post),
    names_to = "time",
    values_to = "value"
  ) %>%
  mutate(time = factor(time, levels = c("envSelfEfficacy_pre", "envSelfEfficacy_post"), labels = c("Pre", "Post")))

env_desc_table <- df %>%
  select(
    envAttitude_post, envAttitude_pre,
    envMotivation_post, envMotivation_pre,
    envSelfEfficacy_post, envSelfEfficacy_pre
  ) %>%
  pivot_longer(
    cols = everything(),
    names_to = "Variable",
    values_to = "value"
  ) %>%
  mutate(
    Variable = dplyr::recode(
      Variable,
      "envAttitude_post" = "Environmental attitude (Post)",
      "envAttitude_pre" = "Environmental attitude (Pre)",
      "envMotivation_post" = "Environmental autonomous motivation (Post)",
      "envMotivation_pre" = "Environmental autonomous motivation (Pre)",
      "envSelfEfficacy_post" = "Environmental self-efficacy (Post)",
      "envSelfEfficacy_pre" = "Environmental self-efficacy (Pre)"
    )
  ) %>%
  group_by(Variable) %>%
  summarise(
    Mean = mean(value, na.rm = TRUE),
    SD = sd(value, na.rm = TRUE),
    Min = min(value, na.rm = TRUE),
    Max = max(value, na.rm = TRUE),
    .groups = "drop"
  )

attitude_aov <- suppressWarnings(suppressMessages(afex::aov_ez(
  id = "ID",
  dv = "value",
  data = attitude_long,
  within = "time",
  between = "group",
  anova_table = list(es = "pes")
)))
motivation_aov <- suppressWarnings(suppressMessages(afex::aov_ez(
  id = "ID",
  dv = "value",
  data = motivation_long,
  within = "time",
  between = "group",
  anova_table = list(es = "pes")
)))
self_efficacy_aov <- suppressWarnings(suppressMessages(afex::aov_ez(
  id = "ID",
  dv = "value",
  data = self_efficacy_long,
  within = "time",
  between = "group",
  anova_table = list(es = "pes")
)))

build_env_anova_summary <- function(aov_obj, variable_label) {
  as.data.frame(aov_obj$anova_table) %>%
    tibble::rownames_to_column("Effect") %>%
    filter(Effect %in% c("group", "time", "group:time")) %>%
    transmute(
      Variable = variable_label,
      Effect = Effect,
      `num Df` = as.character(as.integer(round(`num Df`))),
      `den Df` = as.character(as.integer(round(`den Df`))),
      MSE = MSE,
      F = F,
      p = `Pr(>F)`
    )
}

build_env_prepost_summary <- function(pair_table, variable_label) {
  pair_table %>%
    mutate(
      lower.CL_orig = .data[["lower.CL"]],
      upper.CL_orig = .data[["upper.CL"]]
    ) %>%
    transmute(
      Variable = variable_label,
      Contrast = "post - pre",
      Group = label_group(group),
      `Mean diff` = -.data[["estimate"]],
      SE = SE,
      df = as.character(as.integer(round(df))),
      t = -.data[["t.ratio"]],
      p_adj = .data[["p.value"]],
      `Lower 95% CI` = -.data[["upper.CL_orig"]],
      `Upper 95% CI` = -.data[["lower.CL_orig"]]
    )
}

env_anova_summary_table <- bind_rows(
  build_env_anova_summary(attitude_aov, "Attitude"),
  build_env_anova_summary(motivation_aov, "Motivation"),
  build_env_anova_summary(self_efficacy_aov, "Self-efficacy")
)

env_prepost_summary_table <- bind_rows(
  build_env_prepost_summary(as_tibble(summary(pairs(emmeans(attitude_aov, ~ time | group), adjust = "BH"), infer = TRUE)), "Attitude"),
  build_env_prepost_summary(as_tibble(summary(pairs(emmeans(motivation_aov, ~ time | group), adjust = "BH"), infer = TRUE)), "Motivation"),
  build_env_prepost_summary(as_tibble(summary(pairs(emmeans(self_efficacy_aov, ~ time | group), adjust = "BH"), infer = TRUE)), "Self-efficacy")
)

nudge_feedback_long <- df_nudge %>%
  select(group, nudgeType, nudgeIntensity, nudge_acceptance_post, nudge_reactance_post, nudge_fatigue_post) %>%
  pivot_longer(
    cols = c(nudge_acceptance_post, nudge_reactance_post, nudge_fatigue_post),
    names_to = "outcome",
    values_to = "value"
  ) %>%
  mutate(
    outcome = dplyr::recode(
      outcome,
      "nudge_acceptance_post" = "Nudge acceptance",
      "nudge_reactance_post" = "Nudge reactance",
      "nudge_fatigue_post" = "Nudge fatigue"
    )
  )

nudge_feedback_desc_overall <- nudge_feedback_long %>%
  group_by(outcome) %>%
  summarise(
    Mean = mean(value, na.rm = TRUE),
    SD = sd(value, na.rm = TRUE),
    Min = min(value, na.rm = TRUE),
    Max = max(value, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  rename(Outcome = outcome)

fit_acceptance <- lm(nudge_acceptance_post ~ nudgeType * nudgeIntensity, data = df_nudge)
fit_reactance <- lm(nudge_reactance_post ~ nudgeType * nudgeIntensity, data = df_nudge)
fit_fatigue <- lm(nudge_fatigue_post ~ nudgeType * nudgeIntensity, data = df_nudge)

build_nudge_feedback_anova_summary <- function(fit_obj, outcome_label) {
  anova_df <- as.data.frame(car::Anova(fit_obj, type = 3)) %>%
    tibble::rownames_to_column("Effect")

  anova_df %>%
    filter(Effect %in% c("nudgeType", "nudgeIntensity", "nudgeType:nudgeIntensity")) %>%
    transmute(
      Variable = outcome_label,
      Predictor = dplyr::recode(
        Effect,
        "nudgeType" = "Nudge type",
        "nudgeIntensity" = "Nudge intensity",
        "nudgeType:nudgeIntensity" = "Nudge type x Nudge intensity"
      ),
      Df = as.character(as.integer(round(Df))),
      F = `F value`,
      p = `Pr(>F)`
    )
}

nudge_feedback_anova <- bind_rows(
  build_nudge_feedback_anova_summary(fit_acceptance, "Nudge acceptance"),
  build_nudge_feedback_anova_summary(fit_reactance, "Nudge reactance"),
  build_nudge_feedback_anova_summary(fit_fatigue, "Nudge fatigue")
)

nudge_feedback_pairs <- bind_rows(
  summary(pairs(emmeans(fit_acceptance, ~ nudgeIntensity | nudgeType), adjust = "BH"), infer = TRUE) %>%
    as_tibble() %>%
    mutate(Outcome = "Nudge acceptance"),
  summary(pairs(emmeans(fit_reactance, ~ nudgeIntensity | nudgeType), adjust = "BH"), infer = TRUE) %>%
    as_tibble() %>%
    mutate(Outcome = "Nudge reactance"),
  summary(pairs(emmeans(fit_fatigue, ~ nudgeIntensity | nudgeType), adjust = "BH"), infer = TRUE) %>%
    as_tibble() %>%
    mutate(Outcome = "Nudge fatigue")
) %>%
  transmute(
    Contrast = contrast,
    nudgeType = label_nudge_type(nudgeType),
    estimate = estimate,
    SE = SE,
    df = as.character(as.integer(round(df))),
    `Lower 95% CI` = lower.CL,
    `Upper 95% CI` = upper.CL,
    t = t.ratio,
    p = p.value,
    Outcome = Outcome
  )

focus_groups <- c(
  "information_medium", "information_high",
  "assistant_medium", "assistant_high"
)
focus_group_labels <- c(
  "Information medium", "Information high",
  "Assistant medium", "Assistant high"
)
focus_type_groups <- c(
  "information_medium", "information_high",
  "structure_medium", "structure_high",
  "assistant_medium", "assistant_high"
)
focus_type_group_labels <- c(
  "Information medium", "Information high",
  "Structure medium", "Structure high",
  "Assistant medium", "Assistant high"
)

focus_main_effect_specs <- tibble::tibble(
  Predictor = c(
    "delta_attitude",
    "delta_motivation",
    "delta_self_efficacy",
    "nudge_acceptance_post"
  ),
  Predictor_label = c(
    "Delta attitude",
    "Delta motivation",
    "Delta self-efficacy",
    "Nudge acceptance"
  )
)

focus_baseline_var_map <- c(
  "delta_attitude" = "envAttitude_pre",
  "delta_motivation" = "envMotivation_pre",
  "delta_self_efficacy" = "envSelfEfficacy_pre",
  "nudge_acceptance_post" = NA_character_
)

df_focus_nudge <- df_nudge %>%
  filter(as.character(group) %in% focus_groups) %>%
  mutate(
    nudgeType = factor(as.character(nudgeType), levels = c("information", "assistant"), labels = c("Information", "Assistant")),
    group = factor(as.character(group), levels = focus_groups, labels = focus_group_labels)
  ) %>%
  add_mean_nudge_rt(set_structure_zero = FALSE)

df_focus_type_nudge <- df_nudge %>%
  filter(as.character(group) %in% focus_type_groups) %>%
  mutate(
    nudgeType = factor(as.character(nudgeType), levels = c("information", "structure", "assistant"), labels = c("Information", "Structure", "Assistant")),
    group = factor(as.character(group), levels = focus_type_groups, labels = focus_type_group_labels)
  )

focus_main_effect_table <- purrr::pmap_dfr(
  focus_main_effect_specs,
  function(Predictor, Predictor_label) {
    run_focus_main_effect(df_focus_nudge, Predictor, Predictor_label, focus_baseline_var_map)
  }
)

focus_main_effect_by_type_table <- df_focus_type_nudge %>%
  group_by(nudgeType) %>%
  group_modify(~ {
    purrr::pmap_dfr(
      focus_main_effect_specs,
      function(Predictor, Predictor_label) {
        run_focus_main_effect(.x, Predictor, Predictor_label, focus_baseline_var_map)
      }
    )
  }) %>%
  ungroup() %>%
  transmute(
    `Nudge type` = as.character(nudgeType),
    Predictor = Predictor,
    beta = beta,
    SE = SE,
    t = t,
    p = p,
    `Lower 95% CI` = `Lower 95% CI`,
    `Upper 95% CI` = `Upper 95% CI`
  )

focus_analysis_specs <- tidyr::expand_grid(
  Predictor = c(
    "delta_attitude",
    "delta_motivation",
    "delta_self_efficacy",
    "nudge_acceptance_post"
  ),
  Moderator = c("nudge_fatigue_post", "nudge_reactance_post")
) %>%
  mutate(
    Predictor_label = label_analysis_var(Predictor),
    Moderator_label = label_analysis_var(Moderator)
  )

focus_moderation_table <- purrr::pmap_dfr(
  focus_analysis_specs,
  function(Predictor, Moderator, Predictor_label, Moderator_label) {
    run_focus_moderation(df_focus_nudge, Predictor, Moderator, Predictor_label, Moderator_label, focus_baseline_var_map)
  }
)

focus_moderation_by_type_table <- df_focus_type_nudge %>%
  group_by(nudgeType) %>%
  group_modify(~ {
    purrr::pmap_dfr(
      focus_analysis_specs,
      function(Predictor, Moderator, Predictor_label, Moderator_label) {
        run_focus_moderation(.x, Predictor, Moderator, Predictor_label, Moderator_label, focus_baseline_var_map)
      }
    )
  }) %>%
  ungroup() %>%
  transmute(
    `Nudge type` = as.character(nudgeType),
    Term = Term,
    beta = beta,
    SE = SE,
    t = t,
    p = p,
    `Lower 95% CI` = `Lower 95% CI`,
    `Upper 95% CI` = `Upper 95% CI`
  )

focus_mediation_table <- purrr::map2_dfr(
  c("delta_attitude", "delta_motivation", "delta_self_efficacy", "nudge_acceptance_post"),
  c("Delta attitude", "Delta motivation", "Delta self-efficacy", "Nudge acceptance"),
  ~ run_focus_rt_mediation(df_focus_nudge, .x, .y, focus_baseline_var_map)
) %>%
  rename(
    `Path a (beta)` = `Path a`,
    `Path b (beta)` = `Path b`,
    `Direct effect (beta)` = `Direct effect`,
    `Total effect (beta)` = `Total effect`
  )

rt_nudge_df <- df_focus_nudge

rt_regression_table <- purrr::map2_dfr(
  c("delta_attitude", "delta_motivation", "delta_self_efficacy", "nudge_acceptance_post"),
  c("Delta attitude", "Delta motivation", "Delta self-efficacy", "Nudge acceptance"),
  ~ run_standardized_rt_regression(rt_nudge_df, .x, .y, focus_baseline_var_map)
)

simulation_paired <- pair_simulation_validity_data(simulation_human_wide, simulation_wide)
simulation_group_day <- make_simulation_group_day(simulation_paired)
simulation_validation_summary <- make_simulation_validation_summary(simulation_group_day)
simulation_condition_table <- make_simulation_condition_table(simulation_group_day)

plot_total_df <- day_long %>%
  group_by(nudgeIntensity, day) %>%
  summarise(mean_eco_share = mean(eco_share, na.rm = TRUE), .groups = "drop") %>%
  mutate(panel = "Total")

plot_treat_df <- day_long %>%
  filter(as.character(nudgeType) %in% c("information", "structure", "assistant")) %>%
  group_by(nudgeType, nudgeIntensity, day) %>%
  summarise(mean_eco_share = mean(eco_share, na.rm = TRUE), .groups = "drop") %>%
  mutate(panel = label_nudge_type(nudgeType))

plot_control_df <- day_long %>%
  filter(as.character(nudgeType) == "control") %>%
  group_by(day) %>%
  summarise(mean_eco_share = mean(eco_share, na.rm = TRUE), .groups = "drop") %>%
  mutate(nudgeIntensity = factor("control", levels = c("control", "low", "medium", "high")))

plot_control_expanded_df <- tidyr::crossing(
  panel = c("Information", "Structure", "Assistant"),
  plot_control_df
)

main_trend_plot_df <- bind_rows(
  plot_total_df,
  plot_treat_df,
  plot_control_expanded_df
) %>%
  mutate(
    panel = factor(panel, levels = c("Total", "Information", "Structure", "Assistant")),
    nudgeIntensity = factor(
      as.character(nudgeIntensity),
      levels = c("low", "medium", "high", "control"),
      labels = c("Low", "Medium", "High", "Control")
    )
  )

main_trend_plot <- ggplot(
  main_trend_plot_df,
  aes(x = day, y = mean_eco_share, color = nudgeIntensity, group = nudgeIntensity)
) +
  geom_smooth(
    method = "loess",
    formula = y ~ x,
    span = 0.6,
    se = TRUE,
    color = NA,
    fill = "grey85"
  ) +
  stat_smooth(
    method = "loess",
    formula = y ~ x,
    span = 0.3,
    linewidth = 1.1,
    se = FALSE
  ) +
  geom_point(size = 1.6, alpha = 0.6) +
  facet_grid(. ~ panel) +
  scale_color_manual(values = intensity_palette) +
  scale_x_continuous(breaks = c(1, 4, 7, 10, 14), minor_breaks = 1:14) +
  scale_y_continuous(limits = c(0.15, 0.75), breaks = seq(0, 1, by = 0.2)) +
  labs(
    x = "Day",
    y = "Pro-environmental choice",
    color = NULL
  ) +
  theme_bw(base_size = 15) +
  theme(
    panel.grid.minor = element_line(color = "grey92", linewidth = 0.3),
    panel.grid.major = element_line(color = "grey88", linewidth = 0.4),
    strip.background = element_blank(),
    strip.text = element_text(size = 12, face = "bold"),
    axis.title = element_text(face = "bold"),
    aspect.ratio = 1,
    legend.position = c(0.01, 0.98),
    legend.justification = c("left", "top"),
    legend.direction = "horizontal",
    legend.background = element_rect(fill = scales::alpha("white", 0.7), color = NA),
    legend.key.size = grid::unit(0.4, "cm"),
    legend.text = element_text(size = 9)
  )

hurdle_plot_df <- build_donation_hurdle_plot_data(donation_day_long) %>%
  filter(!is.na(value))

frequency_palette <- c(
  "Control" = "#5C6770",
  "Low" = "#D8C07A",
  "Medium" = "#C67E3D",
  "High" = "#A94733"
)

donation_hurdle_plot <- ggplot(
  hurdle_plot_df,
  aes(x = frequency_label, y = value, color = frequency_label, fill = frequency_label)
) +
  geom_boxplot(
    width = 0.54,
    outlier.shape = NA,
    alpha = 0.25,
    linewidth = 0.55,
    na.rm = TRUE
  ) +
  geom_jitter(
    width = 0.12,
    height = 0,
    size = 1.35,
    alpha = 0.35,
    stroke = 0,
    na.rm = TRUE
  ) +
  stat_summary(
    fun = mean,
    geom = "point",
    shape = 18,
    size = 3.1,
    alpha = 0.95,
    show.legend = FALSE,
    na.rm = TRUE
  ) +
  facet_wrap(~ panel, nrow = 1, scales = "free_y") +
  scale_color_manual(values = frequency_palette, guide = "none") +
  scale_fill_manual(values = frequency_palette, guide = "none") +
  labs(x = NULL, y = NULL) +
  theme_bw(base_size = 12) +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    strip.background = element_rect(fill = "#F3F1ED", color = "#D8D3C8"),
    strip.text = element_text(face = "bold"),
    axis.text.x = element_text(face = "bold")
  )

donation_event_window_plot_df <- donation_event_window_adjusted %>%
  mutate(
    nudgeType = factor(label_nudge_type(nudgeType), levels = c("Information", "Structure", "Assistant")),
    event_window = factor(as.character(event_window), levels = c("NudgeDay", "PostDay1", "PostDay2"), labels = c("Nudge day", "+1 day", "+2 days"))
  )

donation_event_window_position <- position_dodge(width = 0.18)

donation_event_window_plot <- ggplot(
  donation_event_window_plot_df,
  aes(x = event_window, y = emmean, color = nudgeType, group = nudgeType)
) +
  geom_line(position = donation_event_window_position, linewidth = 1.0, alpha = 0.95, na.rm = TRUE) +
  geom_errorbar(aes(ymin = lower.CL, ymax = upper.CL), position = donation_event_window_position, width = 0.06, linewidth = 0.60, na.rm = TRUE) +
  geom_point(position = donation_event_window_position, shape = 21, fill = "white", stroke = 1.1, size = 3.3, na.rm = TRUE) +
  scale_color_manual(values = type_palette[c("Information", "Structure", "Assistant")]) +
  labs(
    x = NULL,
    y = "Mean donation cents",
    color = NULL
  ) +
  theme_bw(base_size = 16) +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    aspect.ratio = 1,
    axis.text.x = element_text(face = "bold"),
    legend.position = c(0.98, 0.98),
    legend.justification = c(1, 1),
    legend.direction = "vertical",
    legend.background = element_rect(
      fill = grDevices::adjustcolor("white", alpha.f = 0.88),
      color = "#D8D8D8"
    ),
    legend.key = element_rect(fill = NA, color = NA),
    legend.text = element_text(face = "bold")
  )

attitude_plot_df <- build_outcome_plot_df(
  data = df,
  pre_col = "envAttitude_pre",
  post_col = "envAttitude_post",
  lower_bound = 1,
  upper_bound = 5
)

attitude_feedback_plot <- build_prepost_feedback_plot(
  analysis_df = attitude_plot_df,
  pre_col = "envAttitude_pre",
  post_col = "envAttitude_post",
  y_label = "Environmental Attitude",
  feedback_source_df = df,
  y_display_limits = NULL,
  legend_mode = "inside"
)

motivation_plot_df <- df %>%
  filter(as.character(group) != "control", !is.na(delta_motivation), !is.na(eco_choice_change)) %>%
  mutate(
    nudgeType = factor(nudgeType, levels = c("information", "structure", "assistant"), labels = c("Information", "Structure", "Assistant")),
    eco_change = factor(eco_choice_change, levels = sort(unique(eco_choice_change)))
  )

motivation_eco_change_plot <- ggplot(
  motivation_plot_df,
  aes(x = eco_change, y = delta_motivation)
  ) +
  geom_boxplot(
    aes(fill = nudgeType, color = nudgeType),
    position = position_dodge(width = 0.72),
    width = 0.42,
    alpha = 0.58,
    outlier.shape = NA,
    na.rm = TRUE
  ) +
  ggbeeswarm::geom_quasirandom(
    aes(color = nudgeType),
    dodge.width = 0.72,
    width = 0.18,
    size = 1.2,
    alpha = 0.30,
    show.legend = FALSE,
    na.rm = TRUE
  ) +
  stat_summary(
    aes(color = nudgeType),
    fun = mean,
    geom = "point",
    position = position_dodge(width = 0.72),
    size = 2.0,
    shape = 18,
    show.legend = FALSE,
    na.rm = TRUE
  ) +
  scale_fill_manual(values = type_palette[c("Information", "Structure", "Assistant")]) +
  scale_color_manual(values = type_palette[c("Information", "Structure", "Assistant")]) +
  guides(color = "none") +
  labs(
    x = "Change in pro-environmental choices",
    y = "Change in environmental motivation",
    fill = "Nudge type"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    panel.border = element_blank(),
    axis.line.x = element_line(colour = "black", linewidth = 0.3),
    axis.line.y = element_line(colour = "black", linewidth = 0.3),
    axis.title.x = element_text(size = 14, face = "bold"),
    axis.title.y = element_text(size = 14, face = "bold"),
    legend.position = "top"
  ) +
  coord_flip()

simulation_main_trend_plot <- build_simulation_main_trend_plot(simulation_wide)
simulation_gap_plot <- build_simulation_gap_plot(simulation_group_day)

save_png(main_trend_plot, "01_main_trend.png", width = 13, height = 4.8)
save_png(donation_hurdle_plot, "02_donation_hurdle.png", width = 9.5, height = 4.8)
save_png(donation_event_window_plot, "03_donation_event_window.png", width = 7, height = 7)
save_png(attitude_feedback_plot, "04_attitude_feedback.png", width = 18, height = 7.8)
save_png(motivation_eco_change_plot, "05_motivation_eco_change.png", width = 7.5, height = 7)
save_png(simulation_main_trend_plot, "06_simulation_main_trend.png", width = 13, height = 4.8)
save_png(simulation_gap_plot, "07_simulation_gap.png", width = 4.8, height = 6.9)

print_subsection("Sample Overview and Aggregate Behavioural Outcomes")
print_table(sample_overview, "Sample Overview.", digits = 0)

print_subsection("Behavioural Comparisons between Day 1 and Day 14")
print_table(day1_vs_control, "Comparisons between control and intervention groups at Day 1.", digits = 2)
print_table(day14_vs_control, "Comparisons between control and intervention groups at Day 14.", digits = 2)
print_table(within_group_change, "Within-group changes from Day 1 to Day 14.", digits = 2)

print_subsection("Comparison of Linear and Nonlinear Fits of behavior")
print_table(fit_compare_df, "Parameters and summary metrics from nonlinear exponential models of daily pro-environmental behaviour.", digits = 2)

print_subsection("Simulation Validity")
print_table(simulation_validation_summary, "Group-level simulation validity summary.", digits = 2)
print_table(simulation_condition_table, "Condition-level validation of 14-day group trajectories.", digits = 2)

print_subsection("Frequency Effects on Daily Donation Cents")
print_table(donation_frequency_means, "Mean Daily Donation (Cents) by Nudge Frequency Condition", digits = 2)
print_table(donation_frequency_rr, "Model-Based Rate Ratios of Daily Donations by Nudge Frequency", digits = 2)
print_table(donation_frequency_pairwise, "Pairwise Rate Ratios of Daily Donations by Nudge Frequency", digits = 2)

print_subsection("Event window Analysis in Medium-Intensity Arms on Choice Behavior")
print_table(event_window_choice_contrasts, "Within-type contrasts for low-intensity arms: Nudge day vs Post-day 1/2.", digits = 2)

print_subsection("Event-Window Analysis in Medium-Intensity Arms on Daily Donation Cents")
print_table(donation_structure_means, "Mean Daily Donations by Structural Cue State", digits = 2)
print_table(donation_structure_rr, "Model-Based Rate Ratios of Daily Donations by Structural Cue State", digits = 2)
print_table(donation_event_window_contrasts, "Event-Window Effects on Daily Donations in Medium-Frequency Conditions", digits = 2)

print_subsection("Hurdle Decomposition of Daily Donation Cents")
print_table(donation_hurdle_binary, "Model-Based Result for the Probability of Any Donation by Nudge Frequency", digits = 2)
print_table(donation_hurdle_positive, "Model-Based Result for Donation Amount Conditional on Donating by Nudge Frequency", digits = 2)

print_subsection("Pre-Post Comparisons between Environmental Attitude, Autonomous Motivation and Self-efficacy")
print_table(env_desc_table, "Descriptive statistics for environmental autonomous motivation, environmental attitude, and environmental self-efficacy.", digits = 2)
print_table(env_anova_summary_table, "ANOVA results for pre-post changes in environmental attitude, autonomous motivation and self-efficacy.", digits = 2)
print_table(env_prepost_summary_table, "Pre-post contrasts of environmental attitude, autonomous motivation and self-efficacy.", digits = 2)

print_subsection("Descriptive Statistics and Between-group Differences in Nudge Acceptance, Reactance and Fatigue")
print_table(nudge_feedback_desc_overall, "Descriptive statistics for post-test nudge acceptance, reactance and fatigue.", digits = 2)
print_table(nudge_feedback_anova, "ANOVA results for post-test nudge acceptance, reactance and fatigue by nudge type, intensity, and their interaction.", digits = 2)
print_table(nudge_feedback_pairs, "Pairwise comparisons for post-test nudge acceptance, reactance and fatigue.", digits = 2)

print_subsection("Associations between internalising variables and pro-environmental choice change")
print_table(focus_main_effect_table, "Pooled regressions predicting pro-environmental choice change from internalising variables in the four focal groups.", digits = 2)
print_table(focus_main_effect_by_type_table, "Pooled regressions predicting pro-environmental choice change from internalising variables within each nudge type in the medium- and high-intensity groups.", digits = 2)

print_subsection("Moderating Effect of Nudge Fatigue and Reactance on the Relationship between Internalizing Variables and Pro-Environmental Choice")
print_table(focus_moderation_table, "Moderation analyses predicting change in pro-environmental choice from interactions between internalisation variables and nudge fatigue or reactance (interaction terms shown).", digits = 2)
print_table(focus_moderation_by_type_table, "Moderation analyses within each nudge type in the medium- and high-intensity groups (interaction terms shown).", digits = 2)

print_subsection("Mediating Role of Internalising Variables in the Relationship between Nudge Response Time and Pro-Environmental Choice")
print_table(rt_regression_table, "Associations between nudge response time and changes in internalisation-related variables in the focal groups.", digits = 2)
print_table(focus_mediation_table, "Mediation analyses of nudge response time, internalising variables, and change in pro-environmental choice in the focal groups.", digits = 2)
