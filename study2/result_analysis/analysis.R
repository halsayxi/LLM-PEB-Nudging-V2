suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(readr)
  library(stringr)
  library(ggplot2)
  library(ggrepel)
})

options(scipen = 999, width = 300)

script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
package_dir <- if (length(script_arg)) dirname(normalizePath(gsub("~\\+~", " ", sub("^--file=", "", script_arg[1])), winslash = "/", mustWork = TRUE)) else normalizePath(getwd(), winslash = "/", mustWork = TRUE)
data_dir <- file.path(package_dir, "data")
output_dir <- file.path(package_dir, "outputs")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

model_files <- c(
  "deepseek-v3" = "data_DeepSeek-V3.csv",
  "deepseek-v4-flash" = "data_DeepSeek-V4-Flash.csv",
  "gpt-3.5-turbo-0125" = "data_GPT-3.5.csv",
  "gpt-4o-2024-11-20" = "data_GPT-4o.csv",
  "claude-haiku-4-5-20251001" = "data_Claude-4.5-Haiku.csv",
  "gpt-5.4" = "data_GPT-5.4.csv"
)

model_labels <- c(
  "deepseek-v3" = "DeepSeek-V3",
  "deepseek-v4-flash" = "DeepSeek-V4-Flash-Think",
  "gpt-3.5-turbo-0125" = "GPT-3.5-Turbo",
  "gpt-4o-2024-11-20" = "GPT-4o",
  "claude-haiku-4-5-20251001" = "Claude-4.5-Haiku",
  "gpt-5.4" = "GPT-5.4-Think"
)

model_order <- c("GPT-3.5-Turbo", "GPT-4o", "Claude-4.5-Haiku", "DeepSeek-V3", "DeepSeek-V4-Flash-Think", "GPT-5.4-Think")
group_order <- c("control", "information_low", "information_medium", "information_high", "structure_low", "structure_medium", "structure_high", "assistant_low", "assistant_medium", "assistant_high")
group_labels <- c("control" = "Control", "information_low" = "Information Low", "information_medium" = "Information Medium", "information_high" = "Information High", "structure_low" = "Structure Low", "structure_medium" = "Structure Medium", "structure_high" = "Structure High", "assistant_low" = "Assistance Low", "assistant_medium" = "Assistance Medium", "assistant_high" = "Assistance High")
type_colors <- c("Control" = "#6B7280", "Information" = "#2F6B9A", "Structure" = "#D97706", "Assistant" = "#7B4FA3")
day_colors <- c("Day 1" = "#3B82B8", "Day 14" = "#E07A5F")
frequency_shapes <- c("Control" = 4, "Low" = 16, "Medium" = 17, "High" = 15)

read_long <- function(path) {
  read_csv(path, show_col_types = FALSE) %>%
    select(ID, group, starts_with("eco_share_day")) %>%
    pivot_longer(starts_with("eco_share_day"), names_to = "day", values_to = "eco_share", names_pattern = "eco_share_day(\\d+)") %>%
    mutate(day = as.integer(day), eco_share = as.numeric(eco_share))
}

human_long <- read_long(file.path(data_dir, "data_human_deidentified.csv"))
simulation_long <- bind_rows(lapply(names(model_files), function(model) read_long(file.path(data_dir, model_files[[model]])) %>% mutate(model = model)))

stopifnot(setequal(unique(human_long$group), group_order), setequal(unique(simulation_long$group), group_order), setequal(unique(human_long$day), 1:14), setequal(unique(simulation_long$day), 1:14))

human_means <- human_long %>%
  group_by(group, day) %>%
  summarise(human_mean = mean(eco_share, na.rm = TRUE), .groups = "drop")

simulation_means <- simulation_long %>%
  group_by(model, group, day) %>%
  summarise(simulation_mean = mean(eco_share, na.rm = TRUE), .groups = "drop")

cell_data <- simulation_means %>%
  inner_join(human_means, by = c("group", "day")) %>%
  mutate(
    model_label = factor(unname(model_labels[model]), levels = model_order),
    condition_type = case_when(group == "control" ~ "Control", str_starts(group, "information") ~ "Information", str_starts(group, "structure") ~ "Structure", TRUE ~ "Assistant"),
    frequency = factor(case_when(group == "control" ~ "Control", str_ends(group, "_low") ~ "Low", str_ends(group, "_medium") ~ "Medium", TRUE ~ "High"), levels = c("Control", "Low", "Medium", "High"))
  )

aggregate_results <- cell_data %>%
  group_by(model) %>%
  group_modify(~{
    test <- cor.test(.x$human_mean, .x$simulation_mean, method = "pearson")
    tibble(r = unname(test$estimate), p = test$p.value, mae = mean(abs(.x$simulation_mean - .x$human_mean)))
  }) %>%
  ungroup() %>%
  mutate(p_adj = p.adjust(p, method = "BH", n = length(model_order)), model_label = unname(model_labels[model]), model_label = factor(model_label, levels = model_order)) %>%
  arrange(model_label)

effect_sizes <- function(data, source_columns = character()) {
  summaries <- data %>%
    group_by(across(all_of(c(source_columns, "group", "day")))) %>%
    summarise(n = sum(is.finite(eco_share)), mean = mean(eco_share, na.rm = TRUE), sd = sd(eco_share, na.rm = TRUE), .groups = "drop")
  controls <- summaries %>%
    filter(group == "control") %>%
    select(all_of(source_columns), day, n_control = n, mean_control = mean, sd_control = sd)
  summaries %>%
    filter(group != "control") %>%
    left_join(controls, by = c(source_columns, "day")) %>%
    mutate(
      pooled_sd = sqrt(((n - 1) * sd^2 + (n_control - 1) * sd_control^2) / (n + n_control - 2)),
      d = case_when(pooled_sd > 0 ~ (mean - mean_control) / pooled_sd, pooled_sd == 0 & mean == mean_control ~ 0, TRUE ~ NA_real_),
      se = sqrt((n + n_control) / (n * n_control) + d^2 / (2 * (n + n_control))),
      ci_lower = d - qnorm(.975) * se,
      ci_upper = d + qnorm(.975) * se
    )
}

human_effects <- effect_sizes(human_long) %>% select(group, day, human_d = d)
simulation_effects <- effect_sizes(simulation_long, "model")
coverage_data <- simulation_effects %>%
  left_join(human_effects, by = c("group", "day")) %>%
  mutate(covered = is.finite(human_d) & human_d >= ci_lower & human_d <= ci_upper)

coverage_results <- coverage_data %>%
  filter(day %in% c(1, 14)) %>%
  group_by(model, day) %>%
  summarise(coverage = 100 * sum(replace_na(covered, FALSE)) / 9, .groups = "drop") %>%
  mutate(model_label = unname(model_labels[model]), model_label = factor(model_label, levels = model_order))

condition_results <- cell_data %>%
  group_by(model, group) %>%
  group_modify(~{
    test <- cor.test(.x$human_mean, .x$simulation_mean, method = "pearson")
    tibble(r = unname(test$estimate), p = test$p.value)
  }) %>%
  ungroup() %>%
  mutate(p_adj = p.adjust(p, method = "BH", n = length(model_order) * length(group_order))) %>%
  mutate(model_label = unname(model_labels[model]), model_label = factor(model_label, levels = model_order), group = factor(group, levels = group_order)) %>%
  arrange(group, model_label)

calibration_results <- cell_data %>%
  group_by(model) %>%
  group_modify(~{
    fit <- lm(simulation_mean ~ human_mean, data = .x)
    ci <- confint(fit, level = .95)
    tibble(intercept = unname(coef(fit)[1]), intercept_lower = ci[1, 1], intercept_upper = ci[1, 2], slope = unname(coef(fit)[2]), slope_lower = ci[2, 1], slope_upper = ci[2, 2], slope_deviation = abs(unname(coef(fit)[2]) - 1))
  }) %>%
  ungroup() %>%
  mutate(model_label = unname(model_labels[model]), model_label = factor(model_label, levels = model_order)) %>%
  arrange(model_label)

fmt2 <- function(x) sub("^(-?)0\\.", "\\1.", sprintf("%.2f", x))
fmt3 <- function(x) sub("^(-?)0\\.", "\\1.", sprintf("%.3f", x))
fmt_p <- function(x) ifelse(x < .001, "<.001", fmt3(x))
fmt_ci <- function(lower, upper) paste0("[", fmt2(lower), ", ", fmt2(upper), "]")

table_1 <- aggregate_results %>% transmute(Model = as.character(model_label), `Pearson's r` = fmt2(r), `BH-adjusted p` = fmt_p(p_adj), MAE = fmt2(mae))
table_2 <- coverage_results %>% select(model_label, day, coverage) %>% pivot_wider(names_from = day, values_from = coverage, names_prefix = "day") %>% arrange(model_label) %>% transmute(Model = as.character(model_label), `Day 1 coverage (%)` = sprintf("%.1f", day1), `Day 14 coverage (%)` = sprintf("%.1f", day14))
table_3 <- condition_results %>% mutate(value = paste0(fmt2(r), " (", fmt_p(p_adj), ")"), Condition = unname(group_labels[as.character(group)])) %>% select(Condition, model_label, value) %>% pivot_wider(names_from = model_label, values_from = value) %>% arrange(factor(Condition, levels = unname(group_labels[group_order])))
table_4 <- calibration_results %>% transmute(Model = as.character(model_label), `Intercept estimate` = fmt2(intercept), `Intercept 95% CI` = fmt_ci(intercept_lower, intercept_upper), `Slope estimate` = fmt2(slope), `Slope 95% CI` = fmt_ci(slope_lower, slope_upper), `|beta - 1|` = fmt2(slope_deviation))

print_table <- function(title, x) {
  cat("\n", title, "\n", sep = "")
  print(as.data.frame(x, check.names = FALSE), row.names = FALSE, right = FALSE)
}

print_table("Table 1. Aggregate correspondence between human and simulated condition-day behavioural means.", table_1)
print_table("Table 2. Effect-size coverage for the initial and end-of-study intervention effects.", table_2)
print_table("Table 3. Condition-specific temporal correspondence between human and simulated behavioural trajectories.", table_3)
print_table("Table 4. Calibration of simulated behavioural trajectories against human condition-day means.", table_4)

theme_paper <- function(base_size = 12) {
  theme_classic(base_size = base_size, base_family = "sans") +
    theme(axis.title = element_text(face = "bold"), axis.text = element_text(color = "#222222"), legend.title = element_text(face = "bold"), legend.position = "bottom", plot.tag = element_text(face = "bold", size = 14, color = "#111111"), plot.margin = margin(10, 12, 10, 10))
}

panel_a <- ggplot(aggregate_results, aes(x = 1 - mae, y = r)) +
  geom_point(size = 5, alpha = .98, color = "#556575") +
  geom_text_repel(aes(label = model_label), size = 3.85, box.padding = .35, point.padding = .25, min.segment.length = 0, segment.color = "#9CA3AF", seed = 20260831, max.overlaps = Inf) +
  scale_x_continuous("1 − MAE", limits = c(min(1 - aggregate_results$mae) - .025, max(1 - aggregate_results$mae) + .015), breaks = seq(.72, .88, .04), expand = expansion(mult = 0)) +
  scale_y_continuous("Human–simulation correlation", limits = c(min(aggregate_results$r) - .08, max(aggregate_results$r) + .08), breaks = seq(.3, .8, .1), expand = expansion(mult = 0)) +
  labs(tag = "a") +
  theme_paper()

coverage_plot_data <- coverage_results %>% mutate(time = factor(if_else(day == 1, "Day 1", "Day 14"), levels = c("Day 1", "Day 14")), model_label = factor(model_label, levels = rev(model_order)))
coverage_wide <- coverage_plot_data %>% select(model, model_label, time, coverage) %>% pivot_wider(names_from = time, values_from = coverage)
overlap_coverage <- coverage_plot_data %>% group_by(model) %>% filter(n_distinct(coverage) == 1, time == "Day 14") %>% ungroup()

panel_b <- ggplot() +
  geom_segment(data = coverage_wide, aes(x = `Day 1`, xend = `Day 14`, y = model_label, yend = model_label), linewidth = 1.8, color = "#CBD5E1") +
  geom_point(data = coverage_plot_data, aes(x = coverage, y = model_label, color = time), shape = 16, size = 4.1, stroke = 1.15) +
  geom_point(data = overlap_coverage, aes(x = coverage, y = model_label), shape = 21, size = 5.5, fill = NA, color = unname(day_colors["Day 14"]), stroke = 1.15) +
  geom_text(data = coverage_plot_data, aes(x = coverage, y = model_label, label = sprintf("%.1f", coverage)), nudge_y = .27, size = 3.35, color = "#374151") +
  scale_color_manual(values = day_colors, name = NULL) +
  scale_x_continuous("Coverage (%)", limits = c(0, 85), breaks = seq(0, 80, 20), expand = expansion(mult = c(0, .02))) +
  labs(y = NULL, tag = "b") +
  theme_paper() +
  theme(panel.grid.major.x = element_line(color = "#E5E7EB", linewidth = .35))

panel_c <- ggplot(cell_data, aes(x = human_mean, y = simulation_mean, color = condition_type, shape = frequency)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", linewidth = .8, color = "#6B7280") +
  geom_smooth(data = cell_data, aes(x = human_mean, y = simulation_mean, group = model_label), inherit.aes = FALSE, method = "lm", formula = y ~ x, se = TRUE, color = "#222222", fill = "#9CA3AF", linewidth = .8, alpha = .16) +
  geom_point(size = 2.7, alpha = .78) +
  scale_color_manual(values = type_colors, name = "Condition") +
  scale_shape_manual(values = frequency_shapes, name = "Frequency") +
  scale_x_continuous("Human mean", breaks = seq(0, 1, .2), expand = expansion(mult = 0)) +
  scale_y_continuous("Simulation mean", breaks = seq(0, 1, .2), expand = expansion(mult = 0)) +
  facet_wrap(~model_label, ncol = 3, axes = "all", axis.labels = "all") +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1), clip = "on") +
  labs(tag = "c") +
  theme_paper(11) +
  theme(strip.background = element_blank(), strip.text = element_text(face = "bold", size = 11.5), panel.spacing = grid::unit(1, "lines"))

ggsave(file.path(output_dir, "panel_A_model_performance.png"), panel_a, width = 7.2, height = 5.5, dpi = 400, bg = "white")
ggsave(file.path(output_dir, "panel_B_day1_day14_coverage.png"), panel_b, width = 7.6, height = 5.5, dpi = 400, bg = "white")
ggsave(file.path(output_dir, "panel_C_human_vs_simulation.png"), panel_c, width = 11.5, height = 7.6, dpi = 400, bg = "white")

cat("\nFigures saved to: ", output_dir, "\n", sep = "")
