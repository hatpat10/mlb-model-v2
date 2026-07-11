# -*- coding: utf-8 -*-
# Collects team-level batting stats from FanGraphs (baseballr::fg_team_batter)
# for 2020-2025 (2020 is needed so 2021 games can join a prior-year batting
# feature).

library(baseballr)
library(dplyr)
library(readr)
library(janitor)

source("R/utils.R")

RAW_DIR <- "data/raw"
dir.create(RAW_DIR, showWarnings = FALSE, recursive = TRUE)
YEARS <- 2020:2025

for (year in YEARS) {
  log_msg("=== %d: fetching FanGraphs team batting ===", year)

  df <- tryCatch(
    fg_team_batter(startseason = as.character(year), endseason = as.character(year)),
    error = function(e) {
      log_msg("  ERROR fetching FanGraphs team batting for %d: %s", year, conditionMessage(e))
      NULL
    }
  )

  if (is.null(df) || nrow(df) == 0) {
    log_msg("  no data for %d, skipping", year)
    next
  }

  out <- df %>%
    transmute(
      team = normalize_team(team_name),
      year = as.integer(year),
      wrc_plus = as.numeric(wRC_plus),
      woba = as.numeric(wOBA),
      obp = as.numeric(OBP),
      slg = as.numeric(SLG),
      iso = as.numeric(ISO),
      bb_pct = as.numeric(BB_pct),
      k_pct = as.numeric(K_pct),
      babip = as.numeric(BABIP),
      ops = as.numeric(OPS),
      pa = as.numeric(PA)
    ) %>%
    filter(!is.na(team)) %>%
    distinct(team, year, .keep_all = TRUE)

  n_unmapped <- nrow(df) - nrow(out)
  if (n_unmapped > 0) {
    log_msg("  WARNING: %d rows dropped due to unmapped team name", n_unmapped)
  }

  out_path <- file.path(RAW_DIR, sprintf("fg_team_batting_%d.csv", year))
  write_csv(out, out_path)
  log_msg("  wrote %s (%d rows)", out_path, nrow(out))
}
