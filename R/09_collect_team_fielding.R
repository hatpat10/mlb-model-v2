# -*- coding: utf-8 -*-
# Team-level defensive quality for 2020-2025 (baseballr::fg_team_fielder()).
# Closes a real gap: nothing in the original pipeline measured defense —
# only offense (batting) and pitching. A team that's -20 runs on defense is
# a real signal source independent of both.

library(baseballr)
library(dplyr)
library(readr)

source("R/utils.R")

RAW_DIR <- "data/raw"
dir.create(RAW_DIR, showWarnings = FALSE, recursive = TRUE)
YEARS <- 2020:current_season()

# FanGraphs discontinued UZR (and its ARM/RngR/ErrR components) in favor of
# Statcast-based OAA for recent seasons, so these columns are genuinely
# absent for some years — not a bug. Access defensively rather than assume
# every column always exists.
safe_col <- function(df, col) if (col %in% names(df)) as.numeric(df[[col]]) else NA_real_

for (year in YEARS) {
  out_path <- file.path(RAW_DIR, sprintf("team_fielding_%d.csv", year))
  if (skip_completed_season(year, out_path)) {
    log_msg("=== %d: finished season already collected, skipping ===", year)
    next
  }
  log_msg("=== %d: fetching FanGraphs team fielding ===", year)

  df <- tryCatch(
    fg_team_fielder(startseason = as.character(year), endseason = as.character(year)),
    error = function(e) {
      log_msg("  ERROR fetching FanGraphs team fielding for %d: %s", year, conditionMessage(e))
      NULL
    }
  )

  if (is.null(df) || nrow(df) == 0) {
    log_msg("  no data for %d, skipping", year)
    next
  }

  out <- tibble(
    # fg_team_fielder() returns team_name as a bare nickname ("Brewers")
    # rather than the full franchise name fg_team_batter()/fg_pitcher_leaders()
    # use ("Milwaukee Brewers") — team_name_abb is a real (if BRef-style)
    # abbreviation and normalizes far more reliably.
    team = normalize_team(df$team_name_abb),
    year = as.integer(year),
    drs = safe_col(df, "DRS"),
    uzr = safe_col(df, "UZR"),
    oaa = safe_col(df, "OAA"),
    def_runs = safe_col(df, "Defense"),
    error_runs = safe_col(df, "ErrR"),
    range_runs = safe_col(df, "RngR")
  ) %>%
    filter(!is.na(team)) %>%
    distinct(team, year, .keep_all = TRUE)

  write_csv(out, out_path)
  log_msg("  wrote %s (%d rows)", out_path, nrow(out))
}
