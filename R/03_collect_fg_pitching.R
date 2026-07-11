# -*- coding: utf-8 -*-
# Collects starting-pitcher-level stats from FanGraphs (baseballr::fg_pitcher_leaders)
# for 2020-2025 (2020 needed for 2021 prior-year joins). Starters only (min 10 GS).

library(baseballr)
library(dplyr)
library(readr)

source("R/utils.R")

RAW_DIR <- "data/raw"
dir.create(RAW_DIR, showWarnings = FALSE, recursive = TRUE)
YEARS <- 2020:2025
MIN_GS <- 10

for (year in YEARS) {
  log_msg("=== %d: fetching FanGraphs SP stats ===", year)

  df <- tryCatch(
    fg_pitcher_leaders(startseason = as.character(year), endseason = as.character(year), qual = "0"),
    error = function(e) {
      log_msg("  ERROR fetching FanGraphs pitching for %d: %s", year, conditionMessage(e))
      NULL
    }
  )

  if (is.null(df) || nrow(df) == 0) {
    log_msg("  no data for %d, skipping", year)
    next
  }

  out <- df %>%
    filter(as.numeric(GS) >= MIN_GS) %>%
    transmute(
      pitcher_name = strip_accents(PlayerName),
      fg_playerid = playerid,
      mlbam_id = xMLBAMID,
      team = normalize_team(team_name),
      year = as.integer(year),
      era = as.numeric(ERA),
      fip = as.numeric(FIP),
      xfip = as.numeric(xFIP),
      siera = as.numeric(SIERA),
      k_pct = as.numeric(K_pct),
      bb_pct = as.numeric(BB_pct),
      whip = as.numeric(WHIP),
      k9 = as.numeric(K_9),
      bb9 = as.numeric(BB_9),
      hr9 = as.numeric(HR_9),
      swstr_pct = as.numeric(SwStr_pct),
      ip = as.numeric(IP),
      gs = as.numeric(GS),
      w = as.numeric(W),
      l = as.numeric(L)
    ) %>%
    filter(!is.na(pitcher_name)) %>%
    distinct(pitcher_name, year, .keep_all = TRUE)

  out_path <- file.path(RAW_DIR, sprintf("fg_sp_stats_%d.csv", year))
  write_csv(out, out_path)
  log_msg("  wrote %s (%d rows, min %d GS)", out_path, nrow(out), MIN_GS)
}
