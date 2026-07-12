# -*- coding: utf-8 -*-
# Start-by-start game logs for starting pitchers, 2020-2025
# (baseballr::fg_pitcher_game_logs()), keyed by the FanGraphs playerid
# already captured in R/03_collect_fg_pitching.R. This is what lets
# features/builder.py compute rolling recent-form features (e.g. ERA over
# a starter's last 3 starts) instead of relying only on a static
# season-long average — a pitcher coming off two rough outings looks very
# different from his season line.
#
# Only ~150-200 starters/year (vs. 1500+ batters), so per-player API calls
# are tractable here in a way they aren't for batter game logs (see
# R/10_collect_batter_stats.R for why that one stays season-level).

library(baseballr)
library(dplyr)
library(readr)

source("R/utils.R")

RAW_DIR <- "data/raw"
dir.create(RAW_DIR, showWarnings = FALSE, recursive = TRUE)
YEARS <- 2020:current_season()

for (year in YEARS) {
  out_path <- file.path(RAW_DIR, sprintf("pitcher_gamelogs_%d.csv", year))
  if (skip_completed_season(year, out_path)) {
    log_msg("=== %d: finished season already collected, skipping ===", year)
    next
  }
  log_msg("=== %d: fetching SP game logs ===", year)

  fg_path <- file.path(RAW_DIR, sprintf("fg_sp_stats_%d.csv", year))
  if (!file.exists(fg_path)) {
    log_msg("  no fg_sp_stats_%d.csv found (run 03_collect_fg_pitching.R first), skipping", year)
    next
  }
  starters <- read_csv(fg_path, show_col_types = FALSE) %>%
    filter(!is.na(fg_playerid)) %>%
    distinct(fg_playerid, .keep_all = TRUE)

  log_msg("  %d starters to pull game logs for", nrow(starters))

  all_logs <- list()
  for (i in seq_len(nrow(starters))) {
    pid <- starters$fg_playerid[i]
    res <- tryCatch(fg_pitcher_game_logs(playerid = pid, year = year), error = function(e) NULL)
    if (!is.null(res) && nrow(res) > 0) all_logs[[length(all_logs) + 1]] <- res
    if (i %% 25 == 0 || i == nrow(starters)) {
      log_msg("  fetched %d/%d starters' game logs for %d", i, nrow(starters), year)
    }
  }

  if (length(all_logs) == 0) {
    log_msg("  no game logs returned for %d, skipping", year)
    next
  }

  combined <- bind_rows(all_logs)
  out <- combined %>%
    transmute(
      pitcher_name = strip_accents(PlayerName),
      fg_playerid = playerid,
      date = as.character(as.Date(Date)),
      year = as.integer(year),
      opp = Opp,
      home_away = HomeAway,
      ip = as.numeric(IP),
      era = as.numeric(ERA),
      fip = as.numeric(FIP),
      xfip = as.numeric(xFIP),
      k_pct = suppressWarnings(as.numeric(`K%`)),
      bb_pct = suppressWarnings(as.numeric(`BB%`)),
      hr9 = suppressWarnings(as.numeric(`HR/9`)),
      whip = as.numeric(WHIP)
    ) %>%
    filter(!is.na(pitcher_name), !is.na(date)) %>%
    distinct(fg_playerid, date, .keep_all = TRUE)

  write_csv(out, out_path)
  log_msg("  wrote %s (%d rows)", out_path, nrow(out))
}
