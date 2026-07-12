# -*- coding: utf-8 -*-
# Individual (not team-aggregate) batter season stats for 2020-2025
# (baseballr::fg_batter_leaders()). Combined with actual starting lineups
# (R/12_collect_lineups.R), this lets features/builder.py estimate the
# real quality of TODAY's lineup instead of a team-season average that
# doesn't know who's actually in the lineup.
#
# Full per-game batter logs (fg_batter_game_logs()) were considered but
# rejected: FanGraphs only exposes that per-player-per-season, and pulling
# it for the 1500+ batters who appeared across 2020-2025 would mean
# 1500+ individual API calls (vs. ~150-200/year for starting pitchers,
# which R/11_collect_pitcher_gamelogs.R does do). Season-level individual
# stats plus real lineup composition captures most of the modeling value
# at a fraction of the request volume.

library(baseballr)
library(dplyr)
library(readr)

source("R/utils.R")

RAW_DIR <- "data/raw"
dir.create(RAW_DIR, showWarnings = FALSE, recursive = TRUE)
YEARS <- 2020:current_season()
MIN_PA <- 50

for (year in YEARS) {
  out_path <- file.path(RAW_DIR, sprintf("batter_stats_%d.csv", year))
  if (skip_completed_season(year, out_path)) {
    log_msg("=== %d: finished season already collected, skipping ===", year)
    next
  }
  log_msg("=== %d: fetching FanGraphs individual batter stats ===", year)

  df <- tryCatch(
    fg_batter_leaders(startseason = as.character(year), endseason = as.character(year), qual = "0"),
    error = function(e) {
      log_msg("  ERROR fetching FanGraphs batter leaders for %d: %s", year, conditionMessage(e))
      NULL
    }
  )

  if (is.null(df) || nrow(df) == 0) {
    log_msg("  no data for %d, skipping", year)
    next
  }

  out <- df %>%
    filter(as.numeric(PA) >= MIN_PA) %>%
    transmute(
      batter_name = strip_accents(PlayerName),
      batter_mlbam_id = xMLBAMID,
      team = normalize_team(team_name),
      year = as.integer(year),
      pa = as.numeric(PA),
      wrc_plus = as.numeric(wRC_plus),
      woba = as.numeric(wOBA),
      obp = as.numeric(OBP),
      slg = as.numeric(SLG),
      iso = as.numeric(ISO),
      bb_pct = as.numeric(BB_pct),
      k_pct = as.numeric(K_pct),
      barrel_pct = as.numeric(Barrel_pct),
      hard_hit_pct = as.numeric(HardHit_pct),
      xwoba = as.numeric(xwOBA)
    ) %>%
    filter(!is.na(batter_name)) %>%
    distinct(batter_name, year, .keep_all = TRUE)

  write_csv(out, out_path)
  log_msg("  wrote %s (%d rows, min %d PA)", out_path, nrow(out), MIN_PA)
}
