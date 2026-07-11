# -*- coding: utf-8 -*-
# Team-level batted-ball quality for 2020-2025.
#
# A direct Baseball Savant Statcast team-batting scrape proved unreliable
# (see 06_collect_park_factors.R for the same class of upstream breakage),
# but FanGraphs' team batting leaderboard already carries Statcast-derived
# quality metrics (barrel%, hard-hit%, xBA/xSLG/xwOBA, exit velo) aggregated
# to the team level, so we pull that directly here (fetched independently of
# 02_collect_fg_batting.R, which keeps a different column subset). Sprint
# speed comes from baseballr::statcast_leaderboards(leaderboard =
# "sprint_speed"), a genuine team-aggregable Statcast leaderboard.

library(baseballr)
library(dplyr)
library(readr)

source("R/utils.R")

RAW_DIR <- "data/raw"
dir.create(RAW_DIR, showWarnings = FALSE, recursive = TRUE)
YEARS <- 2020:2025

for (year in YEARS) {
  log_msg("=== %d: building Statcast team batting quality ===", year)

  fgb <- tryCatch(
    fg_team_batter(startseason = as.character(year), endseason = as.character(year)),
    error = function(e) {
      log_msg("  ERROR fetching FanGraphs team batting for %d: %s", year, conditionMessage(e))
      NULL
    }
  )

  if (is.null(fgb) || nrow(fgb) == 0) {
    log_msg("  no FanGraphs team batting available for %d, skipping", year)
    next
  }

  quality <- fgb %>%
    transmute(
      team = normalize_team(team_name),
      year = as.integer(year),
      barrel_pct = as.numeric(Barrel_pct),
      hard_hit_pct = as.numeric(HardHit_pct),
      xba = as.numeric(xAVG),
      xslg = as.numeric(xSLG),
      xwoba = as.numeric(xwOBA),
      exit_velo_avg = as.numeric(EV)
    ) %>%
    filter(!is.na(team))

  sprint <- tryCatch(
    statcast_leaderboards(leaderboard = "sprint_speed", year = year),
    error = function(e) {
      log_msg("  ERROR fetching sprint_speed leaderboard for %d: %s", year, conditionMessage(e))
      NULL
    }
  )

  if (!is.null(sprint) && nrow(sprint) > 0 && "team" %in% names(sprint)) {
    sprint_team <- sprint %>%
      mutate(team = normalize_team(team)) %>%
      filter(!is.na(team)) %>%
      group_by(team) %>%
      summarise(sprint_speed_avg = weighted.mean(sprint_speed, w = pmax(competitive_runs, 1), na.rm = TRUE), .groups = "drop")
  } else {
    log_msg("  WARNING: no sprint_speed data for %d", year)
    sprint_team <- tibble(team = character(0), sprint_speed_avg = numeric(0))
  }

  out <- quality %>%
    left_join(sprint_team, by = "team") %>%
    distinct(team, year, .keep_all = TRUE)

  out_path <- file.path(RAW_DIR, sprintf("statcast_team_batting_%d.csv", year))
  write_csv(out, out_path)
  log_msg("  wrote %s (%d rows)", out_path, nrow(out))
}
