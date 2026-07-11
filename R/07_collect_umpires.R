# -*- coding: utf-8 -*-
# Home-plate umpire assignments and career run-scoring tendencies.
#
# Retrosheet event files (baseballr::get_retrosheet_data / retrosheet_data)
# require the external Chadwick CLI tool, which isn't installed on this
# machine and can't be assumed present, so we don't use it. Instead we reuse
# the mlb_probables() payload already cached by R/01_collect_gamelogs.R:
# every call to mlb_probables(game_pk) returns home_plate_full_name /
# home_plate_id alongside the probable starters, so umpire assignments come
# for free from data already collected — no extra API calls needed.

library(dplyr)
library(readr)

source("R/utils.R")

RAW_DIR <- "data/raw"
dir.create(RAW_DIR, showWarnings = FALSE, recursive = TRUE)
YEARS <- 2021:2026
MIN_GAMES_FOR_FACTOR <- 10

assignments_all <- list()
runs_all <- list()

for (year in YEARS) {
  cache_path <- file.path(RAW_DIR, sprintf("mlb_probables_cache_%d.csv", year))
  gl_path <- file.path(RAW_DIR, sprintf("game_logs_%d.csv", year))

  if (!file.exists(cache_path)) {
    log_msg("%d: no mlb_probables_cache found (run 01_collect_gamelogs.R first), skipping", year)
    next
  }

  probables <- read_csv(cache_path, show_col_types = FALSE)
  year_assign <- probables %>%
    filter(!is.na(home_plate_full_name)) %>%
    transmute(
      date = as.character(as.Date(game_date)),
      game_pk = as.character(game_pk),
      umpire_name = strip_accents(home_plate_full_name)
    ) %>%
    distinct(game_pk, .keep_all = TRUE)

  log_msg("%d: %d games with known umpire", year, nrow(year_assign))
  assignments_all[[as.character(year)]] <- year_assign

  if (file.exists(gl_path)) {
    gl <- read_csv(gl_path, show_col_types = FALSE)
    game_runs <- gl %>%
      filter(is_home == 1) %>%
      transmute(game_pk = as.character(game_pk), total_runs = runs_scored + runs_allowed)
    runs_all[[as.character(year)]] <- year_assign %>%
      inner_join(game_runs, by = "game_pk")
  } else {
    log_msg("%d: no game_logs_%d.csv, cannot compute run factors for this year", year, year)
  }
}

if (length(assignments_all) == 0) {
  log_msg("WARNING: no umpire assignment data available for any year")
} else {
  assignments <- bind_rows(assignments_all)
  write_csv(assignments, file.path(RAW_DIR, "umpire_assignments.csv"))
  log_msg("wrote umpire_assignments.csv (%d rows)", nrow(assignments))

  if (length(runs_all) > 0) {
    runs <- bind_rows(runs_all)
    league_avg_runs <- mean(runs$total_runs, na.rm = TRUE)

    factors <- runs %>%
      group_by(umpire_name) %>%
      summarise(games = n(), avg_total_runs = mean(total_runs, na.rm = TRUE), .groups = "drop") %>%
      filter(games >= MIN_GAMES_FOR_FACTOR) %>%
      mutate(umpire_run_factor = round(avg_total_runs - league_avg_runs, 3)) %>%
      select(umpire_name, games, avg_total_runs, umpire_run_factor)

    write_csv(factors, file.path(RAW_DIR, "umpire_factors.csv"))
    log_msg("wrote umpire_factors.csv (%d umpires, min %d games, league avg runs/game = %.2f)",
            nrow(factors), MIN_GAMES_FOR_FACTOR, league_avg_runs)
  } else {
    log_msg("WARNING: no game log data joined, umpire_factors.csv not written")
  }
}
