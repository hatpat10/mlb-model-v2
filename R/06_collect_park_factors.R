# -*- coding: utf-8 -*-
# Park factors for 2021-2026.
#
# baseballr::fg_park() is broken as of this baseballr release — FanGraphs
# changed their park-factors page and the scraper throws
# "object 'park_table' not found" on every call/year. Rather than stub the
# column with fabricated numbers, we compute an empirical park factor
# directly from our own collected game logs: for each team-season, compare
# the team's runs environment (runs scored + allowed per game) at home
# against the same team's runs environment on the road. This is the
# standard basic park-factor formula and, unlike a broken scrape, is fully
# reproducible from data we already trust.
#
# park_factor_h (a hits-based factor) is dropped: we only collect per-game
# runs in R/01_collect_gamelogs.R, not box-score hit totals, and computing
# a "hits factor" from runs would just be a relabeled duplicate of
# park_factor. Better to omit it and note why than to stub a fake column.

library(baseballr)
library(dplyr)
library(readr)

source("R/utils.R")

RAW_DIR <- "data/raw"
dir.create(RAW_DIR, showWarnings = FALSE, recursive = TRUE)
YEARS <- 2021:current_season()

all_years <- list()

for (year in YEARS) {
  log_msg("=== %d: computing empirical park factors ===", year)

  gl_path <- file.path(RAW_DIR, sprintf("game_logs_%d.csv", year))
  if (!file.exists(gl_path)) {
    log_msg("  no game_logs_%d.csv found (run 01_collect_gamelogs.R first), skipping", year)
    next
  }
  gl <- read_csv(gl_path, show_col_types = FALSE)
  if (nrow(gl) == 0) {
    log_msg("  game_logs_%d.csv is empty, skipping", year)
    next
  }

  venue_map <- NULL
  sched <- tryCatch(mlb_schedule(season = year, level_ids = "1"), error = function(e) NULL)
  if (!is.null(sched) && nrow(sched) > 0) {
    venue_map <- sched %>%
      filter(game_type == "R") %>%
      mutate(team = normalize_team(teams_home_team_name)) %>%
      filter(!is.na(team)) %>%
      count(team, venue_name, sort = TRUE) %>%
      group_by(team) %>%
      slice_max(n, n = 1, with_ties = FALSE) %>%
      ungroup() %>%
      select(team, venue_name)
  }

  team_env <- gl %>%
    group_by(team, is_home) %>%
    summarise(total_runs = sum(runs_scored + runs_allowed, na.rm = TRUE), g = n(), .groups = "drop") %>%
    mutate(runs_per_g = total_runs / g)

  home_env <- team_env %>% filter(is_home == 1) %>% select(team, home_runs_per_g = runs_per_g)
  road_env <- team_env %>% filter(is_home == 0) %>% select(team, road_runs_per_g = runs_per_g)

  pf <- home_env %>%
    inner_join(road_env, by = "team") %>%
    mutate(
      park_factor = round(100 * home_runs_per_g / road_runs_per_g, 1),
      year = as.integer(year)
    ) %>%
    select(team, year, park_factor)

  if (!is.null(venue_map)) {
    pf <- pf %>% left_join(venue_map, by = "team")
  } else {
    pf$venue_name <- NA_character_
  }

  n_missing_venue <- sum(is.na(pf$venue_name))
  if (n_missing_venue > 0) {
    log_msg("  WARNING: %d/%d teams missing venue_name for %d", n_missing_venue, nrow(pf), year)
  }

  all_years[[as.character(year)]] <- pf
  log_msg("  computed park factors for %d teams in %d", nrow(pf), year)
}

if (length(all_years) > 0) {
  combined <- bind_rows(all_years) %>% distinct(team, year, .keep_all = TRUE)
  out_path <- file.path(RAW_DIR, "park_factors.csv")
  write_csv(combined, out_path)
  log_msg("wrote %s (%d rows)", out_path, nrow(combined))
} else {
  log_msg("WARNING: no park factor data computed for any year")
}
