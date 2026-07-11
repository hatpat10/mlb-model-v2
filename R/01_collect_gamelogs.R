# -*- coding: utf-8 -*-
# Collects game-by-game results (one row per team per game) for 2021-2026
# from the MLB Stats API via baseballr::mlb_schedule() for scores, and
# baseballr::mlb_probables() per game for probable starters + home-plate
# umpire (the same payload also feeds R/07_collect_umpires.R, so results
# are cached to data/raw/mlb_probables_cache_<year>.csv and reused there).

library(baseballr)
library(dplyr)
library(readr)
library(lubridate)
library(stringr)

source("R/utils.R")

RAW_DIR <- "data/raw"
dir.create(RAW_DIR, showWarnings = FALSE, recursive = TRUE)
YEARS <- 2021:2026

fetch_probables_cached <- function(game_pks, year) {
  cache_path <- file.path(RAW_DIR, sprintf("mlb_probables_cache_%d.csv", year))

  # readr infers types on re-read (game_pk as numeric since it's all-digit,
  # game_date as Date since it looks like one) that don't match what a fresh
  # mlb_probables() response gives us — bind_rows() errors the moment a
  # cache load is combined with freshly-fetched rows unless both sides are
  # normalized to the same types first.
  normalize_types <- function(df) {
    if ("game_pk" %in% names(df)) df$game_pk <- as.character(df$game_pk)
    if ("game_date" %in% names(df)) df$game_date <- as.character(df$game_date)
    df
  }

  cached <- NULL
  already_have <- character(0)
  if (file.exists(cache_path)) {
    cached <- normalize_types(read_csv(cache_path, show_col_types = FALSE))
    already_have <- unique(cached$game_pk)
  }

  todo <- setdiff(as.character(game_pks), already_have)
  log_msg("  probables: %d cached, %d to fetch", length(already_have), length(todo))

  new_rows <- list()
  flush_every <- 100
  n_done <- 0

  for (pk in todo) {
    res <- tryCatch(mlb_probables(game_pk = as.numeric(pk)), error = function(e) NULL)
    if (!is.null(res) && nrow(res) > 0) {
      res <- normalize_types(res)
      new_rows[[length(new_rows) + 1]] <- res
    }
    n_done <- n_done + 1

    if (n_done %% flush_every == 0 || n_done == length(todo)) {
      if (length(new_rows) > 0) {
        batch <- normalize_types(bind_rows(new_rows))
        cached <- if (!is.null(cached)) bind_rows(cached, batch) else batch
        write_csv(cached, cache_path)
        new_rows <- list()
      }
      log_msg("  probables: fetched %d/%d for %d", n_done, length(todo), year)
    }
  }

  if (is.null(cached)) return(tibble())
  cached
}

all_years_rows <- list()

for (year in YEARS) {
  log_msg("=== %d: fetching schedule ===", year)

  sched <- tryCatch(mlb_schedule(season = year, level_ids = "1"), error = function(e) {
    log_msg("  ERROR fetching schedule for %d: %s", year, conditionMessage(e))
    NULL
  })

  if (is.null(sched) || nrow(sched) == 0) {
    log_msg("  no schedule data for %d, skipping", year)
    next
  }

  reg <- sched %>%
    filter(game_type == "R", status_detailed_state %in% c("Final", "Completed Early")) %>%
    distinct(game_pk, .keep_all = TRUE)

  log_msg("  %d regular-season completed games", nrow(reg))
  if (nrow(reg) == 0) next

  probables <- fetch_probables_cached(reg$game_pk, year)

  # Build one row per GAME (not per team-row) with home/away starter + umpire,
  # then join that onto both team-perspective rows below.
  if (nrow(probables) > 0) {
    starter_by_team <- probables %>%
      transmute(
        game_pk = as.character(game_pk),
        team_id = team_id,
        starter_name = strip_accents(fullName),
        umpire_name = strip_accents(home_plate_full_name)
      ) %>%
      distinct(game_pk, team_id, .keep_all = TRUE)

    game_starters <- reg %>%
      transmute(game_pk = as.character(game_pk),
                home_team_id = teams_home_team_id,
                away_team_id = teams_away_team_id) %>%
      left_join(starter_by_team %>% select(game_pk, team_id, home_starter = starter_name, umpire_name),
                by = c("game_pk", "home_team_id" = "team_id")) %>%
      left_join(starter_by_team %>% select(game_pk, team_id, away_starter = starter_name),
                by = c("game_pk", "away_team_id" = "team_id")) %>%
      select(game_pk, home_starter, away_starter, umpire_name)
  } else {
    game_starters <- reg %>%
      transmute(game_pk = as.character(game_pk),
                home_starter = NA_character_, away_starter = NA_character_,
                umpire_name = NA_character_)
  }

  reg_j <- reg %>% mutate(game_pk = as.character(game_pk)) %>% left_join(game_starters, by = "game_pk")

  home_rows <- reg_j %>%
    transmute(
      date = as.character(as.Date(official_date)),
      game_pk = game_pk,
      team_full = teams_home_team_name,
      opponent_full = teams_away_team_name,
      is_home = 1L,
      runs_scored = teams_home_score,
      runs_allowed = teams_away_score,
      home_starter, away_starter
    )

  away_rows <- reg_j %>%
    transmute(
      date = as.character(as.Date(official_date)),
      game_pk = game_pk,
      team_full = teams_away_team_name,
      opponent_full = teams_home_team_name,
      is_home = 0L,
      runs_scored = teams_away_score,
      runs_allowed = teams_home_score,
      home_starter, away_starter
    )

  year_rows <- bind_rows(home_rows, away_rows) %>%
    mutate(
      team = normalize_team(team_full),
      opponent = normalize_team(opponent_full),
      win = as.integer(runs_scored > runs_allowed),
      year = year(as.Date(date)),
      month = month(as.Date(date)),
      day_of_week = as.character(wday(as.Date(date), label = TRUE, abbr = TRUE))
    ) %>%
    filter(!is.na(runs_scored), !is.na(runs_allowed), runs_scored != runs_allowed) %>%
    select(date, game_pk, team, opponent, is_home, runs_scored, runs_allowed, win,
           home_starter, away_starter, year, month, day_of_week)

  n_before <- nrow(year_rows)
  year_rows <- year_rows %>% distinct(game_pk, team, .keep_all = TRUE)
  n_dupes <- n_before - nrow(year_rows)
  log_msg("  %d rows, %d duplicate (game_pk, team) rows dropped", nrow(year_rows), n_dupes)

  out_path <- file.path(RAW_DIR, sprintf("game_logs_%d.csv", year))
  write_csv(year_rows, out_path)
  log_msg("  wrote %s (%d rows)", out_path, nrow(year_rows))

  all_years_rows[[as.character(year)]] <- year_rows
}

if (length(all_years_rows) > 0) {
  combined <- bind_rows(all_years_rows)
  write_csv(combined, file.path(RAW_DIR, "game_logs_all.csv"))
  log_msg("wrote game_logs_all.csv (%d rows total)", nrow(combined))
} else {
  log_msg("WARNING: no game log data collected for any year")
}
