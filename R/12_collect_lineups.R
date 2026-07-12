# -*- coding: utf-8 -*-
# Actual starting batting orders per game, 2021-2026
# (baseballr::mlb_batting_orders()). We already know probable STARTERS
# (R/01_collect_gamelogs.R); this adds who's actually hitting, letting
# features/builder.py estimate real lineup strength (via
# R/10_collect_batter_stats.R) instead of assuming a team's full-season
# average lineup is always on the field.
#
# One API call per game, same cost profile as the mlb_probables() pull in
# R/01_collect_gamelogs.R — cached incrementally the same way so repeat
# runs only fetch newly-played games.

library(baseballr)
library(dplyr)
library(readr)

source("R/utils.R")

RAW_DIR <- "data/raw"
dir.create(RAW_DIR, showWarnings = FALSE, recursive = TRUE)
YEARS <- 2021:current_season()

fetch_lineups_cached <- function(game_pks, year) {
  cache_path <- file.path(RAW_DIR, sprintf("lineups_cache_%d.csv", year))

  # readr infers types on re-read (numeric-looking columns become numeric,
  # since it can't know the original API response's types) that don't
  # necessarily match a fresh mlb_batting_orders() response — bind_rows()
  # errors the moment a cache load is combined with freshly-fetched rows
  # unless every column is normalized first. The final transmute() below
  # re-casts each column to its real type anyway, so it's safe to just
  # force everything to character here rather than chase each mismatching
  # column one at a time.
  normalize_types <- function(df) {
    df[] <- lapply(df, as.character)
    df
  }

  cached <- NULL
  already_have <- character(0)
  if (file.exists(cache_path)) {
    cached <- normalize_types(read_csv(cache_path, show_col_types = FALSE))
    already_have <- unique(cached$game_pk)
  }

  todo <- setdiff(as.character(game_pks), already_have)
  log_msg("  lineups: %d cached, %d to fetch", length(already_have), length(todo))

  new_rows <- list()
  flush_every <- 100
  n_done <- 0

  for (pk in todo) {
    res <- tryCatch(mlb_batting_orders(game_pk = as.numeric(pk)), error = function(e) NULL)
    if (!is.null(res) && nrow(res) > 0) {
      res$game_pk <- pk
      new_rows[[length(new_rows) + 1]] <- normalize_types(res)
    }
    n_done <- n_done + 1

    if (n_done %% flush_every == 0 || n_done == length(todo)) {
      if (length(new_rows) > 0) {
        batch <- normalize_types(bind_rows(new_rows))
        cached <- if (!is.null(cached)) bind_rows(cached, batch) else batch
        write_csv(cached, cache_path)
        new_rows <- list()
      }
      log_msg("  lineups: fetched %d/%d for %d", n_done, length(todo), year)
    }
  }

  if (is.null(cached)) return(tibble())
  cached
}

for (year in YEARS) {
  out_path <- file.path(RAW_DIR, sprintf("lineups_%d.csv", year))
  if (skip_completed_season(year, out_path)) {
    log_msg("=== %d: finished season already collected, skipping ===", year)
    next
  }
  log_msg("=== %d: fetching starting lineups ===", year)

  gl_path <- file.path(RAW_DIR, sprintf("game_logs_%d.csv", year))
  if (!file.exists(gl_path)) {
    log_msg("  no game_logs_%d.csv found (run 01_collect_gamelogs.R first), skipping", year)
    next
  }
  game_pks <- unique(read_csv(gl_path, show_col_types = FALSE)$game_pk)
  log_msg("  %d games", length(game_pks))
  if (length(game_pks) == 0) next

  lineups <- fetch_lineups_cached(game_pks, year)

  if (nrow(lineups) == 0) {
    log_msg("  no lineup data returned for %d", year)
    next
  }

  out <- lineups %>%
    transmute(
      game_pk = as.character(game_pk),
      batter_mlbam_id = as.integer(id),
      batter_name = strip_accents(fullName),
      position = abbreviation,
      batting_order = as.integer(batting_order),
      is_home = as.integer(team == "home"),
      year = as.integer(year)
    ) %>%
    filter(!is.na(batter_mlbam_id)) %>%
    distinct(game_pk, batter_mlbam_id, .keep_all = TRUE)

  write_csv(out, out_path)
  log_msg("  wrote %s (%d rows, %d games)", out_path, nrow(out), length(unique(out$game_pk)))
}
