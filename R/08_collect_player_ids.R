# -*- coding: utf-8 -*-
# Chadwick Bureau player ID crosswalk (MLBAM <-> FanGraphs <-> Baseball-Reference
# <-> Retrosheet IDs). This is the general-purpose ID table used to join
# player-level data across sources reliably, instead of matching on name
# strings (which is what R/01_collect_gamelogs.R and features/builder.py
# had to do for starting pitchers before this existed).
#
# The full Chadwick register (~500k rows) covers every professional player,
# manager, and umpire in recorded history; we filter down to players active
# since 2019 (one year before our data window starts) to keep this relevant
# and small.

library(dplyr)
library(readr)
library(baseballr)

source("R/utils.R")

RAW_DIR <- "data/raw"
dir.create(RAW_DIR, showWarnings = FALSE, recursive = TRUE)
MIN_LAST_YEAR <- 2019

log_msg("=== fetching Chadwick Bureau player register ===")

cw <- tryCatch(chadwick_player_lu(), error = function(e) {
  log_msg("ERROR fetching Chadwick register: %s", conditionMessage(e))
  NULL
})

if (is.null(cw) || nrow(cw) == 0) {
  log_msg("WARNING: Chadwick register unavailable, player_id_crosswalk.csv not written")
} else {
  out <- cw %>%
    filter(!is.na(mlb_played_last), mlb_played_last >= MIN_LAST_YEAR) %>%
    transmute(
      key_mlbam = key_mlbam,
      key_fangraphs = key_fangraphs,
      key_bbref = key_bbref,
      key_retro = key_retro,
      name_first = strip_accents(name_first),
      name_last = strip_accents(name_last),
      mlb_played_first = as.integer(mlb_played_first),
      mlb_played_last = as.integer(mlb_played_last)
    ) %>%
    filter(!is.na(key_mlbam)) %>%
    distinct(key_mlbam, .keep_all = TRUE)

  out_path <- file.path(RAW_DIR, "player_id_crosswalk.csv")
  write_csv(out, out_path)
  log_msg("wrote %s (%d players, active %d+)", out_path, nrow(out), MIN_LAST_YEAR)
}
