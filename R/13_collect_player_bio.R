# -*- coding: utf-8 -*-
# Player biographical data (birthdate -> age, bats/throws, height/weight,
# primary position) via baseballr::mlb_people(), for every pitcher and
# batter identified elsewhere in the pipeline (R/01's probables cache +
# R/12's lineups). This is what lets features/builder.py build platoon
# splits (batter handedness vs. pitcher handedness) and age-based features
# — neither existed before since nothing collected player bio info at all.
#
# Runs last in R/00_run_all.R: it depends on the MLBAM IDs surfaced by
# R/01_collect_gamelogs.R (pitchers) and R/12_collect_lineups.R (batters).

library(baseballr)
library(dplyr)
library(readr)

source("R/utils.R")

RAW_DIR <- "data/raw"
dir.create(RAW_DIR, showWarnings = FALSE, recursive = TRUE)
BATCH_SIZE <- 100

log_msg("=== gathering known player MLBAM IDs ===")

pitcher_ids <- character(0)
probable_files <- list.files(RAW_DIR, pattern = "^mlb_probables_cache_\\d{4}\\.csv$", full.names = TRUE)
for (f in probable_files) {
  df <- tryCatch(read_csv(f, show_col_types = FALSE), error = function(e) NULL)
  if (!is.null(df) && "id" %in% names(df)) pitcher_ids <- c(pitcher_ids, as.character(df$id))
}

batter_ids <- character(0)
lineup_files <- list.files(RAW_DIR, pattern = "^lineups_\\d{4}\\.csv$", full.names = TRUE)
for (f in lineup_files) {
  df <- tryCatch(read_csv(f, show_col_types = FALSE), error = function(e) NULL)
  if (!is.null(df) && "batter_mlbam_id" %in% names(df)) batter_ids <- c(batter_ids, as.character(df$batter_mlbam_id))
}

all_ids <- unique(c(pitcher_ids, batter_ids))
all_ids <- all_ids[!is.na(all_ids) & all_ids != ""]
log_msg("  %d unique player IDs (%d from probables, %d from lineups)", length(all_ids), length(unique(pitcher_ids)), length(unique(batter_ids)))

if (length(all_ids) == 0) {
  log_msg("WARNING: no player IDs found (run 01_collect_gamelogs.R and 12_collect_lineups.R first)")
} else {
  id_batches <- split(all_ids, ceiling(seq_along(all_ids) / BATCH_SIZE))
  all_bio <- list()

  for (i in seq_along(id_batches)) {
    batch <- as.numeric(id_batches[[i]])
    res <- tryCatch(mlb_people(person_ids = batch), error = function(e) {
      log_msg("  batch %d/%d ERROR: %s", i, length(id_batches), conditionMessage(e))
      NULL
    })
    if (!is.null(res) && nrow(res) > 0) all_bio[[length(all_bio) + 1]] <- res
    if (i %% 5 == 0 || i == length(id_batches)) {
      log_msg("  fetched bio batch %d/%d", i, length(id_batches))
    }
  }

  if (length(all_bio) == 0) {
    log_msg("WARNING: no bio data returned for any player")
  } else {
    combined <- bind_rows(all_bio)
    out <- combined %>%
      transmute(
        mlbam_id = id,
        full_name = strip_accents(full_name),
        birth_date = as.character(as.Date(birth_date)),
        bat_side = bat_side_code,
        throw_hand = pitch_hand_code,
        height = height,
        weight_lbs = as.numeric(weight),
        primary_position = primary_position_abbreviation
      ) %>%
      filter(!is.na(mlbam_id)) %>%
      distinct(mlbam_id, .keep_all = TRUE)

    out_path <- file.path(RAW_DIR, "player_bio.csv")
    write_csv(out, out_path)
    log_msg("wrote %s (%d players)", out_path, nrow(out))
  }
}
