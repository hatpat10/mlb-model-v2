# -*- coding: utf-8 -*-
# Orchestrates all R data-collection scripts in order. A failure in any one
# script is logged and does not stop the others from running. Prints a
# final summary of which scripts succeeded/failed and the row counts of
# every output CSV.
#
# Run from the project root: Rscript R/00_run_all.R

library(readr)

source("R/utils.R")

SCRIPTS <- c(
  "R/01_collect_gamelogs.R",
  "R/02_collect_fg_batting.R",
  "R/03_collect_fg_pitching.R",
  "R/04_collect_statcast_team.R",
  "R/05_collect_statcast_sp.R",
  "R/06_collect_park_factors.R",
  "R/07_collect_umpires.R",
  "R/08_collect_player_ids.R",
  "R/09_collect_team_fielding.R",
  "R/10_collect_batter_stats.R",
  "R/11_collect_pitcher_gamelogs.R",
  "R/12_collect_lineups.R",
  "R/13_collect_player_bio.R"
)

CUR <- current_season()
OUTPUT_GLOBS <- list(
  "R/01_collect_gamelogs.R" = c(sprintf("game_logs_%d.csv", 2021:CUR), "game_logs_all.csv"),
  "R/02_collect_fg_batting.R" = sprintf("fg_team_batting_%d.csv", 2020:CUR),
  "R/03_collect_fg_pitching.R" = sprintf("fg_sp_stats_%d.csv", 2020:CUR),
  "R/04_collect_statcast_team.R" = sprintf("statcast_team_batting_%d.csv", 2020:CUR),
  "R/05_collect_statcast_sp.R" = sprintf("statcast_sp_%d.csv", 2020:CUR),
  "R/06_collect_park_factors.R" = c("park_factors.csv"),
  "R/07_collect_umpires.R" = c("umpire_assignments.csv", "umpire_factors.csv"),
  "R/08_collect_player_ids.R" = c("player_id_crosswalk.csv"),
  "R/09_collect_team_fielding.R" = sprintf("team_fielding_%d.csv", 2020:CUR),
  "R/10_collect_batter_stats.R" = sprintf("batter_stats_%d.csv", 2020:CUR),
  "R/11_collect_pitcher_gamelogs.R" = sprintf("pitcher_gamelogs_%d.csv", 2020:CUR),
  "R/12_collect_lineups.R" = sprintf("lineups_%d.csv", 2021:CUR),
  "R/13_collect_player_bio.R" = c("player_bio.csv")
)

RAW_DIR <- "data/raw"
results <- list()

log_msg("========== R data collection: starting %d scripts ==========", length(SCRIPTS))

for (script in SCRIPTS) {
  log_msg("---------- running %s ----------", script)
  t0 <- Sys.time()
  status <- tryCatch({
    source(script, local = new.env())
    "SUCCESS"
  }, error = function(e) {
    log_msg("SCRIPT FAILED: %s -- %s", script, conditionMessage(e))
    "FAILED"
  })
  elapsed <- round(as.numeric(Sys.time() - t0, units = "secs"), 1)
  results[[script]] <- list(status = status, elapsed = elapsed)
  log_msg("---------- %s: %s (%.1fs) ----------", script, status, elapsed)
}

log_msg("========== SUMMARY ==========")
n_success <- sum(vapply(results, function(r) r$status == "SUCCESS", logical(1)))
log_msg("%d/%d scripts succeeded", n_success, length(SCRIPTS))

CORE_SCRIPTS <- SCRIPTS[1:7]  # the original foundation the model can't run without
n_core_success <- sum(vapply(results[CORE_SCRIPTS], function(r) r$status == "SUCCESS", logical(1)))

for (script in SCRIPTS) {
  r <- results[[script]]
  log_msg("%-35s %-8s (%.1fs)", basename(script), r$status, r$elapsed)
  files <- OUTPUT_GLOBS[[script]]
  for (f in files) {
    fpath <- file.path(RAW_DIR, f)
    if (file.exists(fpath)) {
      n <- tryCatch(nrow(read_csv(fpath, show_col_types = FALSE)), error = function(e) NA)
      log_msg("    %-40s %s rows", f, ifelse(is.na(n), "?", n))
    } else {
      log_msg("    %-40s MISSING", f)
    }
  }
}

if (n_core_success < 4) {
  log_msg("CRITICAL: fewer than 4/7 CORE collection scripts (01-07) succeeded. Data foundation is not solid.")
  log_msg("Do not proceed to feature engineering until this is investigated.")
}
n_enhancement_success <- n_success - n_core_success
log_msg("Enhancement scripts (08-13, player IDs/fielding/batters/pitcher logs/lineups/bio): %d/%d succeeded",
        n_enhancement_success, length(SCRIPTS) - length(CORE_SCRIPTS))

log_msg("========== R data collection: done ==========")
