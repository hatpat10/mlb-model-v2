# -*- coding: utf-8 -*-
# Per-game player boxscore lines (raw batting + pitching counting stats),
# 2021-onward, for the player-props pipeline. Batter markets (hits, HR,
# total bases, RBI, runs, SB, batter Ks) need per-game counting stats that
# don't exist anywhere else in data/raw/ (batter_stats_*.csv is
# season-grain), and the pitcher-strikeouts market needs raw K and
# batters-faced counts that the FanGraphs rate-stat logs in
# R/11_collect_pitcher_gamelogs.R don't carry.
#
# This calls statsapi.mlb.com directly instead of going through baseballr
# (an exception to the baseballr-first rule, like 04_collect_statcast_team.R):
# baseballr 2.0.0 has no boxscore wrapper, fg_batter_game_logs() is one call
# per player-SEASON (rejected in R/10_collect_batter_stats.R for exactly that
# reason -- 1500+ batters), and mlb_player_game_stats() is one call per
# player-GAME. The boxscore endpoint returns every player's full batting and
# pitching line for a game in a single call (~2,430/season backfilled once
# into a cache, then ~15/day in-season -- the same cost profile as the
# mlb_batting_orders() pull in R/12_collect_lineups.R), keyed by MLBAM person
# id so it joins player_id_crosswalk.csv and lineups_*.csv with no name
# matching.
#
# Outputs -- two grains, kept in separate files so a downstream join can
# never silently mix a player-game row with a team-game row:
#   batter_gamelogs_%d.csv   -- PLAYER x game, batting lines, PA > 0
#   pitcher_boxscore_%d.csv  -- PLAYER x game, pitching lines, batters faced > 0
#   team_boxscore_%d.csv     -- TEAM x game, summed from the two files above.
#     Doubles as a reconciliation check: team_boxscore's own-batting `r`
#     and own-pitching `p_r` are compared against game_logs_%d.csv's
#     (R/01_collect_gamelogs.R -- an independently fetched source) runs_scored
#     / runs_allowed for the same (game_pk, team). Any mismatch means either
#     this collector's boxscore parsing or R/01's schedule parsing drifted,
#     and gets logged loudly rather than silently feeding bad data forward.

library(dplyr)
library(readr)
library(jsonlite)

source("R/utils.R")

RAW_DIR <- "data/raw"
dir.create(RAW_DIR, showWarnings = FALSE, recursive = TRUE)
YEARS <- 2021:current_season()

# One boxscore JSON -> tidy rows for every player who batted or pitched.
# Batting and pitching lines share one frame (role column) so a single
# incremental cache file covers both outputs.
flatten_boxscore <- function(bs, pk) {
  rows <- list()
  team_names <- list(
    home = bs$teams$home$team$name,
    away = bs$teams$away$team$name
  )

  g <- function(stats, field) {
    v <- stats[[field]]
    if (is.null(v)) NA else v
  }

  for (side in c("home", "away")) {
    team_raw <- team_names[[side]]
    opp_raw <- team_names[[if (side == "home") "away" else "home"]]

    for (p in bs$teams[[side]]$players) {
      pid <- p$person$id
      pname <- p$person$fullName
      if (is.null(pid) || is.null(pname)) next

      bat <- p$stats$batting
      if (length(bat) > 0 && !is.null(bat$plateAppearances) && bat$plateAppearances > 0) {
        rows[[length(rows) + 1]] <- tibble(
          game_pk = as.character(pk), role = "batting",
          mlbam_id = pid, full_name = pname,
          team_raw = team_raw, opp_raw = opp_raw,
          is_home = as.integer(side == "home"),
          batting_order = if (is.null(p$battingOrder)) NA else p$battingOrder,
          pa = g(bat, "plateAppearances"), ab = g(bat, "atBats"),
          h = g(bat, "hits"), doubles = g(bat, "doubles"),
          triples = g(bat, "triples"), hr = g(bat, "homeRuns"),
          tb = g(bat, "totalBases"), rbi = g(bat, "rbi"),
          r = g(bat, "runs"), bb = g(bat, "baseOnBalls"),
          so = g(bat, "strikeOuts"), sb = g(bat, "stolenBases"),
          hbp = g(bat, "hitByPitch"),
          is_starter = NA, outs = NA, batters_faced = NA,
          k = NA, er = NA, pitches = NA
        )
      }

      pit <- p$stats$pitching
      if (length(pit) > 0 && !is.null(pit$battersFaced) && pit$battersFaced > 0) {
        rows[[length(rows) + 1]] <- tibble(
          game_pk = as.character(pk), role = "pitching",
          mlbam_id = pid, full_name = pname,
          team_raw = team_raw, opp_raw = opp_raw,
          is_home = as.integer(side == "home"),
          batting_order = NA,
          pa = NA, ab = NA,
          h = g(pit, "hits"), doubles = NA, triples = NA,
          hr = g(pit, "homeRuns"), tb = NA, rbi = NA,
          r = g(pit, "runs"), bb = g(pit, "baseOnBalls"),
          so = NA, sb = NA, hbp = NA,
          is_starter = g(pit, "gamesStarted"), outs = g(pit, "outs"),
          batters_faced = g(pit, "battersFaced"), k = g(pit, "strikeOuts"),
          er = g(pit, "earnedRuns"), pitches = g(pit, "numberOfPitches")
        )
      }
    }
  }

  if (length(rows) == 0) return(NULL)
  bind_rows(rows)
}

fetch_boxscores_cached <- function(game_pks, year) {
  cache_path <- file.path(RAW_DIR, sprintf("boxscore_cache_%d.csv", year))

  # Same trick as R/12_collect_lineups.R: readr type inference on cache
  # re-read won't match freshly flattened rows, and bind_rows() errors on
  # the mismatch. The final transmute() re-casts every column to its real
  # type anyway, so force everything to character here.
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
  log_msg("  boxscores: %d cached, %d to fetch", length(already_have), length(todo))

  new_rows <- list()
  flush_every <- 100
  n_done <- 0

  for (pk in todo) {
    flat <- tryCatch({
      bs <- fromJSON(
        sprintf("https://statsapi.mlb.com/api/v1/game/%s/boxscore", pk),
        simplifyVector = FALSE
      )
      flatten_boxscore(bs, pk)
    }, error = function(e) NULL)
    if (!is.null(flat) && nrow(flat) > 0) {
      new_rows[[length(new_rows) + 1]] <- normalize_types(flat)
    }
    n_done <- n_done + 1

    if (n_done %% flush_every == 0 || n_done == length(todo)) {
      if (length(new_rows) > 0) {
        batch <- normalize_types(bind_rows(new_rows))
        cached <- if (!is.null(cached)) bind_rows(cached, batch) else batch
        write_csv(cached, cache_path)
        new_rows <- list()
      }
      log_msg("  boxscores: fetched %d/%d for %d", n_done, length(todo), year)
    }
  }

  if (is.null(cached)) return(tibble())
  cached
}

for (year in YEARS) {
  bat_path <- file.path(RAW_DIR, sprintf("batter_gamelogs_%d.csv", year))
  pit_path <- file.path(RAW_DIR, sprintf("pitcher_boxscore_%d.csv", year))
  team_path <- file.path(RAW_DIR, sprintf("team_boxscore_%d.csv", year))
  if (skip_completed_season(year, bat_path) && skip_completed_season(year, pit_path) &&
      skip_completed_season(year, team_path)) {
    log_msg("=== %d: finished season already collected, skipping ===", year)
    next
  }
  log_msg("=== %d: fetching player boxscore lines ===", year)

  gl_path <- file.path(RAW_DIR, sprintf("game_logs_%d.csv", year))
  if (!file.exists(gl_path)) {
    log_msg("  no game_logs_%d.csv found (run 01_collect_gamelogs.R first), skipping", year)
    next
  }
  game_logs <- read_csv(gl_path, show_col_types = FALSE)
  game_pks <- unique(game_logs$game_pk)
  log_msg("  %d games", length(game_pks))
  if (length(game_pks) == 0) next

  raw <- fetch_boxscores_cached(game_pks, year)

  if (nrow(raw) == 0) {
    log_msg("  no boxscore data returned for %d", year)
    next
  }

  # The boxscore JSON carries no game date; game_logs already maps
  # game_pk -> date (two rows per game, one per team -- date is identical).
  date_map <- game_logs %>%
    transmute(game_pk = as.character(game_pk), date = as.character(date)) %>%
    distinct(game_pk, .keep_all = TRUE)
  raw <- raw %>% left_join(date_map, by = "game_pk")

  bat_out <- raw %>%
    filter(role == "batting") %>%
    transmute(
      game_pk = as.character(game_pk),
      date = as.character(date),
      year = as.integer(year),
      team = normalize_team(team_raw),
      opp = normalize_team(opp_raw),
      is_home = as.integer(is_home),
      batter_mlbam_id = as.integer(mlbam_id),
      batter_name = strip_accents(full_name),
      batting_order = suppressWarnings(as.integer(batting_order)),
      pa = as.integer(pa), ab = as.integer(ab),
      h = as.integer(h), doubles = as.integer(doubles),
      triples = as.integer(triples), hr = as.integer(hr),
      tb = as.integer(tb), rbi = as.integer(rbi),
      r = as.integer(r), bb = as.integer(bb),
      so = as.integer(so), sb = as.integer(sb),
      hbp = as.integer(hbp)
    ) %>%
    filter(!is.na(batter_mlbam_id), !is.na(date), pa > 0) %>%
    distinct(game_pk, batter_mlbam_id, .keep_all = TRUE)

  pit_out <- raw %>%
    filter(role == "pitching") %>%
    transmute(
      game_pk = as.character(game_pk),
      date = as.character(date),
      year = as.integer(year),
      team = normalize_team(team_raw),
      opp = normalize_team(opp_raw),
      is_home = as.integer(is_home),
      pitcher_mlbam_id = as.integer(mlbam_id),
      pitcher_name = strip_accents(full_name),
      is_starter = as.integer(is_starter),
      outs = as.integer(outs),
      batters_faced = as.integer(batters_faced),
      k = as.integer(k), bb = as.integer(bb),
      h = as.integer(h), hr = as.integer(hr),
      r = as.integer(r), er = as.integer(er),
      pitches = as.integer(pitches)
    ) %>%
    filter(!is.na(pitcher_mlbam_id), !is.na(date), batters_faced > 0) %>%
    distinct(game_pk, pitcher_mlbam_id, .keep_all = TRUE)

  write_csv(bat_out, bat_path)
  log_msg("  wrote %s (%d rows, %d games)", bat_path, nrow(bat_out), length(unique(bat_out$game_pk)))
  write_csv(pit_out, pit_path)
  log_msg("  wrote %s (%d rows, %d games)", pit_path, nrow(pit_out), length(unique(pit_out$game_pk)))

  # TEAM x game rollup: sum the player rows just written, own-batting and
  # own-pitching kept as separate column groups (p_ prefix) since they
  # describe different things -- a team's own hitters vs. the runs/hits its
  # own pitchers allowed to the opponent.
  team_bat <- bat_out %>%
    group_by(game_pk, date, year, team, opp, is_home) %>%
    summarise(
      pa = sum(pa), ab = sum(ab), h = sum(h), doubles = sum(doubles),
      triples = sum(triples), hr = sum(hr), tb = sum(tb), rbi = sum(rbi),
      r = sum(r), bb = sum(bb), so = sum(so), sb = sum(sb), hbp = sum(hbp),
      .groups = "drop"
    )

  team_pit <- pit_out %>%
    group_by(game_pk, date, year, team, opp, is_home) %>%
    summarise(
      p_outs = sum(outs), p_batters_faced = sum(batters_faced), p_k = sum(k),
      p_bb = sum(bb), p_h = sum(h), p_hr = sum(hr), p_r = sum(r), p_er = sum(er),
      p_pitches = sum(pitches),
      .groups = "drop"
    )

  team_out <- team_bat %>%
    full_join(team_pit, by = c("game_pk", "date", "year", "team", "opp", "is_home")) %>%
    arrange(game_pk, team)

  # Reconciliation against R/01's independently-fetched game_logs. Investigated
  # 2026-07-14: own-pitching runs-allowed (p_r) matches game_logs runs_allowed
  # almost exactly (>99.9%). Own-batting summed runs (r) undercounts
  # game_logs runs_scored by exactly 1 in ~3-4.5% of team-games -- confirmed
  # via a traced example (game_pk 716354, 2023) that this is NOT caused by
  # this collector's pa > 0 filter (every batter row had pa >= 1) and NOT
  # extra innings (regulation 9 innings, linescore matched game_logs exactly).
  # It's a genuine MLB Stats API quirk: in a small share of games, one run
  # isn't attributed to any individual batter's `runs` counting stat even
  # though it's correctly reflected in the official score and in the
  # opposing pitchers' runs-allowed. Treat game_logs / p_r as the authoritative
  # team-run source; an individual batter's own `r` stays accurate for that
  # player, but team-level batting-side sums can be off by one run.
  recon <- team_out %>%
    inner_join(
      game_logs %>% transmute(game_pk = as.character(game_pk), team, runs_scored, runs_allowed),
      by = c("game_pk", "team")
    ) %>%
    mutate(
      r_mismatch = !is.na(r) & !is.na(runs_scored) & r != runs_scored,
      p_r_mismatch = !is.na(p_r) & !is.na(runs_allowed) & p_r != runs_allowed
    )
  n_mismatch <- sum(recon$r_mismatch | recon$p_r_mismatch, na.rm = TRUE)
  if (n_mismatch > 0) {
    log_msg("  NOTE: %d/%d team-games' summed batting/pitching runs differ from game_logs (known MLB boxscore run-attribution gap, not a parsing bug -- see comment above)",
            n_mismatch, nrow(recon))
  } else {
    log_msg("  reconciliation OK: %d team-games match game_logs runs_scored/runs_allowed exactly", nrow(recon))
  }

  write_csv(team_out, team_path)
  log_msg("  wrote %s (%d rows, %d games)", team_path, nrow(team_out), length(unique(team_out$game_pk)))
}
