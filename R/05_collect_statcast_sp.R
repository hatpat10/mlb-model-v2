# -*- coding: utf-8 -*-
# Pitcher-level Statcast pitch-tracking data for starting pitchers, 2020-2025.
#
# baseballr::statcast_search_pitchers() is broken as of this baseballr release:
# Baseball Savant's CSV export now returns 119 columns but the package still
# hardcodes a stale 92-column rename list, so every call errors with
# "Can't assign 92 names to a 119-column data.table". The underlying CSV
# endpoint itself works fine and already includes a real header row, so we
# bypass the package wrapper and pull it directly with data.table::fread
# (the same HTTP call baseballr makes internally, just without the broken
# post-processing step).
#
# To keep request volume sane we only pull pitches thrown by pitchers who
# actually started a game in our game logs (identified via MLBAM id from
# R/01_collect_gamelogs.R's cached mlb_probables output), rather than the
# entire league, and batch them into groups of 20 pitcher IDs per request.

library(baseballr)
library(dplyr)
library(readr)
library(data.table)

source("R/utils.R")

RAW_DIR <- "data/raw"
dir.create(RAW_DIR, showWarnings = FALSE, recursive = TRUE)
YEARS <- 2020:2025
BATCH_SIZE <- 20
MIN_IP_FOR_SP <- 50

get_starter_ids_for_year <- function(year) {
  cache_path <- file.path(RAW_DIR, sprintf("mlb_probables_cache_%d.csv", year))

  if (file.exists(cache_path)) {
    probables <- read_csv(cache_path, show_col_types = FALSE)
  } else {
    log_msg("  no cached probables for %d, fetching schedule + probables fresh", year)
    sched <- tryCatch(mlb_schedule(season = year, level_ids = "1"), error = function(e) NULL)
    if (is.null(sched) || nrow(sched) == 0) return(tibble())
    reg <- sched %>%
      filter(game_type == "R", status_detailed_state %in% c("Final", "Completed Early")) %>%
      distinct(game_pk, .keep_all = TRUE)

    rows <- list()
    for (pk in reg$game_pk) {
      res <- tryCatch(mlb_probables(game_pk = pk), error = function(e) NULL)
      if (!is.null(res) && nrow(res) > 0) rows[[length(rows) + 1]] <- res
    }
    if (length(rows) == 0) return(tibble())
    probables <- bind_rows(rows)
    write_csv(probables, cache_path)
  }

  probables %>%
    transmute(pitcher_name = strip_accents(fullName), pitcher_id = id) %>%
    filter(!is.na(pitcher_id)) %>%
    distinct(pitcher_id, .keep_all = TRUE)
}

fetch_statcast_batch <- function(pitcher_ids, year) {
  lookup_params <- paste0("pitchers_lookup%5B%5D=", pitcher_ids, collapse = "&")
  url <- paste0(
    "https://baseballsavant.mlb.com/statcast_search/csv?all=true&hfPT=&hfAB=&hfBBT=&hfPR=&hfZ=",
    "&stadium=&hfBBL=&hfNewZones=&hfGT=R%7C&hfC=&hfSea=", year, "%7C",
    "&hfSit=&hfOuts=&opponent=&pitcher_throws=&batter_stands=&hfSA=&player_type=pitcher",
    "&hfInfield=&team=&position=&hfOutfield=&hfRO=&home_road=",
    "&game_date_gt=", year, "-01-01&game_date_lt=", year, "-12-31",
    "&hfFlag=&hfPull=&metric_1=&hfInn=&min_pitches=0&min_results=0&group_by=name",
    "&sort_col=pitches&player_event_sort=h_launch_speed&sort_order=desc&min_abs=0&type=details",
    "&", lookup_params
  )
  tryCatch(fread(url, encoding = "UTF-8", showProgress = FALSE), error = function(e) {
    log_msg("    batch fetch ERROR: %s", conditionMessage(e))
    NULL
  })
}

for (year in YEARS) {
  log_msg("=== %d: pitcher-level Statcast for SPs ===", year)

  starters <- get_starter_ids_for_year(year)
  if (nrow(starters) == 0) {
    log_msg("  no starter IDs available for %d, skipping", year)
    next
  }
  log_msg("  %d unique starters to pull", nrow(starters))

  ip_lookup <- NULL
  fg_path <- file.path(RAW_DIR, sprintf("fg_sp_stats_%d.csv", year))
  if (file.exists(fg_path)) {
    ip_lookup <- read_csv(fg_path, show_col_types = FALSE) %>%
      select(pitcher_name, ip) %>%
      distinct(pitcher_name, .keep_all = TRUE)
  }

  id_batches <- split(starters$pitcher_id, ceiling(seq_along(starters$pitcher_id) / BATCH_SIZE))
  all_pitches <- list()

  for (i in seq_along(id_batches)) {
    batch_ids <- id_batches[[i]]
    res <- fetch_statcast_batch(batch_ids, year)
    if (!is.null(res) && nrow(res) > 0) all_pitches[[length(all_pitches) + 1]] <- res
    log_msg("  batch %d/%d fetched (%d pitches)", i, length(id_batches),
            if (!is.null(res)) nrow(res) else 0)
  }

  if (length(all_pitches) == 0) {
    log_msg("  no pitch-level data returned for %d, skipping", year)
    next
  }

  pitches <- bind_rows(all_pitches) %>%
    filter(!is.na(pitcher)) %>%
    left_join(starters %>% rename(pitcher = pitcher_id), by = "pitcher")

  swing_desc <- c("swinging_strike", "swinging_strike_blocked", "foul", "foul_tip",
                  "hit_into_play", "foul_bunt", "missed_bunt")
  whiff_desc <- c("swinging_strike", "swinging_strike_blocked", "missed_bunt")

  pa_ending <- pitches %>% filter(!is.na(events) & events != "")

  per_pitcher <- pitches %>%
    group_by(pitcher_name) %>%
    summarise(
      spin_rate_avg = mean(release_spin_rate, na.rm = TRUE),
      velo_avg = mean(release_speed, na.rm = TRUE),
      extension_avg = mean(release_extension, na.rm = TRUE),
      n_swings = sum(description %in% swing_desc, na.rm = TRUE),
      n_whiffs = sum(description %in% whiff_desc, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(whiff_pct = if_else(n_swings > 0, n_whiffs / n_swings, NA_real_))

  pa_summary <- pa_ending %>%
    group_by(pitcher_name) %>%
    summarise(
      k = sum(events %in% c("strikeout", "strikeout_double_play"), na.rm = TRUE),
      bb = sum(events == "walk", na.rm = TRUE),
      hbp = sum(events == "hit_by_pitch", na.rm = TRUE),
      hr = sum(events == "home_run", na.rm = TRUE),
      fb = sum(bb_type == "fly_ball", na.rm = TRUE),
      batted_balls = sum(!is.na(bb_type), na.rm = TRUE),
      barrels = sum(launch_speed_angle == 6, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(barrel_pct_against = if_else(batted_balls > 0, barrels / batted_balls, NA_real_))

  lg_hr_fb_rate <- sum(pa_summary$hr) / max(sum(pa_summary$fb), 1)

  out <- per_pitcher %>%
    left_join(pa_summary, by = "pitcher_name") %>%
    left_join(ip_lookup, by = "pitcher_name") %>%
    mutate(year = as.integer(year)) %>%
    filter(!is.na(ip), ip >= MIN_IP_FOR_SP)

  if (nrow(out) > 0 && !all(is.na(out$k))) {
    lg_era_proxy <- if (file.exists(fg_path)) {
      fg <- read_csv(fg_path, show_col_types = FALSE)
      weighted.mean(fg$era, w = pmax(fg$ip, 1), na.rm = TRUE)
    } else {
      4.10
    }
    lg_totals <- pa_summary %>%
      summarise(k = sum(k), bb = sum(bb), hbp = sum(hbp))
    # Use the SAME (all-starters) population for both the numerator (K/BB/HBP/HR)
    # and denominator (IP) so the league constant isn't biased by the later
    # per-pitcher MIN_IP_FOR_SP filter applied to `out`.
    total_ip <- pa_summary %>% left_join(ip_lookup, by = "pitcher_name") %>%
      summarise(total = sum(ip, na.rm = TRUE)) %>% pull(total)
    fip_constant <- lg_era_proxy - (((13 * sum(pa_summary$hr)) + (3 * (lg_totals$bb + lg_totals$hbp)) - (2 * lg_totals$k)) / max(total_ip, 1))

    out <- out %>%
      mutate(
        xfip_statcast = ((13 * (fb * lg_hr_fb_rate)) + (3 * (bb + hbp)) - (2 * k)) / ip + fip_constant
      )
  } else {
    out$xfip_statcast <- NA_real_
  }

  out <- out %>%
    transmute(
      pitcher_name, year, spin_rate_avg, velo_avg, extension_avg,
      xfip_statcast, whiff_pct, barrel_pct_against
    ) %>%
    distinct(pitcher_name, year, .keep_all = TRUE)

  out_path <- file.path(RAW_DIR, sprintf("statcast_sp_%d.csv", year))
  write_csv(out, out_path)
  log_msg("  wrote %s (%d rows, min %d IP)", out_path, nrow(out), MIN_IP_FOR_SP)
}
