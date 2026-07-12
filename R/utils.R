# -*- coding: utf-8 -*-
# Shared helpers for team abbreviation normalization and name cleanup.
# Sourced by every R collection script — this is the single source of
# truth for mapping any upstream team code/name onto the standard 30-team set.

STANDARD_TEAMS <- c(
  "ARI", "ATL", "BAL", "BOS", "CHC", "CIN", "CLE", "COL", "CWS", "DET",
  "HOU", "KC", "LAA", "LAD", "MIA", "MIL", "MIN", "NYM", "NYY", "OAK",
  "PHI", "PIT", "SD", "SEA", "SF", "STL", "TB", "TEX", "TOR", "WSH"
)

# Alternate abbreviations seen across Baseball-Reference, FanGraphs,
# MLB Stats API, Statcast, and Retrosheet.
TEAM_ABBREV_MAP <- c(
  ARI = "ARI", AZ = "ARI",
  ATL = "ATL",
  BAL = "BAL",
  BOS = "BOS",
  CHC = "CHC", CHN = "CHC",
  CIN = "CIN",
  CLE = "CLE",
  COL = "COL",
  CWS = "CWS", CHW = "CWS", CHA = "CWS",
  DET = "DET",
  HOU = "HOU",
  KC = "KC", KCA = "KC", KCR = "KC",
  LAA = "LAA", ANA = "LAA", CAL = "LAA",
  LAD = "LAD", LAN = "LAD",
  MIA = "MIA", FLA = "MIA",
  MIL = "MIL",
  MIN = "MIN",
  NYM = "NYM", NYN = "NYM",
  NYY = "NYY", NYA = "NYY",
  OAK = "OAK", ATH = "OAK",
  PHI = "PHI",
  PIT = "PIT",
  SD = "SD", SDP = "SD", SDN = "SD",
  SEA = "SEA",
  SF = "SF", SFG = "SF", SFN = "SF",
  STL = "STL", SLN = "STL",
  TB = "TB", TBR = "TB", TBA = "TB",
  TEX = "TEX",
  TOR = "TOR",
  WSH = "WSH", WSN = "WSH", WAS = "WSH"
)

# Full franchise names (as returned by FanGraphs, MLB Stats API team.name)
TEAM_FULLNAME_MAP <- c(
  "Arizona Diamondbacks" = "ARI",
  "Atlanta Braves" = "ATL",
  "Baltimore Orioles" = "BAL",
  "Boston Red Sox" = "BOS",
  "Chicago Cubs" = "CHC",
  "Cincinnati Reds" = "CIN",
  "Cleveland Guardians" = "CLE", "Cleveland Indians" = "CLE",
  "Colorado Rockies" = "COL",
  "Chicago White Sox" = "CWS",
  "Detroit Tigers" = "DET",
  "Houston Astros" = "HOU",
  "Kansas City Royals" = "KC",
  "Los Angeles Angels" = "LAA", "Los Angeles Angels of Anaheim" = "LAA",
  "Los Angeles Dodgers" = "LAD",
  "Miami Marlins" = "MIA",
  "Milwaukee Brewers" = "MIL",
  "Minnesota Twins" = "MIN",
  "New York Mets" = "NYM",
  "New York Yankees" = "NYY",
  "Oakland Athletics" = "OAK", "Athletics" = "OAK",
  "Philadelphia Phillies" = "PHI",
  "Pittsburgh Pirates" = "PIT",
  "San Diego Padres" = "SD",
  "Seattle Mariners" = "SEA",
  "San Francisco Giants" = "SF",
  "St. Louis Cardinals" = "STL",
  "Tampa Bay Rays" = "TB",
  "Texas Rangers" = "TEX",
  "Toronto Blue Jays" = "TOR",
  "Washington Nationals" = "WSH"
)

#' Normalize any team code or full name to the standard 30-team abbreviation.
#' Returns NA (with a warning tally left to the caller) for unmapped values.
normalize_team <- function(x) {
  x <- trimws(as.character(x))
  out <- rep(NA_character_, length(x))

  is_full <- x %in% names(TEAM_FULLNAME_MAP)
  out[is_full] <- TEAM_FULLNAME_MAP[x[is_full]]

  remaining <- !is_full
  x_upper <- toupper(x)
  is_abbr <- remaining & (x_upper %in% names(TEAM_ABBREV_MAP))
  out[is_abbr] <- TEAM_ABBREV_MAP[x_upper[is_abbr]]

  out
}

#' Strip accents/diacritics from a name so downstream Python joins are
#' guaranteed ASCII (e.g. "Shohei Ohtani", "Shota Imanaga").
strip_accents <- function(x) {
  iconv(x, from = "UTF-8", to = "ASCII//TRANSLIT")
}

#' Simple structured logger used by every collection script.
log_msg <- function(...) {
  cat(sprintf("[%s] %s\n", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), sprintf(...)))
}

#' The season being played right now. Collectors loop <start>:current_season()
#' instead of a hardcoded range so they never silently stop collecting when
#' a new season starts.
current_season <- function() {
  as.integer(format(Sys.Date(), "%Y"))
}

#' TRUE when `year` is a finished (immutable) season whose output file
#' already exists — collectors skip the refetch entirely. The current
#' season always re-collects so in-season stats stay fresh.
skip_completed_season <- function(year, path) {
  year < current_season() && file.exists(path)
}
