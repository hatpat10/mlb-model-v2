# -*- coding: utf-8 -*-
"""Central configuration for the MLB moneyline model."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PATHS = {
    "root": PROJECT_ROOT,
    "raw": PROJECT_ROOT / "data" / "raw",
    "processed": PROJECT_ROOT / "data" / "processed",
    "models": PROJECT_ROOT / "models",
    "outputs": PROJECT_ROOT / "outputs",
    "logs": PROJECT_ROOT / "logs",
    "r_scripts": PROJECT_ROOT / "R",
}

for _p in PATHS.values():
    _p.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Betting parameters
# --------------------------------------------------------------------------
MIN_EDGE_MONEYLINE = 0.06
MAX_EDGE_MONEYLINE = 0.20
KELLY_FRACTION = 0.25
DRAWDOWN_PAUSE_THRESHOLD = 0.20

# --------------------------------------------------------------------------
# Modeling windows
# --------------------------------------------------------------------------
TRAIN_YEARS = [2021, 2022, 2023]
TEST_YEAR = 2024
CALIBRATION_YEARS = [2022, 2023]
ALL_DATA_YEARS = [2020, 2021, 2022, 2023, 2024, 2025, 2026]

# --------------------------------------------------------------------------
# Standard 30-team abbreviation set
# --------------------------------------------------------------------------
TEAM_ABBREVS = [
    "ARI", "ATL", "BAL", "BOS", "CHC", "CIN", "CLE", "COL", "CWS", "DET",
    "HOU", "KC", "LAA", "LAD", "MIA", "MIL", "MIN", "NYM", "NYY", "OAK",
    "PHI", "PIT", "SD", "SEA", "SF", "STL", "TB", "TEX", "TOR", "WSH",
]

# Maps common alternate/legacy abbreviations (from Baseball Reference,
# FanGraphs, MLB Stats API, Statcast, Retrosheet) onto the standard set above.
TEAM_ABBREV_MAP = {
    "ARI": "ARI", "AZ": "ARI",
    "ATL": "ATL",
    "BAL": "BAL",
    "BOS": "BOS",
    "CHC": "CHC", "CHN": "CHC",
    "CIN": "CIN",
    "CLE": "CLE",
    "COL": "COL",
    "CWS": "CWS", "CHW": "CWS", "CHA": "CWS",
    "DET": "DET",
    "HOU": "HOU",
    "KC": "KC", "KCA": "KC", "KCR": "KC",
    "LAA": "LAA", "ANA": "LAA", "CAL": "LAA",
    "LAD": "LAD", "LAN": "LAD",
    "MIA": "MIA", "FLA": "MIA",
    "MIL": "MIL",
    "MIN": "MIN",
    "NYM": "NYM", "NYN": "NYM",
    "NYY": "NYY", "NYA": "NYY",
    "OAK": "OAK", "ATH": "OAK",
    "PHI": "PHI",
    "PIT": "PIT",
    "SD": "SD", "SDP": "SD", "SDN": "SD",
    "SEA": "SEA",
    "SF": "SF", "SFG": "SF", "SFN": "SF",
    "STL": "STL", "SLN": "STL",
    "TB": "TB", "TBR": "TB", "TBA": "TB",
    "TEX": "TEX",
    "TOR": "TOR",
    "WSH": "WSH", "WSN": "WSH", "WAS": "WSH",
}

# --------------------------------------------------------------------------
# Stadium coordinates (home venue, lat/lon) — used for travel_penalty feature
# --------------------------------------------------------------------------
STADIUM_COORDS = {
    "ARI": (33.4455, -112.0667),   # Chase Field
    "ATL": (33.8907, -84.4677),    # Truist Park
    "BAL": (39.2839, -76.6217),    # Oriole Park at Camden Yards
    "BOS": (42.3467, -71.0972),    # Fenway Park
    "CHC": (41.9484, -87.6553),    # Wrigley Field
    "CIN": (39.0975, -84.5061),    # Great American Ball Park
    "CLE": (41.4962, -81.6852),    # Progressive Field
    "COL": (39.7559, -104.9942),   # Coors Field
    "CWS": (41.8299, -87.6338),    # Rate Field (Guaranteed Rate Field)
    "DET": (42.3390, -83.0485),    # Comerica Park
    "HOU": (29.7573, -95.3555),    # Minute Maid Park
    "KC": (39.0517, -94.4803),     # Kauffman Stadium
    "LAA": (33.8003, -117.8827),   # Angel Stadium
    "LAD": (34.0739, -118.2400),   # Dodger Stadium
    "MIA": (25.7781, -80.2196),    # loanDepot park
    "MIL": (43.0280, -87.9712),    # American Family Field
    "MIN": (44.9817, -93.2776),    # Target Field
    "NYM": (40.7571, -73.8458),    # Citi Field
    "NYY": (40.8296, -73.9262),    # Yankee Stadium
    "OAK": (37.7516, -122.2005),   # Oakland Coliseum
    "PHI": (39.9061, -75.1665),    # Citizens Bank Park
    "PIT": (40.4469, -80.0057),    # PNC Park
    "SD": (32.7073, -117.1566),    # Petco Park
    "SEA": (47.5914, -122.3325),   # T-Mobile Park
    "SF": (37.7786, -122.3893),    # Oracle Park
    "STL": (38.6226, -90.1928),    # Busch Stadium
    "TB": (27.7683, -82.6534),     # Tropicana Field
    "TEX": (32.7473, -97.0842),    # Globe Life Field
    "TOR": (43.6414, -79.3894),    # Rogers Centre
    "WSH": (38.8730, -77.0074),    # Nationals Park
}

# --------------------------------------------------------------------------
# League-average SP stats by year — used to impute when a probable starter
# cannot be matched to a FanGraphs/Statcast record (rookie, call-up, etc).
# Values are broad MLB seasonal norms; used only as fallback imputation,
# never as a real feature signal.
# --------------------------------------------------------------------------
LEAGUE_AVG_SP = {
    2020: {"era": 4.45, "fip": 4.45, "xfip": 4.45, "siera": 4.35, "k_pct": 0.235,
           "bb_pct": 0.090, "whip": 1.32, "k9": 8.8, "bb9": 3.4, "hr9": 1.30,
           "swstr_pct": 0.110, "velo_avg": 92.8, "whiff_pct": 0.240},
    2021: {"era": 4.30, "fip": 4.20, "xfip": 4.20, "siera": 4.15, "k_pct": 0.240,
           "bb_pct": 0.085, "whip": 1.28, "k9": 9.0, "bb9": 3.3, "hr9": 1.25,
           "swstr_pct": 0.112, "velo_avg": 93.0, "whiff_pct": 0.245},
    2022: {"era": 4.10, "fip": 4.05, "xfip": 4.05, "siera": 4.05, "k_pct": 0.225,
           "bb_pct": 0.082, "whip": 1.27, "k9": 8.6, "bb9": 3.2, "hr9": 1.05,
           "swstr_pct": 0.110, "velo_avg": 93.2, "whiff_pct": 0.243},
    2023: {"era": 4.35, "fip": 4.25, "xfip": 4.25, "siera": 4.20, "k_pct": 0.222,
           "bb_pct": 0.084, "whip": 1.31, "k9": 8.5, "bb9": 3.3, "hr9": 1.20,
           "swstr_pct": 0.111, "velo_avg": 93.5, "whiff_pct": 0.242},
    2024: {"era": 4.05, "fip": 4.00, "xfip": 4.05, "siera": 4.00, "k_pct": 0.228,
           "bb_pct": 0.080, "whip": 1.26, "k9": 8.8, "bb9": 3.1, "hr9": 1.10,
           "swstr_pct": 0.113, "velo_avg": 93.8, "whiff_pct": 0.246},
    2025: {"era": 4.10, "fip": 4.05, "xfip": 4.05, "siera": 4.00, "k_pct": 0.228,
           "bb_pct": 0.080, "whip": 1.27, "k9": 8.8, "bb9": 3.1, "hr9": 1.10,
           "swstr_pct": 0.113, "velo_avg": 94.0, "whiff_pct": 0.246},
    2026: {"era": 4.10, "fip": 4.05, "xfip": 4.05, "siera": 4.00, "k_pct": 0.228,
           "bb_pct": 0.080, "whip": 1.27, "k9": 8.8, "bb9": 3.1, "hr9": 1.10,
           "swstr_pct": 0.113, "velo_avg": 94.0, "whiff_pct": 0.246},
}
