# -*- coding: utf-8 -*-
"""Single source of truth describing every stat this pipeline can pull.

Shared by app/generate_stats_catalog.py (the read-through Word doc) and
app/stats_app.py (the search app), so a stat's label/description/units is
never written twice. Column keys match the raw CSV headers in data/raw/ and
data/processed/feature_matrix.csv exactly — see features/builder.py for how
they're joined together.
"""

SECTIONS = [
    {
        "title": "Team Results & Schedule",
        "source": "data/raw/game_logs_<year>.csv — MLB Stats API",
        "grain": "One row per team per game",
        "stats": [
            ("runs_scored", "Runs Scored", "Runs this team scored in the game.", "count"),
            ("runs_allowed", "Runs Allowed", "Runs this team allowed in the game.", "count"),
            ("win", "Win", "1 if this team won, 0 if it lost.", "0/1"),
            ("home_starter", "Home Starter", "Probable/actual home starting pitcher.", "name"),
            ("away_starter", "Away Starter", "Probable/actual away starting pitcher.", "name"),
        ],
    },
    {
        "title": "Team Batting (season)",
        "source": "data/raw/fg_team_batting_<year>.csv — FanGraphs",
        "grain": "One row per team per season",
        "stats": [
            ("wrc_plus", "wRC+", "Weighted Runs Created Plus — overall offensive production, park- and league-adjusted.", "100=avg"),
            ("woba", "wOBA", "Weighted On-Base Average — single rate stat weighting every way of reaching base by its real run value.", "rate"),
            ("obp", "OBP", "On-Base Percentage.", "rate"),
            ("slg", "SLG", "Slugging Percentage.", "rate"),
            ("iso", "ISO", "Isolated Power (SLG - AVG) — raw power production.", "rate"),
            ("bb_pct", "BB%", "Walk rate (walks per plate appearance).", "%"),
            ("k_pct", "K%", "Strikeout rate (strikeouts per plate appearance).", "%"),
            ("babip", "BABIP", "Batting Average on Balls In Play.", "rate"),
            ("ops", "OPS", "On-base Plus Slugging (OBP + SLG).", "rate"),
            ("pa", "PA", "Plate appearances — sample size behind the season's rate stats.", "count"),
        ],
    },
    {
        "title": "Team Pitching (season, aggregated from starters)",
        "source": "Derived from fg_sp_stats_<year>.csv, IP-weighted per team — features/builder.py:join_team_pitching",
        "grain": "One row per team per season",
        "stats": [
            ("team_fip", "Team FIP", "Fielding Independent Pitching — ERA estimator built only from K/BB/HR, IP-weighted across a team's starters.", "ERA-scale"),
            ("team_xfip", "Team xFIP", "FIP with home-run rate normalized to league average — strips out HR/FB luck.", "ERA-scale"),
            ("team_k_pct", "Team K%", "Team-level strikeout rate (starters).", "%"),
            ("team_bb_pct", "Team BB%", "Team-level walk rate (starters).", "%"),
        ],
    },
    {
        "title": "Team Statcast Batting Quality",
        "source": "data/raw/statcast_team_batting_<year>.csv — Baseball Savant",
        "grain": "One row per team per season",
        "stats": [
            ("barrel_pct", "Barrel%", "Share of batted balls in the ideal exit-velocity/launch-angle \"barrel\" zone — the strongest quality-of-contact indicator.", "%"),
            ("hard_hit_pct", "Hard-Hit%", "Share of batted balls hit 95+ mph.", "%"),
            ("xba", "xBA", "Expected batting average from contact quality, independent of defense/luck.", "rate"),
            ("xslg", "xSLG", "Expected slugging percentage from contact quality.", "rate"),
            ("xwoba", "xwOBA", "Expected wOBA from contact quality — the most defense/luck-independent offense metric.", "rate"),
            ("exit_velo_avg", "Avg Exit Velocity", "Average batted-ball exit velocity.", "mph"),
            ("sprint_speed_avg", "Avg Sprint Speed", "Average team baserunning sprint speed.", "ft/sec"),
        ],
    },
    {
        "title": "Team Fielding",
        "source": "data/raw/team_fielding_<year>.csv — FanGraphs/Statcast composite",
        "grain": "One row per team per season",
        "stats": [
            ("drs", "DRS", "Defensive Runs Saved — total defensive value vs. an average defender.", "runs"),
            ("uzr", "UZR", "Ultimate Zone Rating — range- and error-based defensive value.", "runs"),
            ("oaa", "OAA", "Outs Above Average — Statcast's range-based defensive metric.", "outs"),
            ("def_runs", "Defensive Runs (model input)", "Composite defensive value the model actually uses.", "runs"),
            ("error_runs", "Error Runs", "Defensive value lost specifically to errors.", "runs"),
            ("range_runs", "Range Runs", "Defensive value from range/positioning.", "runs"),
        ],
    },
    {
        "title": "Starting Pitcher Stats (season)",
        "source": "data/raw/fg_sp_stats_<year>.csv — FanGraphs",
        "grain": "One row per starting pitcher per season (min. games-started threshold)",
        "stats": [
            ("era", "ERA", "Earned Run Average.", "ERA-scale"),
            ("fip", "FIP", "Fielding Independent Pitching — ERA estimator from K/BB/HR only.", "ERA-scale"),
            ("xfip", "xFIP", "FIP with HR/FB normalized to league average.", "ERA-scale"),
            ("siera", "SIERA", "Skill-Interactive ERA — FanGraphs' most predictive ERA estimator, accounts for batted-ball mix plus K/BB.", "ERA-scale"),
            ("k_pct", "K%", "Strikeout rate.", "%"),
            ("bb_pct", "BB%", "Walk rate.", "%"),
            ("whip", "WHIP", "Walks + Hits per Inning Pitched.", "rate"),
            ("k9", "K/9", "Strikeouts per 9 innings.", "rate"),
            ("bb9", "BB/9", "Walks per 9 innings.", "rate"),
            ("hr9", "HR/9", "Home runs per 9 innings.", "rate"),
            ("swstr_pct", "SwStr%", "Swinging-strike rate — whiff-inducing ability.", "%"),
            ("ip", "IP", "Innings pitched — sample size behind the season's rate stats.", "innings"),
            ("gs", "GS", "Games started.", "count"),
            ("w", "W", "Wins (context only — not predictive; team/bullpen dependent).", "count"),
            ("l", "L", "Losses (context only — not predictive).", "count"),
        ],
    },
    {
        "title": "Starting Pitcher Statcast",
        "source": "data/raw/statcast_sp_<year>.csv — Baseball Savant",
        "grain": "One row per starting pitcher per season",
        "stats": [
            ("spin_rate_avg", "Avg Spin Rate", "Average pitch spin rate.", "rpm"),
            ("velo_avg", "Avg Velocity", "Average pitch velocity across all pitch types thrown.", "mph"),
            ("extension_avg", "Avg Extension", "Release extension toward the plate — a deception factor.", "ft"),
            ("xfip_statcast", "xFIP (Statcast)", "Statcast-derived xFIP variant.", "ERA-scale"),
            ("whiff_pct", "Whiff%", "Swinging-strike rate on swings taken against this pitcher.", "%"),
            ("barrel_pct_against", "Barrel% Against", "Rate of barrels allowed — quality of contact given up.", "%"),
        ],
    },
    {
        "title": "Starting Pitcher Rolling Form",
        "source": "data/raw/pitcher_gamelogs_<year>.csv — per-start logs — features/builder.py:join_pitcher_rolling_form",
        "grain": "One row per pitcher per start; rolling stats computed over the prior 3 starts",
        "stats": [
            ("ip", "IP (this start)", "Innings pitched in a single start.", "innings"),
            ("era", "ERA (this start)", "Earned run average for a single start.", "ERA-scale"),
            ("fip", "FIP (this start)", "FIP for a single start.", "ERA-scale"),
            ("era_roll3", "ERA (last 3 starts)", "Rolling ERA over the pitcher's 3 most recent starts — catches a hot or cold stretch a season average can't.", "ERA-scale"),
            ("fip_roll3", "FIP (last 3 starts)", "Rolling FIP over the pitcher's 3 most recent starts.", "ERA-scale"),
        ],
    },
    {
        "title": "Individual Batter Stats (season)",
        "source": "data/raw/batter_stats_<year>.csv — FanGraphs",
        "grain": "One row per batter per season (min. plate-appearance threshold)",
        "stats": [
            ("pa", "PA", "Plate appearances.", "count"),
            ("wrc_plus", "wRC+", "Weighted Runs Created Plus — park/league-adjusted overall offensive value.", "100=avg"),
            ("woba", "wOBA", "Weighted On-Base Average.", "rate"),
            ("obp", "OBP", "On-Base Percentage.", "rate"),
            ("slg", "SLG", "Slugging Percentage.", "rate"),
            ("iso", "ISO", "Isolated Power (SLG - AVG).", "rate"),
            ("bb_pct", "BB%", "Walk rate.", "%"),
            ("k_pct", "K%", "Strikeout rate.", "%"),
            ("barrel_pct", "Barrel%", "Share of batted balls in the ideal barrel zone.", "%"),
            ("hard_hit_pct", "Hard-Hit%", "Share of batted balls hit 95+ mph.", "%"),
            ("xwoba", "xwOBA", "Expected wOBA from contact quality.", "rate"),
        ],
    },
    {
        "title": "Starting Lineups",
        "source": "data/raw/lineups_<year>.csv — MLB boxscore feed",
        "grain": "One row per batter per game",
        "stats": [
            ("position", "Position", "Defensive position started.", "text"),
            ("batting_order", "Batting Order", "Spot in the batting order (1-9).", "1-9"),
        ],
    },
    {
        "title": "Player Bio",
        "source": "data/raw/player_bio.csv — MLB Stats API",
        "grain": "One row per player",
        "stats": [
            ("birth_date", "Birth Date", "Used to derive current age.", "date"),
            ("bat_side", "Bats", "Batting handedness.", "L/R/S"),
            ("throw_hand", "Throws", "Throwing handedness.", "L/R"),
            ("height", "Height", "Listed height.", "text"),
            ("weight_lbs", "Weight", "Listed weight.", "lbs"),
            ("primary_position", "Primary Position", "Primary defensive position.", "text"),
        ],
    },
    {
        "title": "Umpire Context",
        "source": "data/raw/umpire_assignments.csv + umpire_factors.csv — MLB Stats API",
        "grain": "One row per game (assignment) / one row per umpire (factor)",
        "stats": [
            ("umpire_name", "Home Plate Umpire", "Umpire assigned behind the plate for the game.", "name"),
            ("umpire_run_factor", "Umpire Run Factor", "How many runs above/below league average games with this HP umpire tend to have — a proxy for strike-zone tightness.", "runs vs. league avg"),
        ],
    },
    {
        "title": "Park Factors",
        "source": "data/raw/park_factors.csv — empirically computed from game logs",
        "grain": "One row per team (home park) per season",
        "stats": [
            ("park_factor", "Park Factor", "Run-scoring environment multiplier for the home park.", "index, 100 = neutral"),
        ],
    },
    {
        "title": "Engineered Team Form (model features)",
        "source": "data/processed/feature_matrix.csv — features/builder.py:build_rolling_features / build_elo_ratings / build_schedule_features",
        "grain": "One row per game, home- and away-prefixed",
        "stats": [
            ("win_rate_roll_5", "Win Rate (last 5)", "Rolling winning percentage over the last 5 games.", "%"),
            ("win_rate_roll_10", "Win Rate (last 10)", "Rolling winning percentage over the last 10 games.", "%"),
            ("win_rate_roll_20", "Win Rate (last 20)", "Rolling winning percentage over the last 20 games.", "%"),
            ("run_diff_roll_10", "Run Diff (last 10)", "Rolling run differential over the last 10 games.", "runs"),
            ("runs_scored_roll_10", "Runs Scored (last 10)", "Rolling average runs scored per game, last 10 games.", "runs/game"),
            ("runs_allowed_roll_10", "Runs Allowed (last 10)", "Rolling average runs allowed per game, last 10 games.", "runs/game"),
            ("pyth_win_pct", "Pyth Win% (season)", "Expected win percentage from season-to-date run differential (Pythagorean formula).", "%"),
            ("pyth_win_pct_roll_20", "Pythagorean Win% (last 20)", "Rolling Pythagorean win percentage, last 20 games.", "%"),
            ("pyth_win_pct_roll_40", "Pythagorean Win% (last 40)", "Rolling Pythagorean win percentage, last 40 games.", "%"),
            ("rest_days", "Rest Days", "Days since the team's last game, capped at 7.", "days"),
            ("travel_penalty", "Travel Penalty", "Flag for a long-distance trip (1000+ miles) on short rest (<=2 days).", "0/1"),
            ("consec_away", "Consecutive Road Games", "Road games in a row entering this game, capped at 7.", "count"),
            ("lineup_wrc_plus", "Lineup wRC+", "Actual starting lineup's average prior-season wRC+ (falls back to team average pre-lineup).", "100=avg"),
            ("platoon_advantage", "Platoon Advantage", "Share of the starting lineup with the handedness advantage against today's opposing starter.", "%"),
            ("elo_rating", "Elo Rating", "Team strength rating, updated after every game (regressed 33% toward 1500 each new season).", "Elo points"),
            ("elo_diff", "Elo Diff", "This team's Elo rating minus the opponent's.", "Elo points"),
            ("elo_win_prob", "Elo Win Probability", "Win probability implied purely by the Elo gap.", "%"),
        ],
    },
    {
        "title": "Engineered Matchup Features (model features)",
        "source": "data/processed/feature_matrix.csv — features/builder.py:join_sp_stats",
        "grain": "One row per game",
        "stats": [
            ("sp_era_adv", "SP ERA Advantage", "Away starter ERA minus home starter ERA (positive favors the home starter).", "ERA-scale"),
            ("sp_fip_adv", "SP FIP Advantage", "Away starter FIP minus home starter FIP.", "ERA-scale"),
            ("sp_k_pct_adv", "SP K% Advantage", "Home starter K% minus away starter K%.", "%"),
            ("home_sp_known", "Home SP Known", "1 if the home starter had a real (non-imputed) prior-season stat line, 0 if league-average was used instead.", "0/1"),
            ("away_sp_known", "Away SP Known", "Same as above for the away starter.", "0/1"),
        ],
    },
]


def all_stat_keys_by_section():
    """Returns {section_title: [raw_column_key, ...]} for quick lookups."""
    return {s["title"]: [row[0] for row in s["stats"]] for s in SECTIONS}


def stat_lookup():
    """Returns {raw_column_key: (label, description, units)} across all sections.

    Some column names are reused at a different grain across sections (e.g.
    "era" is both a starting pitcher's season ERA and a single start's ERA in
    the rolling-form log) — first occurrence in SECTIONS wins, which is the
    coarser/season-level definition in every such case.
    """
    lookup = {}
    for s in SECTIONS:
        for key, label, desc, units in s["stats"]:
            lookup.setdefault(key, (label, desc, units))
    return lookup
