# -*- coding: utf-8 -*-
"""Search app: look up a 2026 team or player and see every stat that matters
for it in one place. Pulls straight from data/raw/*_2026.csv and the trained
model's feature_matrix.csv — nothing here is recomputed independently of the
pipeline, so what you see here matches what 04_predict.py sees.

Run:
    venv\\Scripts\\python.exe -m streamlit run app\\stats_app.py
"""
import difflib
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.stat_definitions import stat_lookup  # noqa: E402
from config.config import PATHS, TEAM_ABBREVS  # noqa: E402

RAW = PATHS["raw"]
PROC = PATHS["processed"]
SEASON = 2026
LABELS = stat_lookup()

TEAM_FULL_NAMES = {
    "ARI": "Arizona Diamondbacks", "ATL": "Atlanta Braves", "BAL": "Baltimore Orioles",
    "BOS": "Boston Red Sox", "CHC": "Chicago Cubs", "CIN": "Cincinnati Reds",
    "CLE": "Cleveland Guardians", "COL": "Colorado Rockies", "CWS": "Chicago White Sox",
    "DET": "Detroit Tigers", "HOU": "Houston Astros", "KC": "Kansas City Royals",
    "LAA": "Los Angeles Angels", "LAD": "Los Angeles Dodgers", "MIA": "Miami Marlins",
    "MIL": "Milwaukee Brewers", "MIN": "Minnesota Twins", "NYM": "New York Mets",
    "NYY": "New York Yankees", "OAK": "Athletics", "PHI": "Philadelphia Phillies",
    "PIT": "Pittsburgh Pirates", "SD": "San Diego Padres", "SEA": "Seattle Mariners",
    "SF": "San Francisco Giants", "STL": "St. Louis Cardinals", "TB": "Tampa Bay Rays",
    "TEX": "Texas Rangers", "TOR": "Toronto Blue Jays", "WSH": "Washington Nationals",
}

# Fixed categorical order per the dataviz palette — never reassigned per-render.
SERIES_COLOR = {
    "light": {"1": "#2a78d6", "2": "#1baf7a", "muted": "#898781", "grid": "#e1e0d9", "text2": "#52514e"},
    "dark": {"1": "#3987e5", "2": "#199e70", "muted": "#898781", "grid": "#2c2c2a", "text2": "#c3c2b7"},
}

st.set_page_config(page_title="MLB Model V2 — Stat Search", page_icon="⚾", layout="wide")


def get_theme() -> str:
    try:
        t = st.context.theme.type
        return t if t in ("light", "dark") else "light"
    except Exception:
        return "light"


@st.cache_data(ttl=300)
def load_data():
    def _read(name, **kw):
        p = RAW / name
        return pd.read_csv(p, **kw) if p.exists() else pd.DataFrame()

    d = {
        "game_logs": _read(f"game_logs_{SEASON}.csv"),
        "team_batting": _read(f"fg_team_batting_{SEASON}.csv"),
        "sp_stats": _read(f"fg_sp_stats_{SEASON}.csv"),
        "statcast_team": _read(f"statcast_team_batting_{SEASON}.csv"),
        "statcast_sp": _read(f"statcast_sp_{SEASON}.csv"),
        "team_fielding": _read(f"team_fielding_{SEASON}.csv"),
        "batter_stats": _read(f"batter_stats_{SEASON}.csv"),
        "pitcher_gamelogs": _read(f"pitcher_gamelogs_{SEASON}.csv"),
        "lineups": _read(f"lineups_{SEASON}.csv"),
        "player_bio": _read("player_bio.csv"),
        "park_factors": _read("park_factors.csv"),
    }
    fm_path = PROC / "feature_matrix.csv"
    fm = pd.read_csv(fm_path) if fm_path.exists() else pd.DataFrame()
    d["feature_matrix"] = fm[fm["year"] == SEASON] if not fm.empty else fm
    if not d["park_factors"].empty:
        d["park_factors"] = d["park_factors"][d["park_factors"]["year"] == SEASON]
    return d


DATA = load_data()

FORM_COLS = [
    "win_rate_roll_5", "win_rate_roll_10", "win_rate_roll_20", "run_diff_roll_10",
    "runs_scored_roll_10", "runs_allowed_roll_10", "pyth_win_pct",
    "pyth_win_pct_roll_20", "pyth_win_pct_roll_40", "elo_rating",
]


def team_current_form(abbr: str) -> dict:
    fm = DATA["feature_matrix"]
    if fm.empty:
        return {}
    home = fm[fm["home_team"] == abbr][["date"] + [f"home_{c}" for c in FORM_COLS]]
    home = home.rename(columns={f"home_{c}": c for c in FORM_COLS})
    away = fm[fm["away_team"] == abbr][["date"] + [f"away_{c}" for c in FORM_COLS]]
    away = away.rename(columns={f"away_{c}": c for c in FORM_COLS})
    combined = pd.concat([home, away], ignore_index=True).sort_values("date")
    return combined.iloc[-1].to_dict() if len(combined) else {}


def metric(col, key, value, fmt="{:.3f}"):
    label, desc, units = LABELS.get(key, (key, "", ""))
    text = "—" if pd.isna(value) else fmt.format(value)
    col.metric(f"{label} ({units})" if units else label, text, help=desc)


def metric_grid(items, per_row=4):
    """items: list of (key, value, fmt) tuples, wrapped at `per_row` columns
    so long labels (with units) don't get clipped."""
    for i in range(0, len(items), per_row):
        chunk = items[i:i + per_row]
        cols = st.columns(len(chunk))
        for col, (key, value, fmt) in zip(cols, chunk):
            metric(col, key, value, fmt)


def find_team_matches(query: str) -> list:
    q = query.strip().lower()
    if not q:
        return []
    hits = []
    for abbr in TEAM_ABBREVS:
        full = TEAM_FULL_NAMES.get(abbr, "")
        if q == abbr.lower() or q in abbr.lower() or q in full.lower():
            hits.append(abbr)
    return hits


def all_player_names() -> set:
    names = set()
    for key, col in [("sp_stats", "pitcher_name"), ("statcast_sp", "pitcher_name"),
                      ("pitcher_gamelogs", "pitcher_name"), ("batter_stats", "batter_name"),
                      ("lineups", "batter_name"), ("player_bio", "full_name")]:
        df = DATA[key]
        if not df.empty and col in df.columns:
            names |= set(df[col].dropna().unique())
    return names


def find_player_matches(query: str) -> list:
    q = query.strip().lower()
    if not q:
        return []
    names = all_player_names()
    substr = sorted(n for n in names if q in n.lower())
    if substr:
        return substr[:15]
    return difflib.get_close_matches(query, names, n=8, cutoff=0.6)


def show_team(abbr: str):
    full = TEAM_FULL_NAMES.get(abbr, abbr)
    st.header(f"{full} ({abbr}) — {SEASON}")

    gl = DATA["game_logs"]
    team_games = gl[gl["team"] == abbr].sort_values("date") if not gl.empty else pd.DataFrame()
    wins = int(team_games["win"].sum()) if len(team_games) else 0
    losses = len(team_games) - wins if len(team_games) else 0

    form = team_current_form(abbr)

    st.subheader("Record & Current Form")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Record", f"{wins}-{losses}" if len(team_games) else "—")
    metric(c2, "win_rate_roll_10", form.get("win_rate_roll_10"), "{:.1%}")
    metric(c3, "run_diff_roll_10", form.get("run_diff_roll_10"), "{:+.2f}")
    metric(c4, "pyth_win_pct", form.get("pyth_win_pct"), "{:.1%}")
    c5, _, _, _ = st.columns(4)
    metric(c5, "elo_rating", form.get("elo_rating"), "{:.0f}")

    tb = DATA["team_batting"]
    tb_row = tb[tb["team"] == abbr].iloc[0] if not tb.empty and (tb["team"] == abbr).any() else None
    sct = DATA["statcast_team"]
    sct_row = sct[sct["team"] == abbr].iloc[0] if not sct.empty and (sct["team"] == abbr).any() else None

    st.subheader("Offense (season)")
    if tb_row is not None:
        metric_grid([
            ("wrc_plus", tb_row["wrc_plus"], "{:.0f}"), ("woba", tb_row["woba"], "{:.3f}"),
            ("obp", tb_row["obp"], "{:.3f}"), ("slg", tb_row["slg"], "{:.3f}"),
            ("bb_pct", tb_row["bb_pct"], "{:.1%}"), ("k_pct", tb_row["k_pct"], "{:.1%}"),
        ])
    else:
        st.caption("No team batting data yet for 2026.")
    if sct_row is not None:
        c1, c2, c3, c4 = st.columns(4)
        metric(c1, "barrel_pct", sct_row["barrel_pct"], "{:.1%}")
        metric(c2, "hard_hit_pct", sct_row["hard_hit_pct"], "{:.1%}")
        metric(c3, "xwoba", sct_row["xwoba"])
        metric(c4, "exit_velo_avg", sct_row["exit_velo_avg"], "{:.1f}")

    tf = DATA["team_fielding"]
    tf_row = tf[tf["team"] == abbr].iloc[0] if not tf.empty and (tf["team"] == abbr).any() else None
    st.subheader("Defense (season)")
    if tf_row is not None:
        c1, c2, c3 = st.columns(3)
        metric(c1, "drs", tf_row["drs"], "{:.0f}")
        metric(c2, "oaa", tf_row["oaa"], "{:.0f}")
        metric(c3, "def_runs", tf_row["def_runs"], "{:.1f}")
    else:
        st.caption("No team fielding data yet for 2026.")

    pf = DATA["park_factors"]
    pf_row = pf[pf["team"] == abbr] if not pf.empty else pd.DataFrame()
    if len(pf_row):
        st.metric(f"{LABELS['park_factor'][0]} ({LABELS['park_factor'][2]})",
                   f"{pf_row.iloc[0]['park_factor']:.1f}", help=LABELS["park_factor"][1])

    st.subheader("Starting Pitchers (season)")
    sp = DATA["sp_stats"]
    sc = DATA["statcast_sp"]
    team_sp = sp[sp["team"] == abbr] if not sp.empty else pd.DataFrame()
    if len(team_sp):
        merged = team_sp.merge(sc[["pitcher_name", "year", "velo_avg", "whiff_pct"]], on=["pitcher_name", "year"], how="left") if not sc.empty else team_sp
        cols = ["pitcher_name", "era", "fip", "xfip", "siera", "k_pct", "bb_pct", "whip", "k9", "velo_avg", "whiff_pct", "ip", "gs"]
        cols = [c for c in cols if c in merged.columns]
        st.dataframe(merged[cols].sort_values("ip", ascending=False).reset_index(drop=True), use_container_width=True)
    else:
        st.caption("No starting-pitcher data yet for 2026.")

    st.subheader("Recent Games")
    if len(team_games):
        recent = team_games.tail(10)[["date", "opponent", "is_home", "runs_scored", "runs_allowed", "win", "home_starter", "away_starter"]].copy()
        recent["is_home"] = recent["is_home"].map({1: "Home", 0: "Away"})
        recent["win"] = recent["win"].map({1: "W", 0: "L"})
        st.dataframe(recent.sort_values("date", ascending=False).reset_index(drop=True), use_container_width=True)


def rolling_form_chart(pg: pd.DataFrame):
    theme = get_theme()
    c = SERIES_COLOR[theme]
    pg = pg.sort_values("date").copy()
    pg["era_roll3"] = pg["era"].rolling(3, min_periods=1).mean()
    pg["fip_roll3"] = pg["fip"].rolling(3, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pg["date"], y=pg["era_roll3"], name="ERA (last 3 starts)", mode="lines+markers",
        line=dict(color=c["1"], width=2, shape="spline"), marker=dict(size=8),
    ))
    fig.add_trace(go.Scatter(
        x=pg["date"], y=pg["fip_roll3"], name="FIP (last 3 starts)", mode="lines+markers",
        line=dict(color=c["2"], width=2, shape="spline"), marker=dict(size=8),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified", margin=dict(l=10, r=10, t=40, b=10), height=340,
        font=dict(color=c["text2"]),
        xaxis=dict(showgrid=False, color=c["muted"]),
        yaxis=dict(showgrid=True, gridcolor=c["grid"], zeroline=False, color=c["muted"], title="ERA / FIP scale"),
    )
    st.plotly_chart(fig, use_container_width=True)


def show_player(name: str):
    st.header(f"{name} — {SEASON}")

    bio = DATA["player_bio"]
    bio_row = bio[bio["full_name"] == name] if not bio.empty else pd.DataFrame()
    if len(bio_row):
        r = bio_row.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Bats", r.get("bat_side", "—"))
        c2.metric("Throws", r.get("throw_hand", "—"))
        c3.metric("Position", r.get("primary_position", "—"))
        if pd.notna(r.get("birth_date")):
            age = (pd.Timestamp.today() - pd.to_datetime(r["birth_date"])).days // 365
            c4.metric("Age", age)

    sp = DATA["sp_stats"]
    sp_row = sp[sp["pitcher_name"] == name] if not sp.empty else pd.DataFrame()
    sc = DATA["statcast_sp"]
    sc_row = sc[sc["pitcher_name"] == name] if not sc.empty else pd.DataFrame()
    pg = DATA["pitcher_gamelogs"]
    pg_rows = pg[pg["pitcher_name"] == name] if not pg.empty else pd.DataFrame()

    is_pitcher = len(sp_row) or len(sc_row) or len(pg_rows)
    if is_pitcher:
        st.subheader("Starting Pitcher — Season Line")
        if len(sp_row):
            r = sp_row.iloc[0]
            metric_grid([
                ("era", r["era"], "{:.2f}"), ("fip", r["fip"], "{:.2f}"),
                ("xfip", r["xfip"], "{:.2f}"), ("siera", r["siera"], "{:.2f}"),
                ("k_pct", r["k_pct"], "{:.1%}"), ("bb_pct", r["bb_pct"], "{:.1%}"),
                ("whip", r["whip"], "{:.2f}"),
            ])
        if len(sc_row):
            r = sc_row.iloc[0]
            c1, c2, c3, c4 = st.columns(4)
            metric(c1, "velo_avg", r["velo_avg"], "{:.1f}")
            metric(c2, "spin_rate_avg", r["spin_rate_avg"], "{:.0f}")
            metric(c3, "whiff_pct", r["whiff_pct"], "{:.1%}")
            metric(c4, "barrel_pct_against", r["barrel_pct_against"], "{:.1%}")

        if len(pg_rows) >= 1:
            st.subheader("Rolling Form (last 3 starts)")
            rolling_form_chart(pg_rows)
            st.subheader("Game Log")
            cols = [c for c in ["date", "opp", "home_away", "ip", "era", "fip", "xfip", "k_pct", "bb_pct", "hr9", "whip"] if c in pg_rows.columns]
            st.dataframe(pg_rows.sort_values("date", ascending=False)[cols].reset_index(drop=True), use_container_width=True)

    bs = DATA["batter_stats"]
    bs_row = bs[bs["batter_name"] == name] if not bs.empty else pd.DataFrame()
    if len(bs_row):
        st.subheader("Batter — Season Line")
        r = bs_row.iloc[0]
        metric_grid([
            ("wrc_plus", r["wrc_plus"], "{:.0f}"), ("woba", r["woba"], "{:.3f}"),
            ("obp", r["obp"], "{:.3f}"), ("slg", r["slg"], "{:.3f}"),
            ("bb_pct", r["bb_pct"], "{:.1%}"), ("k_pct", r["k_pct"], "{:.1%}"),
            ("barrel_pct", r["barrel_pct"], "{:.1%}"), ("hard_hit_pct", r["hard_hit_pct"], "{:.1%}"),
            ("xwoba", r["xwoba"], "{:.3f}"), ("pa", r["pa"], "{:.0f}"),
        ])

    lu = DATA["lineups"]
    lu_rows = lu[lu["batter_name"] == name] if not lu.empty else pd.DataFrame()
    if len(lu_rows):
        st.subheader("Recent Lineup Appearances")
        st.dataframe(
            lu_rows[["game_pk", "position", "batting_order", "is_home"]].tail(10).reset_index(drop=True),
            use_container_width=True,
        )

    if not is_pitcher and not len(bs_row):
        st.info("Found this player's name but no 2026 stat line yet (insufficient PA/GS, or data hasn't been collected for them).")


# --------------------------------------------------------------------------
st.title("⚾ MLB Model V2 — Stat Search")
st.caption(f"Every stat the model can see, scoped to the {SEASON} season. Search a team (e.g. \"BAL\" or \"Orioles\") or a player name.")

query = st.text_input("Search", placeholder="e.g. Orioles, BAL, or a player's name")

if query:
    team_hits = find_team_matches(query)
    player_hits = find_player_matches(query)

    if team_hits and player_hits:
        tab1, tab2 = st.tabs([f"Teams ({len(team_hits)})", f"Players ({len(player_hits)})"])
        with tab1:
            pick = st.selectbox("Team", team_hits, format_func=lambda a: f"{TEAM_FULL_NAMES.get(a, a)} ({a})", key="team_pick_both")
            show_team(pick)
        with tab2:
            pick = st.selectbox("Player", player_hits, key="player_pick_both")
            show_player(pick)
    elif team_hits:
        pick = team_hits[0] if len(team_hits) == 1 else st.selectbox(
            "Multiple teams matched", team_hits, format_func=lambda a: f"{TEAM_FULL_NAMES.get(a, a)} ({a})")
        show_team(pick)
    elif player_hits:
        pick = player_hits[0] if len(player_hits) == 1 else st.selectbox("Multiple players matched", player_hits)
        show_player(pick)
    else:
        st.warning(f"No team or player found matching \"{query}\".")
else:
    st.info("Type a team name/abbreviation or a player's name to get started.")
    with st.expander("All 30 teams"):
        st.write(", ".join(f"{a} — {TEAM_FULL_NAMES[a]}" for a in TEAM_ABBREVS))
