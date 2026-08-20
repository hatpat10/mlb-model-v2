# -*- coding: utf-8 -*-
"""Shared odds math + schedule lookup for scripts/04_predict.py,
scripts/06_runner.py and scripts/07_capture_closing_lines.py. The odds math
needs to be identical (per-book-de-vig-then-median aggregation) so the
"edge" computed at prediction time and the "closing line" stored for CLV
are on exactly the same scale; `fetch_slate` is shared so every caller
gets game start times from the one MLB Stats API call shape.
"""
import numpy as np
import pandas as pd
import requests
from datetime import datetime

from features import builder

MIN_BOOKMAKERS = 3
MLB_API = "https://statsapi.mlb.com/api/v1"


def fetch_slate(date: str) -> pd.DataFrame:
    """A day's regular-season games from the MLB Stats API: game_pk, team
    names + normalized abbreviations, and first-pitch time (UTC)."""
    resp = requests.get(f"{MLB_API}/schedule", params={"sportId": 1, "date": date, "hydrate": "team"}, timeout=30)
    resp.raise_for_status()
    rows = []
    for date_entry in resp.json().get("dates", []):
        for g in date_entry.get("games", []):
            if g.get("gameType") != "R":
                continue
            home = g["teams"]["home"]["team"]
            away = g["teams"]["away"]["team"]
            rows.append({
                "game_pk": str(g["gamePk"]),
                "home_team_name": home.get("name"),
                "away_team_name": away.get("name"),
                "home_team": home.get("abbreviation"),
                "away_team": away.get("abbreviation"),
                "start_utc": datetime.fromisoformat(g["gameDate"].replace("Z", "+00:00")),
            })
    slate = pd.DataFrame(rows)
    if not slate.empty:
        slate["home_team"] = builder.normalize_team_abbrev(slate["home_team"])
        slate["away_team"] = builder.normalize_team_abbrev(slate["away_team"])
    return slate


def american_to_implied_prob(odds):
    odds = np.asarray(odds, dtype=float)
    # np.where evaluates both branches eagerly over the whole array, so an
    # odds value of exactly +100 (a perfectly normal even-money underdog
    # price) triggers a spurious divide-by-zero warning in the *discarded*
    # negative-odds branch. errstate silences it — the selected result is
    # unaffected either way.
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(odds < 0, -odds / (-odds + 100), 100 / (odds + 100))


def american_to_decimal_odds(odds):
    """American price -> decimal payout multiplier (stake included), e.g.
    +150 -> 2.50, -120 -> 1.8333. Used for pricing actual bet payouts —
    never for computing an edge, where the de-vigged fair PROBABILITY
    (american_to_implied_prob, then no-vigged) is the correct comparison.
    """
    odds = np.asarray(odds, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(odds < 0, 1 + 100 / -odds, 1 + odds / 100)


def aggregate_h2h_event(event, min_bookmakers=MIN_BOOKMAKERS):
    """Collapse one The-Odds-API event's h2h books into a market consensus.

    Aggregates books as no-vig PROBABILITIES, never as raw American prices.
    American odds have a discontinuity at the favorite/dog boundary
    (...-105, -101, +101, +105... with no path through 0), so if books
    disagree on which side is favored in a near-toss-up game, a median/mean
    of the raw prices lands in that gap and decodes to a nonsense
    probability (e.g. -1.5 -> ~1.5%, when the true consensus is close to a
    coin flip). Converting each book to a probability first avoids that.

    Returns None when fewer than `min_bookmakers` books quote the game —
    a single stale/outlier book (e.g. one already showing an
    in-play-looking price like -10000) isn't a market consensus.

    Otherwise returns a dict with a consensus probability plus the best
    independently executable price for each side across the quoted books.
    The consensus determines edge; the side-specific best price determines
    payout and stake. This explicitly models line shopping and never creates
    a synthetic American price.
    """
    home_name = event.get("home_team")
    away_name = event.get("away_team")

    book_no_vig_home, book_prices = [], []
    for book in event.get("bookmakers", []):
        for market in book.get("markets", []):
            if market["key"] != "h2h":
                continue
            home_price = away_price = None
            for outcome in market["outcomes"]:
                if outcome["name"] == home_name:
                    home_price = outcome["price"]
                elif outcome["name"] == away_name:
                    away_price = outcome["price"]
            if home_price is None or away_price is None:
                continue
            p_home = american_to_implied_prob(home_price)
            p_away = american_to_implied_prob(away_price)
            book_no_vig_home.append(float(p_home / (p_home + p_away)))
            book_prices.append({
                "home_ml": home_price,
                "away_ml": away_price,
                "bookmaker_key": book.get("key"),
                "bookmaker_title": book.get("title"),
                "bookmaker_last_update": market.get("last_update") or book.get("last_update"),
            })

    if len(book_no_vig_home) < min_bookmakers:
        return None

    no_vig_home_implied = float(np.median(book_no_vig_home))
    closest_idx = int(np.argmin(np.abs(np.array(book_no_vig_home) - no_vig_home_implied)))
    representative = book_prices[closest_idx]
    best_home = max(book_prices, key=lambda price: float(price["home_ml"]))
    best_away = max(book_prices, key=lambda price: float(price["away_ml"]))
    probabilities = np.asarray(book_no_vig_home, dtype=float)

    return {
        "no_vig_home_implied": no_vig_home_implied,
        "home_ml": best_home["home_ml"],
        "away_ml": best_away["away_ml"],
        "home_bookmaker_key": best_home.get("bookmaker_key"),
        "home_bookmaker_title": best_home.get("bookmaker_title"),
        "home_bookmaker_last_update": best_home.get("bookmaker_last_update"),
        "away_bookmaker_key": best_away.get("bookmaker_key"),
        "away_bookmaker_title": best_away.get("bookmaker_title"),
        "away_bookmaker_last_update": best_away.get("bookmaker_last_update"),
        "representative_home_ml": representative["home_ml"],
        "representative_away_ml": representative["away_ml"],
        "bookmaker_key": representative.get("bookmaker_key"),
        "bookmaker_title": representative.get("bookmaker_title"),
        "bookmaker_last_update": representative.get("bookmaker_last_update"),
        "book_prob_std": float(probabilities.std(ddof=0)),
        "book_prob_range": float(probabilities.max() - probabilities.min()),
        "price_selection_method": "best_available_across_quoted_books",
        "n_books": len(book_no_vig_home),
    }
