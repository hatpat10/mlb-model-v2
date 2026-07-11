# -*- coding: utf-8 -*-
"""Shared odds math for scripts/04_predict.py and
scripts/07_capture_closing_lines.py. Both need the identical
per-book-de-vig-then-median aggregation so that the "edge" computed at
prediction time and the "closing line" stored for CLV are on exactly the
same scale — keeping this in one module is what guarantees that.
"""
import numpy as np

MIN_BOOKMAKERS = 3


def american_to_implied_prob(odds):
    odds = np.asarray(odds, dtype=float)
    # np.where evaluates both branches eagerly over the whole array, so an
    # odds value of exactly +100 (a perfectly normal even-money underdog
    # price) triggers a spurious divide-by-zero warning in the *discarded*
    # negative-odds branch. errstate silences it — the selected result is
    # unaffected either way.
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(odds < 0, -odds / (-odds + 100), 100 / (odds + 100))


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

    Otherwise returns a dict with:
      no_vig_home_implied — median of each book's own no-vig home probability
      home_ml / away_ml   — a real, quotable price from the book whose
                            no-vig probability is closest to that median
                            (for bet sizing; never a synthetic price nobody
                            is actually offering)
      n_books             — how many books contributed
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
            book_prices.append((home_price, away_price))

    if len(book_no_vig_home) < min_bookmakers:
        return None

    no_vig_home_implied = float(np.median(book_no_vig_home))
    closest_idx = int(np.argmin(np.abs(np.array(book_no_vig_home) - no_vig_home_implied)))
    home_ml, away_ml = book_prices[closest_idx]

    return {
        "no_vig_home_implied": no_vig_home_implied,
        "home_ml": home_ml,
        "away_ml": away_ml,
        "n_books": len(book_no_vig_home),
    }
