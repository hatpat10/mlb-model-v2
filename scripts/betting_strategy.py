# -*- coding: utf-8 -*-
"""Shared moneyline sizing rules used by live bankroll and backtests."""
from dataclasses import dataclass

import numpy as np

from config.config import (
    KELLY_FRACTION,
    MAX_BET_BANKROLL_FRACTION,
    MAX_DAILY_EXPOSURE_FRACTION,
    MAX_OPEN_EXPOSURE_FRACTION,
    MIN_BET_SIZE,
)


def american_to_decimal(odds: float) -> float:
    odds = float(odds)
    if not np.isfinite(odds) or odds == 0:
        raise ValueError(f"Invalid American odds: {odds}")
    return 1.0 + (100.0 / -odds if odds < 0 else odds / 100.0)


def full_kelly_fraction(probability: float, decimal_odds: float) -> float:
    probability = float(probability)
    decimal_odds = float(decimal_odds)
    if not 0 < probability < 1:
        raise ValueError(f"Probability must be strictly between 0 and 1: {probability}")
    if decimal_odds <= 1:
        raise ValueError(f"Decimal odds must exceed 1: {decimal_odds}")
    net = decimal_odds - 1.0
    return float(np.clip((probability * decimal_odds - 1.0) / net, 0.0, 1.0))


@dataclass(frozen=True)
class StakeDecision:
    stake: float
    uncapped_stake: float
    full_kelly: float
    limiting_rule: str
    available_bankroll: float


def size_stake(
    probability: float,
    american_odds: float,
    bankroll: float,
    daily_staked: float = 0.0,
    open_staked: float = 0.0,
) -> StakeDecision:
    """Quarter-Kelly stake constrained by single, daily, and open exposure."""
    bankroll = float(bankroll)
    if bankroll <= 0:
        return StakeDecision(0.0, 0.0, 0.0, "no_bankroll", 0.0)

    full_kelly = full_kelly_fraction(probability, american_to_decimal(american_odds))
    available_bankroll = max(0.0, bankroll - float(open_staked))
    uncapped = available_bankroll * KELLY_FRACTION * full_kelly
    limits = {
        "kelly": uncapped,
        "single_bet_cap": bankroll * MAX_BET_BANKROLL_FRACTION,
        "daily_exposure_cap": max(0.0, bankroll * MAX_DAILY_EXPOSURE_FRACTION - float(daily_staked)),
        "open_exposure_cap": max(0.0, bankroll * MAX_OPEN_EXPOSURE_FRACTION - float(open_staked)),
    }
    limiting_rule, stake = min(limits.items(), key=lambda item: item[1])
    if stake < MIN_BET_SIZE:
        return StakeDecision(0.0, uncapped, full_kelly, "below_minimum_or_exhausted_cap", available_bankroll)
    return StakeDecision(round(float(stake), 2), float(uncapped), full_kelly, limiting_rule, available_bankroll)
