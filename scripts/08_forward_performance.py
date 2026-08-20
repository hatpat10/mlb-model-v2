# -*- coding: utf-8 -*-
"""Report actual forward paper/live performance from the durable bet ledger.

This report deliberately excludes retrospective closing-line replays. Only
bets that were recorded before their games and later settled belong here.
Older ledger rows remain visible in an all-history summary, but they cannot
validate the current production process unless their model and decision
lineage is complete.
"""
import json
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config.config import (  # noqa: E402
    PATHS, MIN_STRATEGY_VALIDATION_BETS, MIN_FORWARD_VALIDATION_BETTING_DAYS,
    MIN_FORWARD_VALIDATION_CLV_COVERAGE,
)
from artifact_utils import atomic_write_json, utc_now_iso  # noqa: E402
from performance_metrics import summarize_settled_bets  # noqa: E402

BET_LOG = PATHS["outputs"] / "bet_log.csv"
STATE_PATH = PATHS["models"] / "bankroll_state.json"
REPORT_PATH = PATHS["outputs"] / "forward_performance.json"
LOG_PATH = PATHS["logs"] / "08_forward_performance.log"

logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add(LOG_PATH, level="DEBUG", rotation="5 MB")

LINEAGE_COLUMNS = [
    "run_id", "model_version", "model_mode", "data_version", "feature_build_id",
    "odds_event_id", "bookmaker_key", "price_universe", "decision_timestamp", "start_utc", "execution_mode",
]


def _present(series: pd.Series) -> pd.Series:
    return series.notna() & series.astype(str).str.strip().ne("")


def build_report(ledger: pd.DataFrame, starting_bankroll: float = 10000.0) -> dict:
    status = ledger.get("status", pd.Series(index=ledger.index, dtype=object)).astype(str).str.lower()
    settled = ledger[status.eq("settled")].copy()
    for column in LINEAGE_COLUMNS:
        if column not in settled.columns:
            settled[column] = pd.NA

    lineage_mask = pd.Series(True, index=settled.index)
    for column in LINEAGE_COLUMNS:
        lineage_mask &= _present(settled[column])
    decision_time = pd.to_datetime(settled["decision_timestamp"], utc=True, errors="coerce")
    start_time = pd.to_datetime(settled["start_utc"], utc=True, errors="coerce")
    prestart_mask = decision_time.notna() & start_time.notna() & decision_time.lt(start_time)
    lineage_mask &= prestart_mask
    production_mask = settled["model_mode"].astype(str).str.lower().eq("production")
    auditable = settled[lineage_mask & production_mask].copy()
    configured_price_mask = auditable["price_universe"].astype(str).ne("all_quoted_books")
    deployable = auditable[configured_price_mask].copy()

    all_metrics = summarize_settled_bets(settled, starting_bankroll)
    auditable_metrics = summarize_settled_bets(auditable, starting_bankroll)
    auditable_lead = (
        pd.to_datetime(auditable["start_utc"], utc=True, errors="coerce")
        - pd.to_datetime(auditable["decision_timestamp"], utc=True, errors="coerce")
    ).dt.total_seconds() / 60.0
    auditable_metrics["mean_decision_lead_minutes"] = (
        float(auditable_lead.mean()) if auditable_lead.notna().any() else None
    )
    auditable_metrics["minimum_decision_lead_minutes"] = (
        float(auditable_lead.min()) if auditable_lead.notna().any() else None
    )
    deployable_metrics = summarize_settled_bets(deployable, starting_bankroll)
    ci = deployable_metrics.get("roi_ci_95_date_block")
    gates = {
        "minimum_auditable_bets": MIN_STRATEGY_VALIDATION_BETS,
        "minimum_auditable_betting_days": MIN_FORWARD_VALIDATION_BETTING_DAYS,
        "minimum_clv_coverage": MIN_FORWARD_VALIDATION_CLV_COVERAGE,
        "positive_date_block_roi_ci_lower_bound": True,
        "positive_mean_clv": True,
    }
    passed = {
        "minimum_auditable_bets": deployable_metrics["n_bets"] >= MIN_STRATEGY_VALIDATION_BETS,
        "minimum_auditable_betting_days": deployable_metrics["n_betting_days"] >= MIN_FORWARD_VALIDATION_BETTING_DAYS,
        "minimum_clv_coverage": deployable_metrics["clv_coverage"] >= MIN_FORWARD_VALIDATION_CLV_COVERAGE,
        "positive_date_block_roi_ci_lower_bound": ci is not None and ci[0] > 0,
        "positive_mean_clv": deployable_metrics["mean_clv"] is not None and deployable_metrics["mean_clv"] > 0,
    }
    execution_modes = sorted(auditable["execution_mode"].dropna().astype(str).str.lower().unique().tolist())
    return {
        "report_type": "forward_settled_ledger",
        "generated_at_utc": utc_now_iso(),
        "interpretation": (
            "Forward paper/live decisions recorded before first pitch. This is the evidence source for "
            "deployment validation; retrospective closing-line replays are reported separately."
        ),
        "execution_modes": execution_modes,
        "all_settled_history": all_metrics,
        "auditable_production_forward": auditable_metrics,
        "deployable_configured_book_forward": deployable_metrics,
        "lineage_complete_settled_bets": int(lineage_mask.sum()),
        "lineage_coverage": float(lineage_mask.mean()) if len(settled) else 0.0,
        "settled_bets_not_provably_prestart": int((~prestart_mask).sum()),
        "validation_thresholds": gates,
        "validation_gates_passed": passed,
        "strategy_validated_forward": bool(all(passed.values())),
    }


def main():
    if not BET_LOG.exists():
        logger.warning(f"{BET_LOG} does not exist; writing an empty forward report.")
        ledger = pd.DataFrame()
    else:
        ledger = pd.read_csv(BET_LOG, dtype={"game_pk": str})
    starting = 10000.0
    if STATE_PATH.exists():
        with open(STATE_PATH, encoding="utf-8") as handle:
            starting = float(json.load(handle).get("starting_bankroll", starting))
    report = build_report(ledger, starting)
    atomic_write_json(report, REPORT_PATH)
    metrics = report["deployable_configured_book_forward"]
    logger.info(
        f"Forward evidence: {metrics['n_bets']} deployable configured-book production bets across "
        f"{metrics['n_betting_days']} days; ROI={metrics['roi']}; "
        f"validated={report['strategy_validated_forward']} -> {REPORT_PATH}"
    )


if __name__ == "__main__":
    main()
