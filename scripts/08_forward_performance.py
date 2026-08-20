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
    "odds_event_id", "bookmaker_key", "decision_timestamp", "execution_mode",
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
    production_mask = settled["model_mode"].astype(str).str.lower().eq("production")
    auditable = settled[lineage_mask & production_mask].copy()

    all_metrics = summarize_settled_bets(settled, starting_bankroll)
    auditable_metrics = summarize_settled_bets(auditable, starting_bankroll)
    ci = auditable_metrics.get("roi_ci_95_date_block")
    gates = {
        "minimum_auditable_bets": MIN_STRATEGY_VALIDATION_BETS,
        "minimum_auditable_betting_days": MIN_FORWARD_VALIDATION_BETTING_DAYS,
        "minimum_clv_coverage": MIN_FORWARD_VALIDATION_CLV_COVERAGE,
        "positive_date_block_roi_ci_lower_bound": True,
        "positive_mean_clv": True,
    }
    passed = {
        "minimum_auditable_bets": auditable_metrics["n_bets"] >= MIN_STRATEGY_VALIDATION_BETS,
        "minimum_auditable_betting_days": auditable_metrics["n_betting_days"] >= MIN_FORWARD_VALIDATION_BETTING_DAYS,
        "minimum_clv_coverage": auditable_metrics["clv_coverage"] >= MIN_FORWARD_VALIDATION_CLV_COVERAGE,
        "positive_date_block_roi_ci_lower_bound": ci is not None and ci[0] > 0,
        "positive_mean_clv": auditable_metrics["mean_clv"] is not None and auditable_metrics["mean_clv"] > 0,
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
        "lineage_complete_settled_bets": int(lineage_mask.sum()),
        "lineage_coverage": float(lineage_mask.mean()) if len(settled) else 0.0,
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
    metrics = report["auditable_production_forward"]
    logger.info(
        f"Forward evidence: {metrics['n_bets']} auditable production bets across "
        f"{metrics['n_betting_days']} days; ROI={metrics['roi']}; "
        f"validated={report['strategy_validated_forward']} -> {REPORT_PATH}"
    )


if __name__ == "__main__":
    main()
