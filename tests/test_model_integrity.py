# -*- coding: utf-8 -*-
import importlib.util
import sys
import unittest
import tempfile
from unittest import mock
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from features import builder, props_builder  # noqa: E402
from scripts.betting_strategy import size_stake  # noqa: E402
from scripts.decision_log import append_decisions  # noqa: E402
from scripts.odds_utils import aggregate_h2h_event  # noqa: E402
from scripts.performance_metrics import date_block_roi_ci  # noqa: E402


def load_numbered_script(name):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"test_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TemporalIntegrityTests(unittest.TestCase):
    def test_production_cv_validates_each_season_on_prior_seasons_only(self):
        training = load_numbered_script("02_train")
        years = pd.Series([2021, 2021, 2022, 2022, 2023, 2023])
        splitter = training.ExpandingYearSplit(years)
        folds = list(splitter.split(pd.DataFrame({"x": range(len(years))})))
        self.assertEqual(len(folds), 2)
        for train_index, validation_index in folds:
            self.assertLess(years.iloc[train_index].max(), years.iloc[validation_index].min())

    def test_fold_imputation_handles_features_absent_in_early_seasons(self):
        training = load_numbered_script("02_train")
        features = pd.DataFrame({"known": [1.0, None, 3.0], "future_source": [None, None, 7.0]})
        train, validation = training.fold_matrices(features, [0, 1], [2])
        self.assertFalse(train.isna().any().any())
        self.assertFalse(validation.isna().any().any())
        self.assertEqual(validation.loc[2, "future_source"], 7.0)

    def test_appending_future_games_does_not_mutate_prior_rolling_features(self):
        rows = []
        for game_pk, day, home, away, hs, aws in [
            ("1", "2026-04-01", "NYY", "BOS", 4, 2),
            ("2", "2026-04-02", "BOS", "NYY", 3, 5),
            ("3", "2026-04-03", "NYY", "BOS", 1, 9),
        ]:
            rows.extend([
                {"game_pk": game_pk, "date": day, "team": home, "opponent": away, "is_home": 1,
                 "runs_scored": hs, "runs_allowed": aws, "win": int(hs > aws), "year": 2026},
                {"game_pk": game_pk, "date": day, "team": away, "opponent": home, "is_home": 0,
                 "runs_scored": aws, "runs_allowed": hs, "win": int(aws > hs), "year": 2026},
            ])
        base = builder.build_rolling_features(pd.DataFrame(rows[:4]))
        extended = builder.build_rolling_features(pd.DataFrame(rows))
        cols = ["game_pk", "team", "runs_scored_roll_10", "win_rate_roll_5", "pyth_win_pct"]
        pd.testing.assert_frame_equal(
            base[cols].reset_index(drop=True),
            extended[extended["game_pk"].isin(["1", "2"])][cols].reset_index(drop=True),
        )

    def test_future_umpire_result_cannot_change_prior_game_factor(self):
        dates = pd.date_range("2026-04-01", periods=12, freq="D")
        first_twelve = pd.DataFrame({
            "game_pk": [str(i) for i in range(1, 13)], "date": dates,
            "umpire_name": ["Ump"] * 12, "total_runs": list(range(4, 16)),
        })
        assignments = first_twelve[["game_pk", "date", "umpire_name"]].copy()
        games = assignments.loc[assignments["game_pk"] == "12", ["game_pk", "date"]]
        with_future = pd.concat([first_twelve, pd.DataFrame({
            "game_pk": ["13"], "date": ["2026-04-13"], "umpire_name": ["Ump"], "total_runs": [100],
        })], ignore_index=True)
        before = builder.join_umpires(games, assignments, first_twelve)
        after = builder.join_umpires(games, assignments, with_future)
        self.assertTrue(before["umpire_run_factor"].notna().all())
        pd.testing.assert_series_equal(before["umpire_run_factor"], after["umpire_run_factor"])

    def test_pitcher_rolling_uses_latest_completed_start(self):
        logs = pd.DataFrame({
            "pitcher_name": ["Home Ace"] * 3 + ["Away Ace"] * 3,
            "date": ["2026-04-01", "2026-04-05", "2026-04-10"] * 2,
            "era": [1.0, 2.0, 9.0, 4.0, 5.0, 6.0],
            "fip": [2.0, 3.0, 10.0, 5.0, 6.0, 7.0],
        })
        games = pd.DataFrame({
            "date": ["2026-04-10", "2026-04-12"],
            "home_starter": ["Home Ace", "Home Ace"],
            "away_starter": ["Away Ace", "Away Ace"],
        })
        out = builder.join_pitcher_rolling_form(games, logs)
        self.assertAlmostEqual(out.loc[0, "home_sp_era_roll3"], 1.5)
        self.assertAlmostEqual(out.loc[1, "home_sp_era_roll3"], 4.0)

    def test_props_pitcher_rolling_uses_latest_completed_start(self):
        logs = pd.DataFrame({
            "pitcher_name": ["Ace"] * 3,
            "date": ["2026-04-01", "2026-04-05", "2026-04-10"],
            "era": [1.0, 2.0, 9.0], "fip": [2.0, 3.0, 10.0],
        })
        games = pd.DataFrame({"date": ["2026-04-12"], "opp_starter_name": ["Ace"]})
        out = props_builder.join_opposing_pitcher_rolling(games, logs)
        self.assertAlmostEqual(out.loc[0, "opp_sp_era_roll3"], 4.0)

    def test_park_factor_is_lagged_for_moneyline_and_props(self):
        factors = pd.DataFrame({"team": ["NYY"], "year": [2025], "park_factor": [1.08]})
        money = pd.DataFrame({"team": ["NYY"], "opponent": ["BOS"], "is_home": [1], "year": [2026]})
        props = pd.DataFrame({"team": ["NYY"], "opp": ["BOS"], "is_home": [1], "year": [2026]})
        self.assertAlmostEqual(builder.join_park_factors(money, factors).loc[0, "park_factor"], 1.08)
        self.assertAlmostEqual(props_builder.join_park_factors(props, factors).loc[0, "park_factor"], 1.08)

    def test_switch_hitters_count_as_platoon_advantage(self):
        games = pd.DataFrame({
            "game_pk": ["1", "1"], "is_home": [1, 0],
            "home_starter": ["Left Arm", "Left Arm"],
            "away_starter": ["Right Arm", "Right Arm"],
        })
        bio = pd.DataFrame({
            "mlbam_id": [1, 2, 3], "full_name": ["Left Arm", "Right Arm", "Switch Bat"],
            "throw_hand": ["L", "R", "R"], "bat_side": ["R", "L", "S"],
        })
        lineups = pd.DataFrame({"game_pk": ["1", "1"], "is_home": [1, 1], "batter_mlbam_id": [2, 3]})
        out = builder.join_player_bio(games, bio, lineups)
        self.assertAlmostEqual(out.loc[out["is_home"] == 1, "platoon_advantage"].iloc[0], 1.0)


class RiskAndMatchingTests(unittest.TestCase):
    def test_stake_caps_are_enforced(self):
        single = size_stake(0.80, 150, 10000)
        self.assertLessEqual(single.stake, 200.0)
        daily = size_stake(0.80, 150, 10000, daily_staked=790.0)
        self.assertEqual(daily.stake, 10.0)
        open_cap = size_stake(0.80, 150, 10000, open_staked=1500.0)
        self.assertEqual(open_cap.stake, 0.0)
        reserved = size_stake(0.60, 120, 10000, open_staked=1000.0)
        self.assertEqual(reserved.available_bankroll, 9000.0)

    def test_bankroll_state_reconciles_from_settled_ledger(self):
        bankroll = load_numbered_script("05_bankroll")
        ledger = pd.DataFrame({"status": ["settled", "settled", "pending"], "pnl": [100.0, -40.0, None]})
        state = bankroll.reconcile_state_from_ledger(
            {"starting_bankroll": 10000.0, "bankroll": 1.0, "peak_bankroll": 1.0}, ledger,
        )
        self.assertEqual(state["bankroll"], 10060.0)
        self.assertEqual(state["peak_bankroll"], 10100.0)

    def test_doubleheader_events_are_one_to_one(self):
        capture = load_numbered_script("07_capture_closing_lines")
        slate = pd.DataFrame({
            "game_pk": ["100", "101"],
            "home_team_name": ["New York Yankees"] * 2,
            "away_team_name": ["Boston Red Sox"] * 2,
            "start_utc": [
                datetime(2026, 8, 19, 17, tzinfo=timezone.utc),
                datetime(2026, 8, 19, 23, tzinfo=timezone.utc),
            ],
        })
        def event(event_id, start):
            books = []
            for i in range(3):
                books.append({"key": f"book{i}", "markets": [{"key": "h2h", "outcomes": [
                    {"name": "New York Yankees", "price": -120},
                    {"name": "Boston Red Sox", "price": 110},
                ]}]})
            return {"id": event_id, "commence_time": start, "home_team": "New York Yankees",
                    "away_team": "Boston Red Sox", "bookmakers": books}
        matched = capture.match_events_to_slate([
            event("e1", "2026-08-19T17:00:00Z"), event("e2", "2026-08-19T23:00:00Z")
        ], slate, "2026-08-19")
        self.assertEqual(set(matched), {"100", "101"})
        self.assertEqual({value["odds_event_id"] for value in matched.values()}, {"e1", "e2"})

    def test_consensus_uses_best_executable_price_for_each_side(self):
        event = {
            "home_team": "Home", "away_team": "Away",
            "bookmakers": [
                {"key": "a", "title": "A", "markets": [{"key": "h2h", "outcomes": [
                    {"name": "Home", "price": -120}, {"name": "Away", "price": 110},
                ]}]},
                {"key": "b", "title": "B", "markets": [{"key": "h2h", "outcomes": [
                    {"name": "Home", "price": -115}, {"name": "Away", "price": 105},
                ]}]},
                {"key": "c", "title": "C", "markets": [{"key": "h2h", "outcomes": [
                    {"name": "Home", "price": -125}, {"name": "Away", "price": 115},
                ]}]},
            ],
        }
        result = aggregate_h2h_event(event)
        self.assertEqual(result["home_ml"], -115)
        self.assertEqual(result["home_bookmaker_key"], "b")
        self.assertEqual(result["away_ml"], 115)
        self.assertEqual(result["away_bookmaker_key"], "c")
        self.assertGreater(result["book_prob_range"], 0)

        restricted = aggregate_h2h_event(event, allowed_bookmakers=["a"])
        self.assertEqual(restricted["home_bookmaker_key"], "a")
        self.assertEqual(restricted["away_bookmaker_key"], "a")
        self.assertEqual(restricted["n_books"], 3)
        self.assertEqual(restricted["n_executable_books"], 1)
        self.assertEqual(restricted["price_universe"], "a")

    def test_lineup_poll_retries_then_forces_a_cutoff_decision(self):
        capture = load_numbered_script("07_capture_closing_lines")
        now = datetime(2026, 8, 20, 18, tzinfo=timezone.utc)
        start = now + timedelta(hours=2)
        run_now, retry, next_wake = capture.plan_lineup_decision(now, start, ["1", "2"], {"1"})
        self.assertEqual(run_now, ["1"])
        self.assertEqual(retry, ["2"])
        self.assertEqual(next_wake, now + capture.LINEUP_POLL_INTERVAL)
        at_cutoff = start - capture.PREGAME_DECISION_CUTOFF
        run_now, retry, next_wake = capture.plan_lineup_decision(at_cutoff, start, ["2"], set())
        self.assertEqual(run_now, ["2"])
        self.assertEqual(retry, [])
        self.assertIsNone(next_wake)

    def test_decision_log_is_append_only_and_idempotent_per_run(self):
        predictions = pd.DataFrame([{
            "run_id": "predict_1", "game_pk": "999", "date": "2026-08-20",
            "generated_at_utc": "2026-08-20T18:00:00+00:00",
            "decision_eligible": False, "bet_flag": False,
            "ineligibility_reason": "lineups_not_posted",
        }])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "decision_log.csv"
            self.assertEqual(append_decisions(predictions, path, "paper"), 1)
            self.assertEqual(append_decisions(predictions, path, "paper"), 0)
            logged = pd.read_csv(path, dtype={"game_pk": str})
        self.assertEqual(len(logged), 1)
        self.assertEqual(logged.loc[0, "decision_status"], "ineligible")
        self.assertEqual(logged.loc[0, "ineligibility_reason"], "lineups_not_posted")

    def test_roi_uncertainty_resamples_whole_betting_dates(self):
        records = pd.DataFrame({
            "date": ["2026-08-01", "2026-08-01", "2026-08-02", "2026-08-02"],
            "pnl": [100.0, -20.0, -50.0, -50.0],
            "stake": [100.0, 100.0, 100.0, 100.0],
        })
        interval = date_block_roi_ci(records, n_bootstrap=1000)
        self.assertEqual(len(interval), 2)
        self.assertLess(interval[0], interval[1])

    def test_forward_report_excludes_unattributed_legacy_bets(self):
        report_module = load_numbered_script("08_forward_performance")
        complete = {
            "date": "2026-08-20", "bet_size": 100.0, "pnl": 10.0, "result": "win", "status": "settled",
            "run_id": "predict_1", "model_version": "model_1", "model_mode": "production",
            "data_version": "data_1", "feature_build_id": "features_1", "odds_event_id": "event_1",
            "bookmaker_key": "book_1", "decision_timestamp": "2026-08-20T18:00:00Z",
            "start_utc": "2026-08-20T20:00:00Z",
            "price_universe": "book_1",
            "execution_mode": "paper", "clv": 0.01,
        }
        legacy = {"date": "2026-08-19", "bet_size": 100.0, "pnl": 5.0, "result": "win", "status": "settled"}
        report = report_module.build_report(pd.DataFrame([complete, legacy]))
        self.assertEqual(report["all_settled_history"]["n_bets"], 2)
        self.assertEqual(report["auditable_production_forward"]["n_bets"], 1)
        self.assertEqual(report["deployable_configured_book_forward"]["n_bets"], 1)
        self.assertFalse(report["strategy_validated_forward"])

    def test_bankroll_log_preserves_lineage_and_caps_stake(self):
        bankroll = load_numbered_script("05_bankroll")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            outputs, models = root / "outputs", root / "models"
            outputs.mkdir()
            models.mkdir()
            date = "2026-08-20"
            pd.DataFrame([{
                "date": date, "game_pk": "999", "home_team": "NYY", "away_team": "BOS",
                "bet_flag": True, "decision_eligible": True, "bet_side": "HOME", "home_win_prob": 0.65,
                "home_ml": 120, "away_ml": -130, "no_vig_home_implied": 0.52,
                "edge": 0.13, "run_id": "predict_test",
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "model_version": "model_test", "model_mode": "production",
                "data_version": "data_test", "feature_build_id": "feature_test",
                "odds_event_id": "event_test", "bookmaker_key": "book_test",
                "bookmaker_title": "Book Test", "odds_snapshot_utc": datetime.now(timezone.utc).isoformat(),
                "start_utc": (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat(),
                "home_bookmaker_key": "best_home", "home_bookmaker_title": "Best Home",
                "price_selection_method": "best_available_across_quoted_books", "execution_mode": "paper",
                "lineup_available": True, "umpire_available": True,
            }]).to_csv(outputs / f"predictions_{date}.csv", index=False)
            with mock.patch.multiple(
                bankroll, OUTPUTS=outputs, MODELS=models,
                STATE_PATH=models / "bankroll_state.json", BET_LOG_PATH=outputs / "bet_log.csv",
            ):
                bankroll.cmd_log_bets(date, resume=False, force=False, game_pks=["999"])
                log = pd.read_csv(outputs / "bet_log.csv", dtype={"game_pk": str})
            self.assertEqual(log.loc[0, "model_version"], "model_test")
            self.assertEqual(log.loc[0, "odds_event_id"], "event_test")
            self.assertEqual(log.loc[0, "bookmaker_key"], "best_home")
            self.assertEqual(log.loc[0, "execution_mode"], "paper")
            self.assertGreater(log.loc[0, "decision_lead_minutes"], 0)
            self.assertLessEqual(log.loc[0, "bet_size"], 200.0)

    def test_bankroll_refuses_a_post_first_pitch_decision(self):
        bankroll = load_numbered_script("05_bankroll")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            outputs, models = root / "outputs", root / "models"
            outputs.mkdir()
            models.mkdir()
            date = "2026-08-20"
            pd.DataFrame([{
                "date": date, "game_pk": "999", "home_team": "NYY", "away_team": "BOS",
                "bet_flag": True, "decision_eligible": True, "bet_side": "HOME",
                "home_win_prob": 0.65, "home_ml": 120, "away_ml": -130,
                "run_id": "predict_late", "generated_at_utc": "2026-08-20T20:01:00Z",
                "start_utc": "2026-08-20T20:00:00Z",
            }]).to_csv(outputs / f"predictions_{date}.csv", index=False)
            with mock.patch.multiple(
                bankroll, OUTPUTS=outputs, MODELS=models,
                STATE_PATH=models / "bankroll_state.json", BET_LOG_PATH=outputs / "bet_log.csv",
            ):
                with self.assertRaises(SystemExit):
                    bankroll.cmd_log_bets(date, resume=False, force=False, game_pks=["999"])
            self.assertFalse((outputs / "bet_log.csv").exists())


if __name__ == "__main__":
    unittest.main()
