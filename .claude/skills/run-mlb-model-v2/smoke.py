# -*- coding: utf-8 -*-
"""Smoke driver for the MLB_MODEL_V2 pipeline. Run with the project venv:

    venv\\Scripts\\python.exe .claude\\skills\\run-mlb-model-v2\\smoke.py            # fast, offline
    venv\\Scripts\\python.exe .claude\\skills\\run-mlb-model-v2\\smoke.py --predict  # full E2E (network)
    venv\\Scripts\\python.exe .claude\\skills\\run-mlb-model-v2\\smoke.py --capture-once
                                                       # one closing-line snapshot (costs 1 Odds API credit)

Fast mode (default, no network, ~30s):
  1. environment: venv interpreter, .env keys present, raw data + trained models on disk
  2. import-loads every pipeline module (catches syntax/import errors in any script)
  3. scores the 2024 holdout through the calibrated models and checks AUC/Brier sanity
  4. runs scripts/03_backtest.py end-to-end (exit code + refreshed backtest json)

Exit code 0 = all checks passed. Any failure prints FAIL and exits 1.
"""
import sys
import json
import argparse
import subprocess
import importlib.util
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[3]
VENV_PY = ROOT / "venv" / "Scripts" / "python.exe"
SCRIPTS = ROOT / "scripts"

# joblib models pickle classes from scripts/model_classes.py — unpickling
# anywhere outside scripts/ fails unless scripts/ is importable first.
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))

FAILURES = []


def check(label, ok, detail=""):
    status = "ok  " if ok else "FAIL"
    print(f"[{status}] {label}" + (f" - {detail}" if detail else ""))
    if not ok:
        FAILURES.append(label)
    return ok


def check_environment():
    check("venv interpreter", Path(sys.executable).resolve() == VENV_PY.resolve(),
          f"running under {sys.executable}")
    env_text = (ROOT / ".env").read_text() if (ROOT / ".env").exists() else ""
    check(".env has ODDS_API_KEY", "ODDS_API_KEY=" in env_text)
    for rel in ("data/raw/game_logs_all.csv", "data/processed/feature_matrix.csv",
                "models/xgb_calibrated.joblib", "models/lgbm_calibrated.joblib",
                "models/train_medians.joblib", "models/feature_names.json"):
        check(rel, (ROOT / rel).exists())


def check_imports():
    # Script filenames start with digits, so they can't be `import`ed by
    # name — load through importlib, which is also how you'd call their
    # functions directly (see SKILL.md "Direct invocation").
    for name in ("model_classes", "odds_utils", "01_build_features", "02_train",
                 "03_backtest", "04_predict", "05_bankroll", "06_runner",
                 "07_capture_closing_lines", "wait_for_odds_and_predict"):
        path = SCRIPTS / f"{name}.py"
        try:
            spec = importlib.util.spec_from_file_location(f"smoke_{name}", path)
            spec.loader.exec_module(importlib.util.module_from_spec(spec))
            check(f"import {name}.py", True)
        except Exception as e:
            check(f"import {name}.py", False, repr(e))
    try:
        from features import builder  # noqa: F401
        check("import features/builder.py", True)
    except Exception as e:
        check("import features/builder.py", False, repr(e))


def check_model_scoring():
    import joblib
    import pandas as pd
    from sklearn.metrics import roc_auc_score, brier_score_loss

    with open(ROOT / "models" / "feature_names.json") as f:
        features = json.load(f)["passing_features"]
    medians = joblib.load(ROOT / "models" / "train_medians.joblib")
    xgb = joblib.load(ROOT / "models" / "xgb_calibrated.joblib")
    lgbm = joblib.load(ROOT / "models" / "lgbm_calibrated.joblib")

    df = pd.read_csv(ROOT / "data" / "processed" / "feature_matrix.csv")
    holdout = df[df["year"] == 2024]
    X = holdout[features].fillna(medians)
    y = holdout["home_win"].astype(int).values
    p = (xgb.predict_proba(X)[:, 1] + lgbm.predict_proba(X)[:, 1]) / 2.0

    check("holdout rows scored", len(p) > 2000, f"{len(p)} rows")
    check("probabilities in (0,1)", bool((p > 0).all() and (p < 1).all()))
    auc = roc_auc_score(y, p)
    brier = brier_score_loss(y, p)
    check("2024 AUC > 0.55", auc > 0.55, f"AUC={auc:.4f}")
    check("2024 Brier < 0.25", brier < 0.25, f"Brier={brier:.4f}")


def run_script(rel, args=(), timeout=600):
    cmd = [str(VENV_PY), rel, *args]
    print(f"--> {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=timeout)
    tail = "\n".join((res.stdout + res.stderr).strip().splitlines()[-3:])
    check(f"{rel} exit 0", res.returncode == 0, tail if res.returncode != 0 else "")
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict", action="store_true",
                        help="also run scripts/04_predict.py end-to-end (network; overwrites today's predictions file)")
    parser.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--capture-once", action="store_true",
                        help="also run 07_capture_closing_lines.py --once (spends 1 Odds API credit)")
    args = parser.parse_args()

    check_environment()
    check_imports()
    check_model_scoring()
    run_script("scripts/03_backtest.py")

    if args.predict:
        run_script(f"scripts/04_predict.py", ["--date", args.date], timeout=900)
        pred = ROOT / "outputs" / f"predictions_{args.date}.csv"
        ok = pred.exists()
        if ok:
            import pandas as pd
            n = len(pd.read_csv(pred))
            check(f"predictions_{args.date}.csv written", n > 0, f"{n} games")
        else:
            check(f"predictions_{args.date}.csv written", False)

    if args.capture_once:
        run_script("scripts/07_capture_closing_lines.py", ["--once", "--date", args.date])

    print()
    if FAILURES:
        print(f"SMOKE FAILED — {len(FAILURES)} check(s): {FAILURES}")
        sys.exit(1)
    print("SMOKE PASSED")


if __name__ == "__main__":
    main()
