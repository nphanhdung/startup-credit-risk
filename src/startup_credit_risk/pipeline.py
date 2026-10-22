"""Run the whole analysis from the command line.

    python -m startup_credit_risk.pipeline --data data/startup_failures.csv --out outputs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from . import model, plots, products, scoring
from .preprocessing import build_dataset


def run(data_path: str | Path, out_dir: str | Path = "outputs") -> dict:
    out = Path(out_dir)
    fig_dir = out / "figures"
    out.mkdir(parents=True, exist_ok=True)

    df = build_dataset(data_path)
    splits = model.split_data(df)
    base, calibrated = model.train(splits)

    # --- Evaluation on the untouched test set -------------------------------
    p_base = base.predict_proba(splits.X_test)[:, 1]
    p_cal = calibrated.predict_proba(splits.X_test)[:, 1]
    metrics = {
        "rows_after_scoping": len(df),
        "failure_rate": float(df["BAD"].mean()),
        "uncalibrated": model.evaluate(base, splits.X_test, splits.y_test),
        "calibrated": model.evaluate(calibrated, splits.X_test, splits.y_test),
        "naive_brier": model.naive_brier(splits.y_test),
    }
    buckets_test = scoring.bucket_summary(p_cal, splits.y_test)

    # --- Score every company for the portfolio view -------------------------
    scored = df.copy()
    scored["failure_probability"] = calibrated.predict_proba(scored[splits.X_test.columns])[:, 1]
    scored["score"] = scoring.credit_score(scored["failure_probability"])
    scored["bucket"] = scoring.assign_bucket(scored["score"].to_numpy()).to_numpy()
    portfolio = products.portfolio(scored)

    importances = model.feature_importance(base)

    # --- Save results -------------------------------------------------------
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    buckets_test.to_csv(out / "buckets_test_set.csv")
    portfolio.to_csv(out / "portfolio.csv")
    importances.to_csv(out / "feature_importance.csv", header=["importance"])

    plots.outcome_boxplots(df, fig_dir / "outcome_boxplots.png")
    plots.marci_chart(df, fig_dir / "marci_chart.png")
    plots.calibration(splits.y_test, {"Uncalibrated": p_base, "Isotonic calibration": p_cal}, fig_dir / "calibration.png")
    plots.roc(splits.y_test, p_cal, fig_dir / "roc.png")
    plots.importance(importances, fig_dir / "feature_importance.png")
    plots.score_distribution(scored["score"].to_numpy(), fig_dir / "score_distribution.png")

    return {"metrics": metrics, "buckets_test": buckets_test, "portfolio": portfolio, "importance": importances}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/startup_failures.csv", help="path to the raw CSV")
    parser.add_argument("--out", default="outputs", help="folder for results and figures")
    args = parser.parse_args()
    results = run(args.data, args.out)

    pd.set_option("display.width", 120)
    print(json.dumps(results["metrics"], indent=2))
    print("\nRisk buckets (test set):\n", results["buckets_test"])
    print("\nPortfolio (eligible buckets):\n", results["portfolio"])


if __name__ == "__main__":
    main()
