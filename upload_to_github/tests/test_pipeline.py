import numpy as np
import pandas as pd

from startup_credit_risk import config, scoring
from startup_credit_risk.industries import assign_industry_group
from startup_credit_risk.pipeline import run
from startup_credit_risk.preprocessing import _month_index, build_dataset


def test_month_index_handles_old_dates():
    months = _month_index(pd.Series(["2015-03-01", "1201-11-23"]))
    assert list(months) == [2015 * 12 + 3, 1201 * 12 + 11]


def test_missing_categories_are_other():
    groups = assign_industry_group(pd.Series([None, "Software", "Something unknown"]))
    assert list(groups) == ["Other", "Software", "Other"]


def test_scope_filters(raw_csv):
    df = build_dataset(raw_csv)
    assert df["founded_year"].min() >= config.MIN_FOUNDED_YEAR
    assert df["country_code"].notna().all()
    assert set(df.columns) == set(config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES + [config.TARGET])


def test_buckets_cover_every_score():
    scores = np.array([0, 699.9, 700, 849.9, 950, 1000])
    assert list(scoring.assign_bucket(scores)) == ["F", "F", "E", "D", "A", "A"]


def test_pipeline_runs_end_to_end(raw_csv, tmp_path):
    results = run(raw_csv, tmp_path / "out", tmp_path / "figures")
    metrics = results["metrics"]
    assert 0.5 < metrics["calibrated"]["auc"] <= 1
    for name in ["marci_chart", "portfolio_pies", "industry_failure_rates", "calibration"]:
        assert (tmp_path / "figures" / f"{name}.png").exists()
    assert set(results["portfolio"].index) <= set(config.LOAN_PRODUCTS)
