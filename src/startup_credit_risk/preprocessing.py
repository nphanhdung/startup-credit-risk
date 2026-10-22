"""Load the raw Crunchbase export, clean it and build model features."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config
from .industries import assign_industry_group

DATE_COLUMNS = ["founded_at", "first_funding_at", "last_funding_at"]

# Obvious typing errors in the raw dates (e.g. year 2105 instead of 2015).
DATE_TYPO_FIXES: dict[str, dict[str, str]] = {
    "founded_at": {
        "2105-02-17": "2015-02-17",
        "2041-09-21": "2014-09-21",
        "2104-01-01": "2014-01-01",
        "2914-01-01": "2014-01-01",
        "1015-01-30": "2015-01-30",
    },
    "first_funding_at": {
        "1201-11-23": "2014-12-03",
        "1532-06-26": "2013-06-26",
        "1014-03-01": "2014-03-01",
    },
    "last_funding_at": {
        "2105-05-01": "2015-05-01",
        "2115-12-01": "2015-12-01",
    },
}

# Company-specific corrections checked by hand against public sources.
COMPANY_DATE_FIXES: dict[str, dict[str, str]] = {
    "/organization/atipica": {"founded_at": "2015-01-30"},
    "/organization/gamewheel": {"first_funding_at": "2014-11-23"},
}


def load_raw(path: str | Path) -> pd.DataFrame:
    """Read the raw CSV file."""
    return pd.read_csv(path, dtype={"funding_total_usd": "string"})


def _month_index(dates: pd.Series) -> pd.Series:
    """Convert 'YYYY-MM-DD' strings to a month counter (year * 12 + month).

    Plain integer arithmetic avoids pandas' datetime range limits, which some
    raw dates would otherwise break.
    """
    parts = dates.astype(str).str.split("-", expand=True)
    return parts[0].astype(int) * 12 + parts[1].astype(int)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix dates, fill gaps and create the target variable."""
    df = df.copy()

    df[config.TARGET] = (df["status"] == "closed").astype(int)
    df["industry_group"] = assign_industry_group(df["category_list"])

    for permalink, fixes in COMPANY_DATE_FIXES.items():
        for column, value in fixes.items():
            df.loc[df["permalink"] == permalink, column] = value
    for column, fixes in DATE_TYPO_FIXES.items():
        df[column] = df[column].replace(fixes)

    # Founding date missing -> use the first (or last) funding date instead.
    df["founded_at"] = df["founded_at"].fillna(df["first_funding_at"]).fillna(df["last_funding_at"])
    df["first_funding_at"] = df["first_funding_at"].fillna(df["last_funding_at"])
    df = df.dropna(subset=DATE_COLUMNS)

    df["funding_rounds"] = pd.to_numeric(df["funding_rounds"], errors="coerce")
    df["funding_total_usd"] = pd.to_numeric(df["funding_total_usd"].replace("-", pd.NA), errors="coerce")
    return df


def add_features(df: pd.DataFrame, reference_date: str | None = config.REFERENCE_DATE) -> pd.DataFrame:
    """Create time-based and category features."""
    df = df.copy()
    founded = _month_index(df["founded_at"])
    first = _month_index(df["first_funding_at"])
    last = _month_index(df["last_funding_at"])
    now = _month_index(pd.Series([reference_date]))[0] if reference_date else last.max()

    df["months_in_business"] = now - founded
    df["delta_t_first_last"] = last - first
    df["delta_t_founded_first"] = first - founded
    df["delta_t_last_now"] = now - last

    df["founded_year"] = founded.sub(1) // 12
    df["first_funding_year"] = first.sub(1) // 12
    df["last_funding_year"] = last.sub(1) // 12

    df["num_categories"] = df["category_list"].fillna("").map(lambda s: len(s.split("|")) if s else 0)
    return df


def apply_scope(df: pd.DataFrame) -> pd.DataFrame:
    """Keep recent start-ups from countries that are well represented."""
    df = df[df["founded_year"] >= config.MIN_FOUNDED_YEAR]
    shares = df["country_code"].value_counts(normalize=True)
    countries = shares[shares > config.MIN_COUNTRY_SHARE].index
    return df[df["country_code"].isin(countries)].reset_index(drop=True)


def build_dataset(path: str | Path) -> pd.DataFrame:
    """Full preprocessing: raw CSV -> modelling dataset."""
    df = apply_scope(add_features(clean(load_raw(path))))
    columns = config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES + [config.TARGET]
    return df[columns]
