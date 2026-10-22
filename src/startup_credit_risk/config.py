"""Project-wide settings. Change values here rather than inside the code."""

from __future__ import annotations

RANDOM_STATE = 42

# --- Scope ---------------------------------------------------------------
MIN_FOUNDED_YEAR = 2005  # keep start-ups founded in or after this year
MIN_COUNTRY_SHARE = 0.01  # keep countries with more than 1% of companies

# Reference date for "time until now" features. None = the latest funding date
# in the data. Using a fixed date (not today's date) makes results reproducible.
REFERENCE_DATE: str | None = None

# --- Data split (train / calibration / test) ------------------------------
TEST_SIZE = 0.20
CALIBRATION_SIZE = 0.20  # share of the full data

# --- Features ------------------------------------------------------------
CATEGORICAL_FEATURES = ["country_code", "state_code", "industry_group"]
NUMERIC_FEATURES = [
    "funding_total_usd",
    "funding_rounds",
    "months_in_business",
    "delta_t_first_last",
    "delta_t_founded_first",
    "delta_t_last_now",
    "num_categories",
    "founded_year",
    "first_funding_year",
    "last_funding_year",
]
TARGET = "BAD"  # 1 = company closed, 0 = otherwise

# --- Rating scale ----------------------------------------------------------
# Score = (1 - probability of failure) * 1000. Bucket ranges follow the six
# credit quality steps of Commission Implementing Regulation (EU) 2016/1799.
BUCKETS: dict[str, tuple[float, float]] = {
    "A": (950, 1000.0001),
    "B": (900, 950),
    "C": (850, 900),
    "D": (750, 850),
    "E": (700, 750),
    "F": (0, 700),
}

# --- Loan products -----------------------------------------------------------
# max_amount: share of the company's total funding that can be lent
# risk_premium: percentage points above the risk-free rate
# Buckets E and F are not eligible for a loan.
LOAN_PRODUCTS: dict[str, dict] = {
    "A": {"max_amount": 0.80, "term": "3-5 years", "risk_premium": 3},
    "B": {"max_amount": 0.60, "term": "3 years", "risk_premium": 8},
    "C": {"max_amount": 0.40, "term": "2 years", "risk_premium": 15},
    "D": {"max_amount": 0.25, "term": "18-24 months", "risk_premium": 20},
}
