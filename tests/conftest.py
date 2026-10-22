"""Synthetic data shaped like the raw Crunchbase export, so tests run without the real file."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

CATEGORIES = ["Software", "Mobile|Apps", "Biotechnology", "E-Commerce", "Curated Web|Media", "Health Care", None]
COUNTRIES = ["USA"] * 6 + ["GBR", "CAN", "DEU", None]


def make_raw(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    founded_year = rng.integers(2000, 2015, n)
    first_year = np.minimum(founded_year + rng.integers(0, 3, n), 2015)
    last_year = np.minimum(first_year + rng.integers(0, 4, n), 2015)
    month = rng.integers(1, 13, n)
    rounds = rng.integers(1, 6, n)
    funding = rng.lognormal(14, 1.5, n).round()

    # Failure is more likely for small, older, single-round companies.
    logit = -2.3 + 0.5 * (rounds == 1) - 0.25 * (np.log(funding) - 14) + 0.1 * (2015 - last_year)
    closed = rng.random(n) < 1 / (1 + np.exp(-logit))

    def date(year):
        return [f"{y}-{m:02d}-01" for y, m in zip(year, month)]

    founded = pd.Series(date(founded_year), dtype="object")
    founded[rng.random(n) < 0.2] = None  # missing founding dates, as in the real data
    funding_str = funding.astype(int).astype(str).astype(object)
    funding_str[rng.random(n) < 0.1] = "-"

    return pd.DataFrame(
        {
            "permalink": [f"/organization/company-{i}" for i in range(n)],
            "name": [f"Company {i}" for i in range(n)],
            "category_list": rng.choice(np.array(CATEGORIES, dtype=object), n),
            "funding_total_usd": funding_str,
            "status": np.where(closed, "closed", rng.choice(["operating", "acquired"], n, p=[0.9, 0.1])),
            "country_code": rng.choice(np.array(COUNTRIES, dtype=object), n),
            "state_code": rng.choice(["CA", "NY", "MA", None], n),
            "funding_rounds": rounds,
            "founded_at": founded,
            "first_funding_at": date(first_year),
            "last_funding_at": date(last_year),
        }
    )


@pytest.fixture
def raw_csv(tmp_path):
    path = tmp_path / "raw.csv"
    make_raw().to_csv(path, index=False)
    return path
