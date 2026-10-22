"""Loan product per risk bucket and the resulting portfolio view."""

from __future__ import annotations

import pandas as pd

from . import config


def portfolio(scored: pd.DataFrame) -> pd.DataFrame:
    """Lending amount, expected loss and expected return per eligible bucket.

    ``scored`` needs the columns ``bucket``, ``failure_probability`` and
    ``funding_total_usd``. Buckets without a loan product (E, F) are excluded.
    """
    df = scored[scored["bucket"].isin(config.LOAN_PRODUCTS)].copy()
    terms = pd.DataFrame(config.LOAN_PRODUCTS).T
    df["loan_amount"] = df["funding_total_usd"] * df["bucket"].map(terms["max_amount"]).astype(float)
    df["expected_loss"] = df["failure_probability"] * df["loan_amount"]
    df["expected_premium"] = df["bucket"].map(terms["risk_premium"]).astype(float) / 100 * df["loan_amount"]

    out = df.groupby("bucket").agg(
        startups=("loan_amount", "size"),
        lending_bn_usd=("loan_amount", "sum"),
        expected_loss_bn_usd=("expected_loss", "sum"),
        expected_premium_bn_usd=("expected_premium", "sum"),
    )
    out["expected_loss_rate_pct"] = out["expected_loss_bn_usd"] / out["lending_bn_usd"] * 100
    out["risk_premium_pct"] = terms["risk_premium"].astype(float)
    out["term"] = terms["term"]
    out[["lending_bn_usd", "expected_loss_bn_usd", "expected_premium_bn_usd"]] /= 1e9
    return out.round(2)
