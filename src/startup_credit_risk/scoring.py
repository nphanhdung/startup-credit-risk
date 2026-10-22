"""Turn failure probabilities into a 0-1000 credit score and risk buckets."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def credit_score(failure_probability: np.ndarray) -> np.ndarray:
    """Higher score = lower chance of failure."""
    return (1 - np.asarray(failure_probability)) * 1000


def assign_bucket(scores: np.ndarray) -> pd.Series:
    """Map scores to buckets A (safest) to F (riskiest)."""
    scores = pd.Series(scores)
    buckets = pd.Series("NR", index=scores.index, dtype="object")
    for name, (low, high) in config.BUCKETS.items():
        buckets[(scores >= low) & (scores < high)] = name
    return buckets


def bucket_summary(probability: np.ndarray, y: pd.Series) -> pd.DataFrame:
    """Per bucket: size, average score, predicted vs. observed failure rate (%)."""
    frame = pd.DataFrame(
        {
            "score": credit_score(probability),
            "predicted": np.asarray(probability),
            "observed": np.asarray(y),
        }
    )
    frame["bucket"] = assign_bucket(frame["score"].to_numpy())
    summary = frame.groupby("bucket").agg(
        startups=("score", "size"),
        avg_score=("score", "mean"),
        predicted_failure_pct=("predicted", "mean"),
        observed_failure_pct=("observed", "mean"),
    )
    summary[["predicted_failure_pct", "observed_failure_pct"]] *= 100
    return summary.round(2)
