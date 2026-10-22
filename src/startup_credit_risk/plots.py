"""Figures for the analysis. Each function saves one PNG and returns its path."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import RocCurveDisplay  # noqa: E402

from . import config  # noqa: E402


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def calibration(y, probs: dict[str, np.ndarray], path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    top = 0.0
    for label, p in probs.items():
        frac, mean = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot(mean, frac, "s-", label=label)
        top = max(top, mean.max(), frac.max())
    ax.plot([0, top], [0, top], "k--", label="Perfectly calibrated")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed failure rate", title="Calibration (test set)")
    ax.legend()
    return _save(fig, path)


def roc(y, p, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay.from_predictions(y, p, ax=ax, name="Calibrated model")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_title("ROC curve (test set)")
    return _save(fig, path)


def importance(importances: pd.Series, path: Path, top_n: int = 10) -> Path:
    top = importances.head(top_n)[::-1]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(top.index, top.values)
    ax.set(xlabel="Importance", title=f"Top {top_n} features")
    return _save(fig, path)


def score_distribution(scores: np.ndarray, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(scores, bins=50)
    for name, (low, _) in config.BUCKETS.items():
        if low > 0:
            ax.axvline(low, ls="--", lw=1, color="grey")
            ax.text(low, ax.get_ylim()[1] * 0.95, f" {name}", va="top")
    ax.set(xlabel="Credit score (0-1000)", ylabel="Start-ups", title="Credit score distribution")
    return _save(fig, path)


def outcome_boxplots(df: pd.DataFrame, path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    groups = [0, 1]
    labels = ["Not closed", "Closed"]
    funding = [df.loc[df[config.TARGET] == g, "funding_total_usd"].dropna() for g in groups]
    lifespan = [df.loc[df[config.TARGET] == g, "months_in_business"] for g in groups]
    axes[0].boxplot(funding, tick_labels=labels)
    axes[0].set(yscale="log", ylabel="Total funding (USD, log)", title="Funding by outcome")
    axes[1].boxplot(lifespan, tick_labels=labels)
    axes[1].set(ylabel="Months in business", title="Business age by outcome")
    return _save(fig, path)


def marci_chart(df: pd.DataFrame, path: Path) -> Path:
    """Industry risk map: failure rate (x) vs. average funding (y).

    Bubble size = average business age, colour = average months between the
    first and last funding round. All four measures are min-max scaled.
    """
    agg = df.groupby("industry_group").agg(
        likelihood=(config.TARGET, "mean"),
        impact=("funding_total_usd", "mean"),
        lifespan=("months_in_business", "mean"),
        funding_span=("delta_t_first_last", "mean"),
    )
    scaled = (agg - agg.min()) / (agg.max() - agg.min()).replace(0, 1)
    fig, ax = plt.subplots(figsize=(11, 7))
    points = ax.scatter(
        scaled["likelihood"],
        scaled["impact"],
        s=100 + 700 * scaled["lifespan"],
        c=scaled["funding_span"],
        cmap="coolwarm",
        alpha=0.8,
        edgecolor="white",
    )
    for name, row in scaled.iterrows():
        ax.annotate(name, (row["likelihood"], row["impact"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    fig.colorbar(points, ax=ax, label="Time between first and last funding (scaled)")
    ax.set(xlabel="Likelihood of failure (scaled)", ylabel="Impact: average funding (scaled)", title="Industry risk map (MARCI)")
    ax.margins(x=0.15, y=0.08)
    ax.grid(alpha=0.3)
    return _save(fig, path)
