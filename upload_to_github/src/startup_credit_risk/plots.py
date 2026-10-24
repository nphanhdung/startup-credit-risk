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


FEATURE_LABELS = {
    "delta_t_last_now": "Months since last funding",
    "months_in_business": "Business age (months)",
    "funding_total_usd": "Total funding (USD)",
    "delta_t_founded_first": "Months from founding to first funding",
    "delta_t_first_last": "Months between first and last funding",
    "last_funding_year": "Year of last funding",
    "first_funding_year": "Year of first funding",
    "founded_year": "Founding year",
    "num_categories": "Number of categories",
    "funding_rounds": "Number of funding rounds",
}


def importance(importances: pd.Series, path: Path, top_n: int = 10) -> Path:
    top = importances.head(top_n)[::-1]
    top.index = [FEATURE_LABELS.get(name, name) for name in top.index]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(top.index, top.values, color="#4C72B0")
    ax.set(xlabel="Importance", title=f"Top {top_n} features")
    return _save(fig, path)


def score_distribution(scores: np.ndarray, path: Path) -> Path:
    scores = np.asarray(scores)
    low_edge = max(0, np.floor(np.percentile(scores, 0.5) / 50) * 50)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(scores, bins=np.arange(low_edge, 1001, 10), color="#4C72B0")
    top = ax.get_ylim()[1]
    for name, (low, high) in config.BUCKETS.items():
        left, right = max(low, low_edge), min(high, 1000)
        if low > low_edge:
            ax.axvline(low, ls="--", lw=1, color="grey")
        if right > left:
            ax.text((left + right) / 2, top * 0.97, name, ha="center", va="top", fontweight="bold")
    ax.set(xlim=(low_edge, 1000), xlabel="Credit score (0-1000)", ylabel="Start-ups", title="Credit score distribution and risk buckets")
    return _save(fig, path)


def outcome_boxplots(df: pd.DataFrame, path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    groups = [0, 1]
    labels = ["Not closed", "Closed"]
    colors = ["#a6bddb", "#f4a582"]
    funding = [df.loc[df[config.TARGET] == g, "funding_total_usd"].dropna() for g in groups]
    lifespan = [df.loc[df[config.TARGET] == g, "months_in_business"] for g in groups]
    for ax, data, ylabel, title in [
        (axes[0], funding, "Total funding (USD, log scale)", "Funding by outcome"),
        (axes[1], lifespan, "Months in business", "Business age by outcome"),
    ]:
        box = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False, medianprops={"color": "black"})
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
        ax.set(ylabel=ylabel, title=title)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_yscale("log")
    return _save(fig, path)


def industry_failure_rates(df: pd.DataFrame, path: Path, top_n: int = 10, min_companies: int = 30) -> Path:
    """Industries with the highest share of closed start-ups."""
    rates = df.groupby("industry_group")[config.TARGET].agg(["mean", "size"])
    rates = rates[rates["size"] >= min_companies].sort_values("mean").tail(top_n)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(rates.index, rates["mean"] * 100, color="#d6604d")
    ax.axvline(df[config.TARGET].mean() * 100, ls="--", color="grey", lw=1, label="Average")
    ax.set(xlabel="Failure rate (%)", title=f"Top {top_n} industries by failure rate (min. {min_companies} start-ups)")
    ax.legend()
    return _save(fig, path)


def marci_chart(df: pd.DataFrame, path: Path, min_companies: int = 100) -> Path:
    """Industry risk map: failure rate (x) vs. average funding (y).

    Bubble size = average business age, colour = average months between the
    first and last funding round. All four measures are min-max scaled.
    Industries with fewer than ``min_companies`` start-ups are left out,
    because their averages are too noisy to compare.
    """
    agg = df.groupby("industry_group").agg(
        likelihood=(config.TARGET, "mean"),
        impact=("funding_total_usd", "mean"),
        lifespan=("months_in_business", "mean"),
        funding_span=("delta_t_first_last", "mean"),
        size=(config.TARGET, "size"),
    )
    agg = agg[agg.pop("size") >= min_companies]
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
        ax.annotate(name, (row["likelihood"], row["impact"]), fontsize=8, xytext=(5, 5), textcoords="offset points")
    fig.colorbar(points, ax=ax, label="Time between first and last funding (scaled)")
    ax.set(xlabel="Likelihood of failure (scaled)", ylabel="Impact: average funding (scaled)", title="Industry risk map (MARCI)")
    ax.margins(x=0.15, y=0.08)
    ax.grid(alpha=0.3)
    return _save(fig, path)


def portfolio_pies(portfolio: pd.DataFrame, path: Path) -> Path:
    """Share of lending and of expected loss per bucket.

    Percentages are written in the legend, so small slices stay readable.
    """
    colors = ["#67001f", "#b2182b", "#d6604d", "#f4a582"][: len(portfolio)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, column, title in [
        (axes[0], "lending_bn_usd", "Lending by bucket"),
        (axes[1], "expected_loss_bn_usd", "Expected loss by bucket"),
    ]:
        values = portfolio[column]
        shares = values / values.sum() * 100
        ax.pie(values, colors=colors, startangle=90, counterclock=False, wedgeprops={"edgecolor": "white"})
        ax.legend(
            [f"{bucket}: {share:.1f}%" for bucket, share in shares.items()],
            loc="center left",
            bbox_to_anchor=(0.92, 0.5),
            frameon=False,
        )
        ax.set_title(title)
    return _save(fig, path)
