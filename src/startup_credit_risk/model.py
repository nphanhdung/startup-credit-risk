"""Train, calibrate and evaluate the start-up failure model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


@dataclass
class Splits:
    X_train: pd.DataFrame
    X_cal: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_cal: pd.Series
    y_test: pd.Series


def split_data(df: pd.DataFrame) -> Splits:
    """Stratified 60 / 20 / 20 split into train, calibration and test sets."""
    X = df[config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES]
    y = df[config.TARGET]
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE
    )
    cal_share = config.CALIBRATION_SIZE / (1 - config.TEST_SIZE)
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_rest, y_rest, test_size=cal_share, stratify=y_rest, random_state=config.RANDOM_STATE
    )
    return Splits(X_train, X_cal, X_test, y_train, y_cal, y_test)


def build_pipeline(class_weight: dict[int, float]) -> Pipeline:
    """Preprocessing + class-weighted random forest."""
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocess = ColumnTransformer(
        [("num", numeric, config.NUMERIC_FEATURES), ("cat", categorical, config.CATEGORICAL_FEATURES)]
    )
    forest = RandomForestClassifier(class_weight=class_weight, random_state=config.RANDOM_STATE, n_jobs=-1)
    return Pipeline([("preprocess", preprocess), ("classifier", forest)])


def failure_class_weight(y: pd.Series) -> dict[int, float]:
    """Weight failures by the ratio of surviving to failed companies (about 12:1)."""
    ratio = (y == 0).sum() / max((y == 1).sum(), 1)
    return {0: 1.0, 1: float(ratio)}


def train(splits: Splits) -> tuple[Pipeline, CalibratedClassifierCV]:
    """Fit the forest on the training set, then calibrate it on the calibration set.

    The calibration set is never used to fit the forest, so the calibrated
    probabilities are estimated on data the forest has not seen.
    """
    base = build_pipeline(failure_class_weight(splits.y_train))
    base.fit(splits.X_train, splits.y_train)
    calibrated = CalibratedClassifierCV(FrozenEstimator(base), method="isotonic")
    calibrated.fit(splits.X_cal, splits.y_cal)
    return base, calibrated


def evaluate(model, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """AUC, Gini and Brier score on a held-out set."""
    p = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, p)
    return {"auc": auc, "gini": 2 * auc - 1, "brier": brier_score_loss(y, p)}


def feature_importance(base: Pipeline) -> pd.Series:
    """Random-forest importances, labelled with the encoded feature names."""
    names = base.named_steps["preprocess"].get_feature_names_out()
    names = [n.split("__", 1)[1] for n in names]
    values = base.named_steps["classifier"].feature_importances_
    return pd.Series(values, index=names).sort_values(ascending=False)


def naive_brier(y: pd.Series) -> float:
    """Brier score of always predicting the average failure rate (benchmark)."""
    return brier_score_loss(y, np.full(len(y), y.mean()))
