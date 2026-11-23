"""Train and evaluate the failure-risk model with a time-based split.

Task: from the machine's state at time t, predict which component (if any) fails
within the next 24 h. Split by time with a 24 h gap before each boundary, because a
label looks 24 h ahead and would otherwise leak information across the split.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from copilot.ml.features import (
    CLASSES,
    COMPONENTS,
    ERRORS,
    FEATURES,
    HORIZON,
    add_labels,
    build_features,
)

TRAIN_END = pd.Timestamp("2015-08-01")
VAL_END = pd.Timestamp("2015-10-01")
FIRST_FULL = pd.Timestamp("2015-01-02")  # first 24 h windows are incomplete


@dataclass
class Splits:
    """Feature/label frames per split."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def load_raw(raw_dir: Path, max_machine: int | None = None) -> dict[str, pd.DataFrame]:
    """Read the five raw CSVs into tidy frames (columns machine_id, ts, ...)."""

    def read(name: str, rename: dict[str, str]) -> pd.DataFrame:
        df = pd.read_csv(raw_dir / name, parse_dates=["datetime"] if name != "machines.csv" else [])
        df = df.rename(columns={"datetime": "ts", "machineID": "machine_id", **rename})
        return df if max_machine is None else df[df["machine_id"] <= max_machine]

    tel = read("telemetry.csv", {})
    for s in ("volt", "rotate", "pressure", "vibration"):
        tel[s] = tel[s].astype(np.float32)
    return {
        "telemetry": tel,
        "errors": read("errors.csv", {"errorID": "error_id"}),
        "maint": read("maint.csv", {"comp": "component"}),
        "failures": read("failures.csv", {"failure": "component"}),
        "machines": read("machines.csv", {}),
    }


def make_dataset(
    raw: dict[str, pd.DataFrame],
    step_hours: int = 3,
    machines_per_chunk: int = 100,
    horizon: pd.Timedelta = HORIZON,
) -> pd.DataFrame:
    """Labelled feature table sampled every `step_hours` hours.

    Machines are processed in chunks so the hourly feature table never has to fit in
    memory for the whole fleet at once.
    """
    ids = np.sort(raw["machines"]["machine_id"].unique())
    parts = []
    for start in range(0, len(ids), machines_per_chunk):
        chunk = set(ids[start : start + machines_per_chunk])

        def sel(df: pd.DataFrame, chunk: set = chunk) -> pd.DataFrame:
            return df[df["machine_id"].isin(chunk)]

        feats = build_features(
            sel(raw["telemetry"]), sel(raw["errors"]), sel(raw["maint"]), sel(raw["machines"])
        )
        feats = feats[(feats["ts"] >= FIRST_FULL) & (feats["ts"].dt.hour % step_hours == 0)]
        parts.append(add_labels(feats, sel(raw["failures"]), horizon))
    return pd.concat(parts, ignore_index=True)


def split(data: pd.DataFrame, horizon: pd.Timedelta = HORIZON) -> Splits:
    """Time-based split with a gap of one label horizon before each boundary."""
    train = data[data["ts"] <= TRAIN_END - horizon]
    val = data[(data["ts"] >= TRAIN_END) & (data["ts"] <= VAL_END - horizon)]
    test = data[data["ts"] >= VAL_END]
    return Splits(train, val, test)


@dataclass
class Metrics:
    """Evaluation of a risk score on one split."""

    pr_auc_any: float
    roc_auc_any: float
    alert_rate: float
    precision_at_rate: float
    recall_at_rate: float
    failures_caught: float
    recall_early: float
    pr_auc_per_component: dict[str, float] = field(default_factory=dict)


def evaluate_scores(
    df: pd.DataFrame,
    risk: np.ndarray,
    threshold: float,
    proba: np.ndarray | None = None,
) -> Metrics:
    """Window-level and failure-level metrics for an 'any failure within the horizon' score.

    failures_caught: share of failures with at least one alert in the horizon before them.
    recall_early: recall over positive windows more than 24 h before their failure (the
    simulator's warning errors appear within 24 h; earlier warnings are the hard part).
    """
    y_any = (df["label"] != "none").to_numpy()
    alerts = risk >= threshold
    tp = (alerts & y_any).sum()
    per_comp = {}
    if proba is not None:
        for i, comp in enumerate(CLASSES[1:], start=1):
            y = (df["label"] == comp).to_numpy()
            per_comp[comp] = (
                float(average_precision_score(y, proba[:, i])) if y.any() else float("nan")
            )
    pos = df.loc[y_any, ["machine_id", "ts", "next_failure_ts"]].copy()
    pos["alert"] = alerts[y_any]
    per_failure = pos.groupby(["machine_id", "next_failure_ts"])["alert"].any()
    early = (pos["next_failure_ts"] - pos["ts"]) > HORIZON
    return Metrics(
        pr_auc_any=float(average_precision_score(y_any, risk)),
        roc_auc_any=float(roc_auc_score(y_any, risk)),
        alert_rate=float(alerts.mean()),
        precision_at_rate=float(tp / max(alerts.sum(), 1)),
        recall_at_rate=float(tp / max(y_any.sum(), 1)),
        failures_caught=float(per_failure.mean()) if len(per_failure) else float("nan"),
        recall_early=float(pos.loc[early, "alert"].mean()) if early.any() else float("nan"),
        pr_auc_per_component=per_comp,
    )


def threshold_for_rate(scores: np.ndarray, rate: float) -> float:
    """Smallest threshold t such that at most `rate` of windows have score >= t.

    Handles ties (e.g. a rule score that is 0 for most windows) by moving up to the
    next distinct value instead of flagging every tied window.
    """
    values = np.sort(np.asarray(scores, dtype=float))[::-1]
    k = max(int(rate * len(values)), 1)
    t = values[k - 1]
    if (values >= t).mean() > rate:
        higher = values[values > t]
        t = higher.min() if len(higher) else np.inf
    return float(t)


def rule_score(df: pd.DataFrame) -> np.ndarray:
    """Baseline: total number of errors in the last 24 h (what a simple alarm rule sees)."""
    return df[[f"{e}_count_24h" for e in ERRORS]].sum(axis=1).to_numpy(dtype=float)


def train_logreg(train: pd.DataFrame, seed: int = 0):  # noqa: ANN201
    """Baseline: multinomial logistic regression on standardised features."""
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed))
    model.fit(train[FEATURES], encode(train["label"]))
    return model


def train_lgbm(train: pd.DataFrame, val: pd.DataFrame, params: dict, seed: int = 0):  # noqa: ANN201
    """LightGBM multiclass model with early stopping on the validation split."""
    model = lgb.LGBMClassifier(objective="multiclass", random_state=seed, verbose=-1, **params)
    model.fit(
        train[FEATURES],
        encode(train["label"]),
        eval_X=val[FEATURES],
        eval_y=encode(val["label"]),
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    return model


def class_proba(model, df: pd.DataFrame) -> np.ndarray:  # noqa: ANN001
    """Probabilities in CLASSES order (none, comp1..comp4)."""
    proba = model.predict_proba(df[FEATURES])
    order = [list(model.classes_).index(i) for i in range(len(CLASSES))]
    return proba[:, order]


def encode(labels: pd.Series) -> np.ndarray:
    """Class names -> integer codes in CLASSES order (0 = none)."""
    return labels.map({c: i for i, c in enumerate(CLASSES)}).to_numpy(dtype=np.int64)


__all__ = [
    "COMPONENTS",
    "Metrics",
    "Splits",
    "class_proba",
    "evaluate_scores",
    "load_raw",
    "make_dataset",
    "rule_score",
    "split",
    "threshold_for_rate",
    "train_lgbm",
    "train_logreg",
]
