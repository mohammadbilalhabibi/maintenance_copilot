"""Feature engineering for the failure-risk model.

The same functions build features for training (all machines, whole year) and for
serving (one machine at one point in time), so both paths see identical features.
All features at time t use only data with timestamp <= t.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SIGNALS = ["volt", "rotate", "pressure", "vibration"]
ERRORS = ["error1", "error2", "error3", "error4", "error5"]
COMPONENTS = ["comp1", "comp2", "comp3", "comp4"]
MODELS = ["model1", "model2", "model3", "model4"]
CLASSES = ["none", *COMPONENTS]
HORIZON = pd.Timedelta(hours=24)
MAX_HOURS_SINCE = 24 * 365.0  # cap for "hours since last replacement"

FEATURES = (
    [f"{s}_mean_3h" for s in SIGNALS]
    + [f"{s}_sd_3h" for s in SIGNALS]
    + [f"{s}_mean_24h" for s in SIGNALS]
    + [f"{s}_sd_24h" for s in SIGNALS]
    + [f"{e}_count_24h" for e in ERRORS]
    + [f"hours_since_{c}" for c in COMPONENTS]
    + ["age"]
    + [f"is_{m}" for m in MODELS]
)


def telemetry_features(tel: pd.DataFrame) -> pd.DataFrame:
    """Rolling mean/std of each signal over the last 3 h and 24 h, per machine.

    tel columns: machine_id, ts, volt, rotate, pressure, vibration.
    """
    tel = tel.sort_values(["machine_id", "ts"]).reset_index(drop=True)
    out = tel[["machine_id", "ts"]].copy()
    grouped = tel.set_index("ts").groupby("machine_id")[SIGNALS]
    for window in ("3h", "24h"):
        roll = grouped.rolling(window)
        mean = roll.mean().reset_index(drop=True)
        sd = roll.std().reset_index(drop=True)
        for s in SIGNALS:
            out[f"{s}_mean_{window}"] = mean[s].to_numpy(dtype=np.float32)
            out[f"{s}_sd_{window}"] = sd[s].to_numpy(dtype=np.float32)
    return out


def error_features(grid: pd.DataFrame, errors: pd.DataFrame) -> pd.DataFrame:
    """Count of each error type in the last 24 h, on the (machine_id, ts) grid."""
    counts = (
        pd.get_dummies(errors.set_index(["machine_id", "ts"])["error_id"])
        .reindex(columns=ERRORS, fill_value=0)
        .groupby(level=[0, 1])
        .sum()
    )
    hourly = (
        grid[["machine_id", "ts"]]
        .merge(counts.reset_index(), on=["machine_id", "ts"], how="left")
        .fillna({e: 0 for e in ERRORS})
        .sort_values(["machine_id", "ts"])
    )
    rolled = (
        hourly.set_index("ts").groupby("machine_id")[ERRORS].rolling("24h").sum().reset_index()
    )
    rolled = rolled.rename(columns={e: f"{e}_count_24h" for e in ERRORS})
    return rolled


def maintenance_features(grid: pd.DataFrame, maint: pd.DataFrame) -> pd.DataFrame:
    """Hours since the last replacement of each component (any reason), capped."""
    out = grid[["machine_id", "ts"]].sort_values("ts").reset_index(drop=True)
    for comp in COMPONENTS:
        last = (
            maint.loc[maint["component"] == comp, ["machine_id", "ts"]]
            .rename(columns={"ts": "last"})
            .assign(ts=lambda d: d["last"])
            .sort_values("ts")
        )
        merged = pd.merge_asof(out, last, on="ts", by="machine_id", direction="backward")
        hours = (merged["ts"] - merged["last"]).dt.total_seconds() / 3600.0
        out[f"hours_since_{comp}"] = hours.fillna(MAX_HOURS_SINCE).clip(upper=MAX_HOURS_SINCE)
    return out


def machine_features(machines: pd.DataFrame) -> pd.DataFrame:
    """Static features: age and one-hot model type."""
    out = machines[["machine_id", "age"]].copy()
    for m in MODELS:
        out[f"is_{m}"] = (machines["model"] == m).astype(np.int8)
    return out


def build_features(
    tel: pd.DataFrame,
    errors: pd.DataFrame,
    maint: pd.DataFrame,
    machines: pd.DataFrame,
) -> pd.DataFrame:
    """Full feature table on the telemetry grid (one row per machine and hour)."""
    feats = telemetry_features(tel)
    feats = feats.merge(error_features(feats, errors), on=["machine_id", "ts"], how="left")
    feats = feats.merge(maintenance_features(feats, maint), on=["machine_id", "ts"], how="left")
    feats = feats.merge(machine_features(machines), on="machine_id", how="left")
    return feats.sort_values(["machine_id", "ts"]).reset_index(drop=True)


def add_labels(
    feats: pd.DataFrame, failures: pd.DataFrame, horizon: pd.Timedelta = HORIZON
) -> pd.DataFrame:
    """Label each row with the component that fails within `horizon` (default 24 h), else 'none'.

    If two components fail at the same time, the first in COMPONENTS order is used.
    """
    fail = (
        failures.sort_values(["ts", "component"])
        .drop_duplicates(["machine_id", "ts"])
        .rename(columns={"ts": "fail_ts"})
        .assign(ts=lambda d: d["fail_ts"])
        .sort_values("ts")
    )
    base = feats.sort_values("ts").reset_index(drop=True)
    merged = pd.merge_asof(
        base,
        fail[["machine_id", "ts", "fail_ts", "component"]],
        on="ts",
        by="machine_id",
        direction="forward",
        allow_exact_matches=False,
    )
    within = (merged["fail_ts"] - merged["ts"]) <= horizon
    merged["label"] = np.where(within, merged["component"], "none")
    merged["next_failure_ts"] = merged["fail_ts"].where(within)
    return merged.drop(columns=["fail_ts", "component"]).sort_values(["machine_id", "ts"])
