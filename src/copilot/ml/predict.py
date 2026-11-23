"""Serve the failure-risk model: load once, predict for machines at one point in time."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from copilot.ml.features import CLASSES, COMPONENTS, FEATURES, build_features


@dataclass
class HorizonRisk:
    """Risk for one horizon (24 h or 7 days)."""

    failure_probability: float
    alert: bool
    alert_threshold: float
    component_probabilities: dict[str, float]
    most_likely_component: str | None


@dataclass
class RiskResult:
    """Model output for one machine at one point in time."""

    machine_id: int
    at: datetime
    horizons: dict[str, HorizonRisk]
    notable_inputs: list[str]


class RiskModel:
    """Wrapper around the trained LightGBM models, one per horizon (loaded once)."""

    def __init__(self, bundle: dict[str, Any]) -> None:
        if bundle.get("features") != FEATURES or bundle.get("classes") != CLASSES:
            raise ValueError("model was trained with a different feature set - retrain it")
        self.horizons: dict[str, dict[str, Any]] = bundle["horizons"]
        self._order = {
            name: [list(h["model"].classes_).index(i) for i in range(len(CLASSES))]
            for name, h in self.horizons.items()
        }

    @classmethod
    def load(cls, path: Path) -> RiskModel:
        """Load a bundle written by scripts/train_model.py."""
        return cls(joblib.load(path))

    def predict_frame(self, rows: pd.DataFrame, horizon: str) -> np.ndarray:
        """Class probabilities (CLASSES order) for feature rows."""
        model = self.horizons[horizon]["model"]
        return model.predict_proba(rows[FEATURES])[:, self._order[horizon]]

    def predict_many(self, frames: dict[str, pd.DataFrame], at: datetime) -> list[RiskResult]:
        """Predict for every machine in `frames` from its history up to `at`.

        frames: telemetry (>= 24 h before `at`), errors, maint, machines.
        """
        feats = build_features(
            frames["telemetry"], frames["errors"], frames["maint"], frames["machines"]
        )
        feats = feats[feats["ts"] <= at]
        if feats.empty:
            return []
        last = feats.groupby("machine_id").tail(1).reset_index(drop=True)
        proba = {name: self.predict_frame(last, name) for name in self.horizons}
        out = []
        for i, (_, row) in enumerate(last.iterrows()):
            out.append(
                RiskResult(
                    machine_id=int(row["machine_id"]),
                    at=pd.Timestamp(row["ts"]).to_pydatetime(),
                    horizons={n: self._horizon(n, proba[n][i]) for n in self.horizons},
                    notable_inputs=notable_inputs(row),
                )
            )
        return out

    def _horizon(self, name: str, proba: np.ndarray) -> HorizonRisk:
        risk = float(1.0 - proba[0])
        threshold = float(self.horizons[name]["threshold"])
        comps = {c: round(float(p), 4) for c, p in zip(COMPONENTS, proba[1:], strict=True)}
        alert = risk >= threshold
        return HorizonRisk(
            failure_probability=round(risk, 4),
            alert=alert,
            alert_threshold=round(threshold, 4),
            component_probabilities=comps,
            most_likely_component=max(comps, key=comps.get) if alert else None,
        )


def notable_inputs(row: pd.Series) -> list[str]:
    """Human-readable notable inputs (recent errors, long-unreplaced parts)."""
    out = [
        f"{int(row[f'error{i}_count_24h'])}x error{i} in last 24 h"
        for i in range(1, 6)
        if row[f"error{i}_count_24h"] > 0
    ]
    for c in COMPONENTS:
        days = row[f"hours_since_{c}"] / 24
        if days >= 60:
            out.append(f"{c} last replaced {days:.0f} days ago")
    return out
