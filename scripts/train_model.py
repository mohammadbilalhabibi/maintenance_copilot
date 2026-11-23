"""Train the failure-risk model, compare it with baselines, log to MLflow, save artifacts.

Two horizons: failure within 24 h and within 7 days. Writes models/failure_risk.joblib and
results/model_metrics.json (measured numbers).
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from copilot.config import PROJECT_ROOT
from copilot.ml.features import CLASSES, FEATURES
from copilot.ml.train import (
    class_proba,
    evaluate_scores,
    load_raw,
    make_dataset,
    rule_score,
    split,
    threshold_for_rate,
    train_lgbm,
    train_logreg,
)

LGBM_PARAMS = {
    "n_estimators": 2000,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
}


HORIZONS: dict[str, tuple[pd.Timedelta, float]] = {
    # name: (label horizon, share of machine-hours the alert threshold flags)
    "24h": (pd.Timedelta(hours=24), 0.02),
    "7d": (pd.Timedelta(days=7), 0.10),
}


def run_horizon(raw: dict, name: str, horizon: pd.Timedelta, rate: float, seed: int) -> dict:
    """Train baselines and LightGBM for one horizon; return metrics and the fitted model."""
    t0 = time.perf_counter()
    data = make_dataset(raw, horizon=horizon)
    s = split(data, horizon)
    print(f"[{name}] {len(data):,} rows ({time.perf_counter() - t0:.0f} s); "
          f"train {len(s.train):,} val {len(s.val):,} test {len(s.test):,}")
    results: dict[str, dict] = {}

    def score(model_name: str, val_risk, test_risk, val_p=None, test_p=None) -> None:  # noqa: ANN001
        thr = threshold_for_rate(val_risk, rate)
        results[model_name] = {
            "threshold": thr,
            "val": asdict(evaluate_scores(s.val, val_risk, thr, val_p)),
            "test": asdict(evaluate_scores(s.test, test_risk, thr, test_p)),
        }
        t = results[model_name]["test"]
        print(f"[{name}] {model_name:9s} PR-AUC {t['pr_auc_any']:.3f}  "
              f"recall {t['recall_at_rate']:.3f}  precision {t['precision_at_rate']:.3f}  "
              f"caught {t['failures_caught']:.3f}  recall>24h {t['recall_early']:.3f}")

    rng = np.random.default_rng(seed)
    score("random", rng.random(len(s.val)), rng.random(len(s.test)))
    score("rule", rule_score(s.val), rule_score(s.test))
    t1 = time.perf_counter()
    logreg = train_logreg(s.train, seed)
    lv, lt = class_proba(logreg, s.val), class_proba(logreg, s.test)
    score("logreg", 1 - lv[:, 0], 1 - lt[:, 0], lv, lt)
    results["logreg"]["train_seconds"] = time.perf_counter() - t1
    t1 = time.perf_counter()
    lgbm = train_lgbm(s.train, s.val, LGBM_PARAMS, seed)
    gv, gt = class_proba(lgbm, s.val), class_proba(lgbm, s.test)
    score("lightgbm", 1 - gv[:, 0], 1 - gt[:, 0], gv, gt)
    results["lightgbm"]["train_seconds"] = time.perf_counter() - t1
    results["lightgbm"]["best_iteration"] = int(lgbm.best_iteration_ or 0)
    importance = sorted(
        zip(FEATURES, lgbm.booster_.feature_importance("gain"), strict=True), key=lambda x: -x[1]
    )
    return {
        "model": lgbm,
        "summary": {
            "horizon_hours": horizon.total_seconds() / 3600,
            "alert_rate": rate,
            "rows": {"train": len(s.train), "val": len(s.val), "test": len(s.test)},
            "positive_rate": {
                k: float((getattr(s, k)["label"] != "none").mean())
                for k in ("train", "val", "test")
            },
            "models": results,
            "top_features": [[f, float(g)] for f, g in importance[:10]],
        },
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument("--max-machine", type=int, default=None, help="use machines 1..N only")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "models" / "failure_risk.joblib")
    parser.add_argument(
        "--results", type=Path, default=PROJECT_ROOT / "results" / "model_metrics.json"
    )
    args = parser.parse_args()

    t0 = time.perf_counter()
    raw = load_raw(args.raw, args.max_machine)
    runs = {
        name: run_horizon(raw, name, h, rate, args.seed) for name, (h, rate) in HORIZONS.items()
    }
    payload = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "data": "Microsoft Predictive Maintenance Modelling Guide sample (simulated), CC BY 4.0",
        "machines": int(raw["machines"]["machine_id"].nunique()),
        "split": {"train_end": "2015-08-01", "val_end": "2015-10-01", "gap": "one horizon"},
        "step_hours": 3,
        "seed": args.seed,
        "lgbm_params": LGBM_PARAMS,
        "horizons": {name: r["summary"] for name, r in runs.items()},
        "total_seconds": time.perf_counter() - t0,
        "cpu": platform.processor(),
    }
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(payload, indent=2))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "features": FEATURES,
            "classes": CLASSES,
            "created_at": payload["created_at"],
            "horizons": {
                name: {
                    "model": r["model"],
                    "threshold": r["summary"]["models"]["lightgbm"]["threshold"],
                    "alert_rate": r["summary"]["alert_rate"],
                    "test_metrics": r["summary"]["models"]["lightgbm"]["test"],
                }
                for name, r in runs.items()
            },
        },
        args.out,
    )
    print(f"saved {args.out} and {args.results}")

    if not args.no_mlflow:
        import mlflow

        mlflow.set_tracking_uri(f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}")
        mlflow.set_experiment("failure-risk")
        for hname, r in runs.items():
            for mname, res in r["summary"]["models"].items():
                with mlflow.start_run(run_name=f"{hname}-{mname}"):
                    mlflow.log_params({"horizon": hname, "model": mname, "seed": args.seed,
                                       "alert_rate": r["summary"]["alert_rate"],
                                       "machines": payload["machines"]})
                    if mname == "lightgbm":
                        mlflow.log_params(LGBM_PARAMS)
                    for split_name in ("val", "test"):
                        for k, v in res[split_name].items():
                            if isinstance(v, float) and v == v:
                                mlflow.log_metric(f"{split_name}_{k}", v)
                    if mname == "lightgbm":
                        mlflow.log_artifact(str(args.results))
        print("logged runs to MLflow (mlflow ui --backend-store-uri sqlite:///mlflow.db)")


if __name__ == "__main__":
    main()
