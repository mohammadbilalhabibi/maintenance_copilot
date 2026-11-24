#!/bin/sh
# One-shot setup job for docker compose: data download, model training, database load.
# Each step is skipped when its output already exists (data/ and models/ are bind mounts).
set -e
cd /app
if [ ! -f data/raw/telemetry.csv ]; then
  echo "== downloading sample data (about 810 MB, once)"
  python scripts/download_data.py
fi
if [ ! -f models/failure_risk.joblib ]; then
  echo "== training the failure-risk model (a few minutes, once)"
  python scripts/train_model.py ${TRAIN_ARGS:---no-mlflow}
fi
echo "== loading the fleet into the database (skipped if already loaded)"
python scripts/load_db.py --if-empty ${LOAD_ARGS:-}
if [ -n "${DEMO_USERNAME:-}" ]; then
  echo "== creating the demo account (password from DEMO_PASSWORD)"
  python scripts/create_user.py "$DEMO_USERNAME"
fi
echo "== init done"
