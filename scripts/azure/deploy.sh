#!/usr/bin/env bash
# Deploy Maintenance Copilot to Azure: Container Apps (api, Phoenix, init job) + Postgres.
# Safe to re-run (creates what is missing, updates the apps). See docs/deploy_azure.md.
#
#   IMAGE=ghcr.io/<owner>/maintenance-copilot:<tag> bash scripts/azure/deploy.sh
#
# Reads LLM_API_KEY / LLM_MODEL from .env. Generated passwords and keys are kept in
# .azure-deploy.env (gitignored) and passed to Azure as Container Apps secrets.
set -euo pipefail
cd "$(dirname "$0")/../.."

RG=${RG:-rg-maintenance-copilot}
LOC=${LOC:-germanywestcentral}
IMAGE=${IMAGE:?set IMAGE=ghcr.io/<owner>/maintenance-copilot:<tag>}
PHOENIX_IMAGE=arizephoenix/phoenix:version-21.2.0
ENV_NAME=mcopilot-env
STATE=.azure-deploy.env
DAILY_LIMIT=${DAILY_LIMIT:-20}

step() { echo; echo "== $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }
from_env() { grep -E "^$1=" .env 2>/dev/null | head -1 | cut -d= -f2- || true; }
rand() { python -c "import secrets; print('$1' + secrets.token_urlsafe($2).replace('-', 'x').replace('_', 'y'))"; }
# Azure sometimes answers a successful change with a transient InternalServerError: retry.
retry() { for i in 1 2 3; do "$@" && return 0; echo "   (attempt $i failed, retrying in 20 s)"; sleep 20; done; return 1; }

LLM_API_KEY=$(from_env LLM_API_KEY)
LLM_MODEL=$(from_env LLM_MODEL); LLM_MODEL=${LLM_MODEL:-gpt-4o-mini}
[ -n "$LLM_API_KEY" ] || die "LLM_API_KEY missing in .env"

if [ ! -f "$STATE" ]; then
  step "Generating passwords and keys into $STATE (kept local, gitignored)"
  {
    echo "SUFFIX=$(python -c 'import secrets; print(secrets.token_hex(3))')"
    echo "PG_ADMIN_PASSWORD=$(rand Pg1 24)"
    echo "RO_PASSWORD=$(rand Ro1 24)"
    echo "JWT_SECRET=$(rand J 48)"
    echo "PHOENIX_SECRET=$(rand Ph1 32)"
    echo "PHOENIX_ADMIN_SECRET=$(rand Ad1 32)"
    echo "DEMO_PASSWORD=$(rand Demo 9)"
    echo "PHOENIX_ADMIN_PASSWORD=$(rand Px1 24)"
  } > "$STATE"
fi
# shellcheck disable=SC1090
source "$STATE"
PG="mcopilot-pg-$SUFFIX"
PGHOST="$PG.postgres.database.azure.com"
DB_URL="postgresql+asyncpg://copilotadmin:$PG_ADMIN_PASSWORD@$PGHOST:5432/copilot?ssl=require"
RO_URL="postgresql+asyncpg://copilot_ro:$RO_PASSWORD@$PGHOST:5432/copilot?ssl=require"
PHX_DB_URL="postgresql://copilotadmin:$PG_ADMIN_PASSWORD@$PGHOST:5432/phoenix?sslmode=require"

step "Azure CLI extension and resource providers"
az extension add --name containerapp --upgrade --yes --only-show-errors
for ns in Microsoft.App Microsoft.DBforPostgreSQL Microsoft.OperationalInsights; do
  az provider register --namespace "$ns" --wait
done

step "Resource group $RG (resources in $LOC)"
az group show --name "$RG" --output none 2>/dev/null   || az group create --name "$RG" --location "$LOC" --output none

step "Postgres Flexible Server $PG (Burstable B1ms, 32 GB, TLS)"
if ! az postgres flexible-server show -g "$RG" -n "$PG" --output none 2>/dev/null; then
  az postgres flexible-server create -g "$RG" -n "$PG" -l "$LOC" \
    --tier Burstable --sku-name Standard_B1ms --storage-size 32 --version 16 \
    --admin-user copilotadmin --admin-password "$PG_ADMIN_PASSWORD" \
    --public-access 0.0.0.0 --yes --output none
fi
for db in copilot phoenix; do
  az postgres flexible-server db show -g "$RG" -s "$PG" -n "$db" --output none 2>/dev/null \
    || az postgres flexible-server db create -g "$RG" -s "$PG" -n "$db" --output none
done

step "Container Apps environment $ENV_NAME (Consumption only: supports jobs; logs: none)"
az containerapp env show -g "$RG" -n "$ENV_NAME" --output none 2>/dev/null \
  || az containerapp env create -g "$RG" -n "$ENV_NAME" -l "$LOC" --logs-destination none \
       --environment-mode ConsumptionOnly --output none

# create or update a container app; extra args: secrets then env vars
upsert_app() {
  local name=$1 image=$2 port=$3 secrets=$4 envs=$5
  if az containerapp show -g "$RG" -n "$name" --output none 2>/dev/null; then
    # shellcheck disable=SC2086
    retry az containerapp secret set -g "$RG" -n "$name" --secrets $secrets --output none
    # shellcheck disable=SC2086
    retry az containerapp update -g "$RG" -n "$name" --image "$image" --set-env-vars $envs --output none
  else
    # shellcheck disable=SC2086
    retry az containerapp create -g "$RG" -n "$name" --environment "$ENV_NAME" --image "$image" \
      --target-port "$port" --ingress external --min-replicas 0 --max-replicas 2 \
      --cpu 0.5 --memory 1.0Gi --secrets $secrets --env-vars $envs --output none
  fi
}

step "Phoenix (login on, traces stored in the phoenix database)"
upsert_app copilot-phoenix "$PHOENIX_IMAGE" 6006 \
  "phx-db=$PHX_DB_URL phx-secret=$PHOENIX_SECRET phx-admin=$PHOENIX_ADMIN_SECRET phx-admin-pw=$PHOENIX_ADMIN_PASSWORD" \
  "PHOENIX_SQL_DATABASE_URL=secretref:phx-db PHOENIX_ENABLE_AUTH=true PHOENIX_SECRET=secretref:phx-secret PHOENIX_ADMIN_SECRET=secretref:phx-admin PHOENIX_DEFAULT_ADMIN_INITIAL_PASSWORD=secretref:phx-admin-pw"
PHX_FQDN=$(az containerapp show -g "$RG" -n copilot-phoenix --query properties.configuration.ingress.fqdn -o tsv)

if [ -z "${PHOENIX_API_KEY:-}" ]; then
  step "Creating a Phoenix system API key for the app (waits for Phoenix to start)"
  for _ in $(seq 1 40); do curl -fsS -o /dev/null "https://$PHX_FQDN/healthz" && break; sleep 6; done
  PHOENIX_API_KEY=$(curl -fsS -X POST "https://$PHX_FQDN/graphql" \
    -H "Content-Type: application/json" -H "authorization: Bearer $PHOENIX_ADMIN_SECRET" \
    -d '{"query":"mutation { createSystemApiKey(input: {name: \"copilot-traces\"}) { jwt } }"}' \
    | python -c "import json,sys; print(json.load(sys.stdin)['data']['createSystemApiKey']['jwt'])")
  [ -n "$PHOENIX_API_KEY" ] || die "could not create the Phoenix API key"
  echo "PHOENIX_API_KEY=$PHOENIX_API_KEY" >> "$STATE"
fi

step "Init job (download data, load machines 1-100, read-only role, demo account)"
JOB_SECRETS="db-url=$DB_URL ro-url=$RO_URL demo-pw=$DEMO_PASSWORD"
JOB_ENVS="DATABASE_URL=secretref:db-url TOOLS_DATABASE_URL=secretref:ro-url LOAD_ARGS=--create-ro-role DEMO_USERNAME=demo DEMO_PASSWORD=secretref:demo-pw"
if az containerapp job show -g "$RG" -n copilot-init --output none 2>/dev/null; then
  # shellcheck disable=SC2086
  retry az containerapp job secret set -g "$RG" -n copilot-init --secrets $JOB_SECRETS --output none
  # shellcheck disable=SC2086
  retry az containerapp job update -g "$RG" -n copilot-init --image "$IMAGE" --set-env-vars $JOB_ENVS --output none
else
  # shellcheck disable=SC2086
  retry az containerapp job create -g "$RG" -n copilot-init --environment "$ENV_NAME" \
    --trigger-type Manual --replica-timeout 3600 --replica-retry-limit 0 \
    --replica-completion-count 1 --parallelism 1 --image "$IMAGE" --cpu 2 --memory 4Gi \
    --command sh --args docker/init.sh --secrets $JOB_SECRETS --env-vars $JOB_ENVS --output none
fi
EXEC=$(az containerapp job start -g "$RG" -n copilot-init --query name -o tsv)
echo "started execution $EXEC (first run downloads ~810 MB, a few minutes)"
for _ in $(seq 1 120); do
  STATUS=$(az containerapp job execution show -g "$RG" -n copilot-init --job-execution-name "$EXEC" \
    --query properties.status -o tsv)
  case "$STATUS" in
    Succeeded) echo "init job succeeded"; break ;;
    Failed|Stopped|Degraded) die "init job $STATUS (see: az containerapp job logs show -g $RG -n copilot-init --execution $EXEC --container copilot-init)" ;;
  esac
  sleep 15
done

step "API (scale to zero, registration off, $DAILY_LIMIT questions per day)"
upsert_app copilot-api "$IMAGE" 8000 \
  "db-url=$DB_URL ro-url=$RO_URL llm-key=$LLM_API_KEY jwt=$JWT_SECRET phx-key=$PHOENIX_API_KEY" \
  "DATABASE_URL=secretref:db-url TOOLS_DATABASE_URL=secretref:ro-url LLM_API_KEY=secretref:llm-key LLM_MODEL=$LLM_MODEL JWT_SECRET=secretref:jwt PHOENIX_COLLECTOR_ENDPOINT=https://$PHX_FQDN/v1/traces PHOENIX_API_KEY=secretref:phx-key ALLOW_REGISTRATION=false DAILY_QUESTION_LIMIT=$DAILY_LIMIT CHECKPOINT_DURABILITY=exit"
API_FQDN=$(az containerapp show -g "$RG" -n copilot-api --query properties.configuration.ingress.fqdn -o tsv)

step "Done"
echo "App:     https://$API_FQDN   (demo login: user 'demo', password in $STATE)"
echo "Phoenix: https://$PHX_FQDN   (login admin@localhost, password PHOENIX_ADMIN_PASSWORD in $STATE)"
