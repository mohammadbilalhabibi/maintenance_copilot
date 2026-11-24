#!/usr/bin/env bash
# Delete EVERYTHING of the Azure deployment (resource group, database, apps). Cannot be undone.
# Asks you to type the resource group name before deleting.
set -euo pipefail
cd "$(dirname "$0")/../.."
RG=${RG:-rg-maintenance-copilot}

az group show --name "$RG" --output none 2>/dev/null || { echo "resource group $RG not found"; exit 0; }
echo "This permanently deletes resource group '$RG' with the database, apps and job."
read -r -p "Type the resource group name to confirm: " answer
[ "$answer" = "$RG" ] || { echo "not confirmed - nothing deleted"; exit 1; }
az group delete --name "$RG" --yes --no-wait
echo "deletion started (takes a few minutes). Local passwords stay in .azure-deploy.env;"
echo "delete that file if you will not redeploy."
