# Deploying to Azure (Container Apps + PostgreSQL)

The live demo runs on an Azure for Students subscription in **germanywestcentral** (Frankfurt).

```
GitHub Actions (release.yml) --> ghcr.io image (code + trained model + document index)
                                        |
Azure resource group rg-maintenance-copilot
  Container Apps environment (Consumption, scale to zero, logs: none)
    copilot-api       FastAPI app, 0.5 vCPU / 1 GiB, min 0 replicas
    copilot-phoenix   Arize Phoenix with login, traces stored in Postgres
    copilot-init      manual job: download data, load machines 1-100, read-only role, demo user
  PostgreSQL Flexible Server, Burstable B1ms, 32 GB, TLS required
    databases: copilot (app + fleet), phoenix (traces); role copilot_ro for the agent's tools
```

## Cost (USD list prices from the Azure Retail Prices API, 2026-11-24)

| Item | Price | Per month |
|---|---|---|
| PostgreSQL B1ms compute | $0.0199 / hour | $14.53 |
| PostgreSQL storage, 32 GB | $0.137 / GB-month | $4.38 |
| Container Apps, scale to zero | free grant: 180,000 vCPU-s, 360,000 GiB-s, 2 M requests per month | $0 at demo traffic |
| Images on GHCR, logs destination "none" | - | $0 |

About **$18.91 per month**, so the $100 student credit lasts roughly 5 months. When the credit is
used up, Azure disables the subscription; it does not charge automatically. The first request after
an idle period waits for a cold start (both apps scale to zero).

LLM cost is billed by the LLM provider, not Azure: the demo allows **20 questions per day** for all
users together (`DAILY_QUESTION_LIMIT`), measured at $0.00035 per question with gpt-4o-mini.

## Protection

- Registration is off (`ALLOW_REGISTRATION=false`); visitors use the demo account `demo`.
- 20 questions per day in total, counted from the stored messages (survives restarts), plus the
  per-user rate limit.
- Secrets (LLM key, JWT secret, database passwords, Phoenix keys) exist only in `.env`,
  `.azure-deploy.env` (both gitignored) and as Container Apps secrets.
- The agent's data tools use the read-only role `copilot_ro` (no access to users or chats).
- Phoenix runs with login; the app sends traces with a system API key, not the admin secret.
- Postgres allows connections from Azure services only, always over TLS.

## Steps

1. **Release image.** Run the *Release image* workflow (GitHub -> Actions -> Release image -> Run
   workflow). It downloads the data, trains the model, smoke-tests the image and pushes
   `ghcr.io/<owner>/maintenance-copilot:<sha>`. Make the package public once
   (GitHub -> your profile -> Packages -> maintenance-copilot -> Package settings -> Change visibility),
   or Container Apps cannot pull it without credentials.
2. **Deploy.** With the Azure CLI logged in (`az login`) and `LLM_API_KEY` in `.env`:
   ```bash
   IMAGE=ghcr.io/<owner>/maintenance-copilot:<sha> bash scripts/azure/deploy.sh
   ```
   The script is safe to re-run. It prints the app and Phoenix URLs. The demo password is in
   `.azure-deploy.env`.
3. **Check.**
   ```bash
   python scripts/smoke_test.py --base-url https://<app-url>
   ```
   (The smoke test registers a user, so it fails by design while registration is off; log in
   with the demo account in the browser instead.)
4. **Update** after code changes: run the release workflow again, then the deploy script with the
   new tag.

## Cost control

- Stop the database between demo periods (compute billing stops, storage continues; Azure starts a
  stopped server again after 7 days):
  `az postgres flexible-server stop -g rg-maintenance-copilot -n <server>`
- Delete everything: `bash scripts/azure/teardown.sh` (asks you to type the resource group name).
