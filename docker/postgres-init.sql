-- Read-only role for the agent's data tools. SELECT is granted on the fleet tables
-- (and tickets) by scripts/load_db.py after they exist - never on users or chats.
CREATE ROLE copilot_ro LOGIN PASSWORD 'copilot_ro';
GRANT CONNECT ON DATABASE copilot TO copilot_ro;
GRANT USAGE ON SCHEMA public TO copilot_ro;
