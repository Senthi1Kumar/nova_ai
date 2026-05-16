-- Nova memory layer: one-shot pgvector setup.
--
-- Prereqs (Ubuntu / Debian):
--
--   # Install Postgres + the client.
--   sudo apt-get update
--   sudo apt-get install -y postgresql postgresql-contrib
--
--   # Start it (Postgres on Ubuntu is enabled by default; on minimal images
--   # or some distros you may need to enable + start it manually):
--   sudo systemctl enable --now postgresql
--
--   # Install pgvector. The apt package name tracks the Postgres major
--   # version (e.g. postgresql-16-pgvector). If your distro doesn't ship
--   # one for your Postgres version, build from source instead.
--   PG_MAJOR=$(psql --version | awk '{print $3}' | cut -d. -f1)
--   sudo apt-get install -y "postgresql-$PG_MAJOR-pgvector" || (
--     sudo apt-get install -y build-essential postgresql-server-dev-$PG_MAJOR git
--     git clone --branch v0.8.2 https://github.com/pgvector/pgvector.git /tmp/pgvector
--     cd /tmp/pgvector && make && sudo make install
--   )
--
--   # Create the Nova database + user (one-time).
--   sudo -u postgres psql -c "CREATE USER nova WITH PASSWORD 'nova_dev';"
--   sudo -u postgres psql -c "CREATE DATABASE nova_db OWNER nova;"
--
-- Then run this file against the Nova database. NOTE: `CREATE EXTENSION
-- vector` requires SUPERUSER, so run as the `postgres` role, not `nova`:
--   sudo -u postgres psql -d nova_db -f nova/backend/sql/setup_pgvector.pgsql
-- After this runs once, the `nova` role can use vector columns normally
-- without superuser — only the install step needs root.
--
-- mem0 creates the actual memories table on first use (CREATE TABLE IF NOT
-- EXISTS), so this script just enables the extension and confirms it loaded.

CREATE EXTENSION IF NOT EXISTS vector;

-- Health check: confirm the extension is registered. Output should include
-- a row with extname='vector' and a non-null version.
SELECT extname, extversion
  FROM pg_extension
 WHERE extname = 'vector';
