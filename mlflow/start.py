"""Create the backend database if it is missing, then exec `mlflow server` with the given arguments.

Runs on every start, so it covers a fresh Postgres volume and an existing one (docker-entrypoint-initdb.d
scripts only run on an empty volume).
"""
import os
import sys
import time
from urllib.parse import urlsplit

import psycopg
from psycopg import errors, sql

uri = os.environ["MLFLOW_BACKEND_STORE_URI"]
name = urlsplit(uri).path.lstrip("/")

for attempt in range(30):
    try:
        conn = psycopg.connect(uri, dbname="postgres", connect_timeout=5, autocommit=True)
        break
    except psycopg.OperationalError as exc:
        if attempt == 29:
            raise
        print(f"waiting for Postgres: {exc}".strip(), file=sys.stderr)
        time.sleep(2)

with conn:
    if conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,)).fetchone() is None:
        try:
            conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
            print(f"created database {name}", file=sys.stderr)
        except errors.DuplicateDatabase:
            pass

os.execvp("mlflow", ["mlflow", "server", *sys.argv[1:]])
