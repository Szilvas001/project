"""PostgreSQL persistence for CAMS data.

`psycopg2` is imported lazily so the rest of the project (which uses SQLite
for application state) does not require a Postgres driver in the test image.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)

# Pandas dtype → PostgreSQL type
_PG_TYPES = {
    "int64":  "BIGINT",
    "int32":  "INTEGER",
    "float64": "DOUBLE PRECISION",
    "float32": "REAL",
    "bool":   "BOOLEAN",
    "object": "TEXT",
    "datetime64[ns, UTC]": "TIMESTAMPTZ",
    "datetime64[ns]":      "TIMESTAMP",
}


def _import_psycopg2():
    try:
        import psycopg2
        import psycopg2.extras
        from psycopg2 import sql
        return psycopg2, sql
    except ImportError as exc:
        raise ImportError(
            "psycopg2 is required for the CAMS PostgreSQL store. "
            "Install with: pip install psycopg2-binary"
        ) from exc


def get_connection() -> Any:
    """Open a PostgreSQL connection.

    Reads `DATABASE_URL` if present, otherwise `DB_HOST`/`DB_PORT`/`DB_NAME`/
    `DB_USER`/`DB_PASSWORD`. Falls back to `PGHOST`/`PGPORT`/`PGDATABASE`/
    `PGUSER`/`PGPASSWORD` (matching the rest of this project's env layout).
    """
    psycopg2, _ = _import_psycopg2()

    database_url = os.getenv("DATABASE_URL")
    if database_url:
        return psycopg2.connect(database_url)

    host = os.getenv("DB_HOST") or os.getenv("PGHOST", "localhost")
    port = int(os.getenv("DB_PORT") or os.getenv("PGPORT", "5432"))
    name = os.getenv("DB_NAME") or os.getenv("PGDATABASE", "cams")
    user = os.getenv("DB_USER") or os.getenv("PGUSER", "cams")
    pw   = os.getenv("DB_PASSWORD") or os.getenv("PGPASSWORD", "")

    return psycopg2.connect(host=host, port=port, dbname=name, user=user, password=pw)


def _pg_type(series: pd.Series) -> str:
    return _PG_TYPES.get(str(series.dtype), "TEXT")


def _table_exists(cur, table: str) -> bool:
    cur.execute(
        "SELECT 1 FROM information_schema.tables WHERE table_name = %s", (table,)
    )
    return cur.fetchone() is not None


def ensure_table(cur, table: str, df: pd.DataFrame, primary_key: list[str]) -> None:
    """Create the target table if it doesn't exist."""
    _, sql = _import_psycopg2()
    if _table_exists(cur, table):
        return

    col_defs = [
        sql.SQL("{} {}").format(sql.Identifier(col), sql.SQL(_pg_type(df[col])))
        for col in df.columns
    ]
    pk_cols = sql.SQL(", ").join(sql.Identifier(k) for k in primary_key)

    stmt = sql.SQL(
        "CREATE TABLE IF NOT EXISTS {tbl} ({cols}, PRIMARY KEY ({pk}))"
    ).format(
        tbl=sql.Identifier(table),
        cols=sql.SQL(", ").join(col_defs),
        pk=pk_cols,
    )
    cur.execute(stmt)
    log.info("created table %s", table)


def ensure_columns(cur, table: str, df: pd.DataFrame) -> None:
    """`ALTER TABLE ADD COLUMN IF NOT EXISTS` for any new variables."""
    _, sql = _import_psycopg2()
    cur.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name = %s",
        (table,),
    )
    existing = {row[0] for row in cur.fetchall()}

    for col in df.columns:
        if col not in existing:
            cur.execute(
                sql.SQL("ALTER TABLE {} ADD COLUMN IF NOT EXISTS {} {}").format(
                    sql.Identifier(table),
                    sql.Identifier(col),
                    sql.SQL(_pg_type(df[col])),
                )
            )
            log.info("added column %s.%s", table, col)


def insert_data(cur, table: str, df: pd.DataFrame, primary_key: list[str]) -> int:
    """Upsert rows; updates non-PK columns on conflict."""
    psycopg2, sql = _import_psycopg2()
    if df.empty:
        return 0

    cols = list(df.columns)
    non_pk = [c for c in cols if c not in primary_key]

    col_ids = sql.SQL(", ").join(sql.Identifier(c) for c in cols)
    placeholders = sql.SQL(", ").join(sql.Placeholder() * len(cols))
    pk_ids = sql.SQL(", ").join(sql.Identifier(k) for k in primary_key)

    if non_pk:
        update_clause = sql.SQL(", ").join(
            sql.SQL("{} = EXCLUDED.{}").format(sql.Identifier(c), sql.Identifier(c))
            for c in non_pk
        )
        conflict_action = sql.SQL("DO UPDATE SET ") + update_clause
    else:
        conflict_action = sql.SQL("DO NOTHING")

    stmt = sql.SQL(
        "INSERT INTO {tbl} ({cols}) VALUES ({vals}) ON CONFLICT ({pk}) {action}"
    ).format(
        tbl=sql.Identifier(table),
        cols=col_ids,
        vals=placeholders,
        pk=pk_ids,
        action=conflict_action,
    )

    records = [
        tuple(
            v.isoformat() if isinstance(v, pd.Timestamp) else (None if pd.isna(v) else v)
            for v in row
        )
        for row in df.itertuples(index=False, name=None)
    ]

    psycopg2.extras.execute_batch(cur, stmt, records, page_size=500)
    return len(records)


# ── Read side: feed into the forecast pipeline ────────────────────────────

def read_latest_forecast(
    cur,
    table: str,
    target_time: pd.Timestamp,
    horizon_hours: int = 168,
) -> pd.DataFrame:
    """Return the most recent CAMS forecast covering `target_time` ± horizon.

    The forecast pipeline calls this to enrich Open-Meteo with CAMS atmospheric
    state (AOD, ozone, water vapour, BLH, …). Returns an empty frame if the
    table does not exist or has no matching rows.
    """
    if not _table_exists(cur, table):
        return pd.DataFrame()

    target_iso = target_time.tz_convert("UTC").isoformat() if target_time.tzinfo else target_time.isoformat()
    cur.execute(
        f"""
        SELECT *
          FROM {table}
         WHERE reference_time <= %s
           AND reference_time + (forecast_hours || ' hours')::interval >= %s
         ORDER BY reference_time DESC, forecast_hours ASC
         LIMIT %s
        """,
        (target_iso, target_iso, horizon_hours),
    )
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    return pd.DataFrame(rows, columns=cols)
