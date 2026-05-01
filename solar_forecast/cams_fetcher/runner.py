"""CAMS Fetcher runner — single-shot pipeline invoked by cron / scheduler.

Refactored from the standalone `main.py` so it can be called either from
the CLI (`python -m solar_forecast.cams_fetcher`) or from the Python
scheduler.

    from solar_forecast.cams_fetcher.runner import run_once
    run_once(config_path="cams_config.yaml", dry_run=False)
"""

from __future__ import annotations

import logging
import os
import smtplib
import sys
import tempfile
import traceback
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional

import yaml

from . import backfill, db, grib_processor
from .client import get_client

log = logging.getLogger(__name__)

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config_default.yaml"


# ── Config ────────────────────────────────────────────────────────────────

def load_config(path: Optional[str] = None) -> dict:
    p = Path(path) if path else DEFAULT_CONFIG
    with open(p) as f:
        return yaml.safe_load(f)


def parse_leadtime(spec: str) -> list[str]:
    """`'0-36'` → `['0','1',...,'36']`; `'0,3,6'` → `['0','3','6']`."""
    spec = str(spec).strip()
    if "-" in spec and "," not in spec:
        start, end = spec.split("-")
        return [str(i) for i in range(int(start), int(end) + 1)]
    return [s.strip() for s in spec.split(",")]


def determine_forecast(schedule: dict) -> tuple[str, str]:
    """Most recent forecast run available given the current UTC time."""
    now = datetime.utcnow()
    for time_str, avail_hour in sorted(schedule.items(), key=lambda x: x[1], reverse=True):
        if now.hour >= int(avail_hour):
            return now.strftime("%Y-%m-%d"), time_str
    latest = max(schedule, key=schedule.get)
    return (now - timedelta(days=1)).strftime("%Y-%m-%d"), latest


# ── Request build ─────────────────────────────────────────────────────────

def build_request(ds_cfg: dict, cfg: dict, forecast_date: str, forecast_time: str) -> dict:
    t = cfg["target"]
    lat, lon, margin = t["lat"], t["lon"], t["area_margin"]

    req = {
        "variable": ds_cfg["variables"],
        "date":     [f"{forecast_date}/{forecast_date}"],
        "time":     [forecast_time],
        "leadtime_hour": parse_leadtime(ds_cfg["leadtime_hours"]),
        "type":     ["forecast"],
        "data_format": "grib",
        "area":     [lat + margin, lon - margin, lat - margin, lon + margin],
    }

    if "model_levels" in ds_cfg:
        req["model_level"] = parse_leadtime(ds_cfg["model_levels"])

    return req


# ── Pipeline: download → parse → DB ──────────────────────────────────────

def fetch_and_insert(
    client,
    ds_cfg: dict,
    cfg: dict,
    forecast_date: str,
    forecast_time: str,
    dry_run: bool = False,
) -> str:
    name  = ds_cfg["name"]
    table = ds_cfg["target_table"]
    pk    = ds_cfg["primary_key"]
    target = cfg["target"]

    req = build_request(ds_cfg, cfg, forecast_date, forecast_time)
    log.info("[%s] downloading: %s %s", name, forecast_date, forecast_time)

    tmp = tempfile.NamedTemporaryFile(suffix=".grib", delete=False)
    tmp.close()
    try:
        result = client.retrieve(ds_cfg["dataset"], req)
        result.download(tmp.name)
        log.info("[%s] grib saved: %s", name, tmp.name)

        df_long = grib_processor.parse_grib_file(tmp.name, target["lat"], target["lon"])
        df_wide = grib_processor.pivot_and_clean(df_long, ds_cfg)

        if dry_run:
            msg = f"[DRY-RUN] [{name}] {forecast_date} {forecast_time}: {len(df_wide)} rows (not written)"
            log.info(msg)
            return msg

        conn = db.get_connection()
        cur = conn.cursor()
        try:
            db.ensure_table(cur, table, df_wide, pk)
            db.ensure_columns(cur, table, df_wide)
            conn.commit()
            inserted = db.insert_data(cur, table, df_wide, pk)
            conn.commit()
        finally:
            cur.close()
            conn.close()

        msg = f"[{name}] {forecast_date} {forecast_time}: {inserted} rows"
        log.info(msg)
        return msg

    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


# ── Phases ───────────────────────────────────────────────────────────────

def phase_live(cfg, client, forecast_date, forecast_time, dry_run) -> list[str]:
    results = []
    for ds in cfg["datasets"]:
        try:
            results.append(fetch_and_insert(client, ds, cfg, forecast_date, forecast_time, dry_run))
        except Exception as exc:
            err = f"[{ds['name']}] LIVE ERROR: {exc}"
            log.error(err)
            log.debug(traceback.format_exc())
            results.append(err)
    return results


def phase_backfill(cfg, client, dry_run) -> list[str]:
    results = []
    backfill_cfg = cfg.get("backfill", {})
    if not backfill_cfg.get("enabled", False):
        return results

    for ds in cfg["datasets"]:
        if "backfill" not in ds:
            continue
        try:
            conn = db.get_connection()
            cur = conn.cursor()
            missing = backfill.find_missing_forecasts(cur, ds, cfg["schedule"])
            cur.close()
            conn.close()
        except Exception as exc:
            results.append(f"[{ds['name']}] BACKFILL DETECT ERROR: {exc}")
            continue

        if not missing:
            results.append(f"[{ds['name']}] backfill: 0 missing")
            continue

        for d, h in missing:
            date_str = d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)
            time_str = f"{h:02d}:00"
            try:
                msg = fetch_and_insert(client, ds, cfg, date_str, time_str, dry_run)
                results.append(f"[BACKFILL] {msg}")
            except Exception as exc:
                err = f"[BACKFILL] [{ds['name']}] {date_str} {time_str} ERROR: {exc}"
                log.error(err)
                results.append(err)

    return results


# ── Email ─────────────────────────────────────────────────────────────────

def send_email(cfg: dict, results: list[str]) -> None:
    email_cfg = cfg.get("email", {})
    if not email_cfg.get("enabled", False):
        return

    smtp_server   = os.getenv("SMTP_SERVER")
    smtp_port     = int(os.getenv("SMTP_PORT", "587"))
    smtp_user     = os.getenv("SMTP_USER")
    smtp_password = os.getenv("SMTP_PASSWORD")
    recipient     = email_cfg.get("recipient", "")

    if not all([smtp_server, smtp_user, smtp_password, recipient]):
        log.warning("SMTP not fully configured, skipping email")
        return

    now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    body = (
        f"<html><body><p>CAMS Fetcher – {now_str}</p>"
        f"<pre>{chr(10).join(results)}</pre></body></html>"
    )
    msg = MIMEMultipart("alternative")
    msg["From"] = smtp_user
    msg["To"]   = recipient
    msg["Subject"] = f"CAMS Fetcher – {now_str}"
    msg.attach(MIMEText(body, "html"))

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.send_message(msg)
        log.info("email sent: %s", recipient)
    except Exception as exc:
        log.error("email error: %s", exc)


# ── Public entry point ────────────────────────────────────────────────────

def run_once(
    config_path: Optional[str] = None,
    dry_run: bool = False,
) -> list[str]:
    """Run the full fetch pipeline once. Returns the list of result strings."""
    cfg = load_config(config_path)

    forecast_date, forecast_time = determine_forecast(cfg["schedule"])
    log.info("target=%s coord=%.4f,%.4f run=%s %s",
             cfg["target"].get("name", "-"),
             cfg["target"]["lat"], cfg["target"]["lon"],
             forecast_date, forecast_time)

    client = get_client()
    log.info("ECMWF CADS client ready")

    results = []
    results.extend(phase_live(cfg, client, forecast_date, forecast_time, dry_run))
    results.extend(phase_backfill(cfg, client, dry_run))
    send_email(cfg, results)

    log.info("done — %d result(s)", len(results))
    for r in results:
        log.info("  %s", r)
    return results


# ── CLI ───────────────────────────────────────────────────────────────────

def _cli():
    import argparse
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    p = argparse.ArgumentParser(description="CAMS Fetcher")
    p.add_argument("--config", default=None, help="path to YAML config")
    p.add_argument("--dry-run", action="store_true", help="don't write to DB")
    args = p.parse_args()

    results = run_once(config_path=args.config, dry_run=args.dry_run)
    if any("ERROR" in r for r in results):
        sys.exit(1)


if __name__ == "__main__":
    _cli()
