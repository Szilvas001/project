#!/usr/bin/env python
"""
Step 1 — Download CAMS historical data to PostgreSQL.

Downloads:
  • CAMS EAC4 reanalysis: AOD, ozone, water vapour, pressure, cloud cover
  • CAMS solar radiation timeseries: hourly GHI / DHI / DNI (all-sky + clear-sky)

Both datasets are stored in PostgreSQL for model training.

Usage:
    python scripts/01_download_cams.py
    python scripts/01_download_cams.py --config my_config.yaml
    python scripts/01_download_cams.py --start 2022-01-01 --end 2023-12-31

Prerequisites:
    1. Register at https://ads.atmosphere.copernicus.eu/
    2. Set CAMS_API_KEY=uid:key in .env (or config.yaml)
    3. PostgreSQL running and accessible (see config.yaml [database])
"""

import argparse
import logging
import sys
from pathlib import Path

# Allow running from project root without install
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from solar_forecast.utils import load_config
from solar_forecast.data_ingestion.db_manager import DBManager
from solar_forecast.data_ingestion.cams_loader import CamsLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("download_cams")


def parse_args():
    p = argparse.ArgumentParser(description="Download CAMS training data.")
    p.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    p.add_argument("--start", default=None,
                   help="Start date YYYY-MM-DD (default: config.cams.training_start)")
    p.add_argument("--end", default=None,
                   help="End date YYYY-MM-DD (default: config.cams.training_end)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    start = args.start or cfg["cams"].get("training_start", "2021-01-01")
    end   = args.end   or cfg["cams"].get("training_end",   "2024-12-31")

    logger.info("═" * 60)
    logger.info("CAMS download: %s → %s", start, end)
    logger.info("Location: %(lat)s°, %(lon)s°  %(name)s", cfg["location"])
    logger.info("═" * 60)

    db = DBManager(cfg)
    db.create_tables()

    loader = CamsLoader(cfg, db)

    try:
        counts = loader.run_backfill(start, end)
        logger.info("Download complete.")
        logger.info("  cams_atmo rows inserted   : %d", counts.get("cams_atmo", 0))
        logger.info("  cams_radiation rows inserted: %d", counts.get("cams_radiation", 0))
    except RuntimeError as exc:
        logger.error(str(exc))
        logger.error("Set CAMS_API_KEY in .env and retry.")
        sys.exit(1)


if __name__ == "__main__":
    main()
