"""CAMS API client — lazy-imports cdsapi so package works without it."""

from __future__ import annotations
import logging
import os
from typing import Any

log = logging.getLogger(__name__)


def get_cams_client() -> Any:
    """Return a configured cdsapi.Client.

    Auth priority:
      1. CADS_URL + CADS_KEY env vars  (Docker / CI)
      2. CAMS_API_KEY env var  (legacy format "uid:key")
      3. ~/.cdsapirc file  (local dev)
    """
    try:
        import cdsapi
    except ImportError as exc:
        raise ImportError(
            "cdsapi is required for CAMS ingestion: pip install cdsapi"
        ) from exc

    url = os.getenv("CADS_URL", "https://ads.atmosphere.copernicus.eu/api")
    key = os.getenv("CADS_KEY") or os.getenv("CAMS_API_KEY")

    if key:
        log.info("CAMS auth: environment variable")
        return cdsapi.Client(url=url, key=key, quiet=True, progress=False)

    log.info("CAMS auth: ~/.cdsapirc")
    return cdsapi.Client(quiet=True, progress=False)


def is_cams_configured() -> bool:
    """Return True if CAMS credentials appear to be available."""
    if os.getenv("CADS_KEY") or os.getenv("CAMS_API_KEY"):
        return True
    cdsapirc = os.path.expanduser("~/.cdsapirc")
    return os.path.isfile(cdsapirc)
