"""CAMS API client — thin wrapper around the ECMWF cdsapi."""

from __future__ import annotations

import logging
import os
from typing import Any

log = logging.getLogger(__name__)


def get_client() -> Any:
    """Return a configured `cdsapi.Client`.

    Authentication order:
      1. `CADS_URL` + `CADS_KEY` environment variables (Docker / CI)
      2. `~/.cdsapirc` file (local development)

    `~/.cdsapirc` format::

        url: https://ads.atmosphere.copernicus.eu/api
        key: <UID>:<API-KEY>
    """
    try:
        import cdsapi
    except ImportError as exc:
        raise ImportError(
            "cdsapi is required for CAMS access. "
            "Install with: pip install cdsapi"
        ) from exc

    url = os.getenv("CADS_URL")
    key = os.getenv("CADS_KEY") or os.getenv("CAMS_API_KEY")

    if url and key:
        log.info("CADS auth: from environment variables")
        return cdsapi.Client(url=url, key=key, quiet=True, progress=False)

    log.info("CADS auth: from ~/.cdsapirc")
    return cdsapi.Client(quiet=True, progress=False)
