"""Automated CAMS fetch scheduler.

Two backends are supported:

  * In-process scheduler — uses APScheduler if available, else a simple
    threading-based loop. Suitable for the dashboard / API container.
  * Cron — `setup_cron` writes crontab entries that invoke
    `python -m solar_forecast.cams_fetcher` at the configured times.

Default schedule
----------------
The runner determines which CAMS forecast run to download from `schedule`
in the config. The scheduler calls the runner shortly after each run is
expected to be available (the `availability` UTC hour from the config).
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

from .runner import load_config, run_once

log = logging.getLogger(__name__)


# ── In-process scheduler ──────────────────────────────────────────────────

class CamsScheduler:
    """Run `run_once()` shortly after every configured CAMS run becomes
    available.

    Use as a long-running thread inside the dashboard or API process. For
    production set up cron via `setup_cron()` instead — it is more robust
    against process restarts.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        offset_minutes: int = 15,
        on_complete: Optional[Callable[[list[str]], None]] = None,
    ):
        self.config_path = config_path
        self.offset_minutes = offset_minutes
        self.on_complete = on_complete
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # --- public lifecycle ------------------------------------------------

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="cams-scheduler", daemon=True
        )
        self._thread.start()
        log.info("CAMS scheduler started")

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        log.info("CAMS scheduler stopped")

    # --- core loop -------------------------------------------------------

    def _next_run_time(self) -> datetime:
        """Compute the next UTC datetime at which a CAMS run is expected."""
        cfg = load_config(self.config_path)
        sched = cfg["schedule"]
        availability_hours = sorted(int(h) for h in sched.values())
        now = datetime.utcnow()
        for h in availability_hours:
            cand = now.replace(hour=h, minute=self.offset_minutes,
                               second=0, microsecond=0)
            if cand > now:
                return cand
        # All today's runs done — first run of tomorrow
        h0 = availability_hours[0]
        tomorrow = now + timedelta(days=1)
        return tomorrow.replace(hour=h0, minute=self.offset_minutes,
                                second=0, microsecond=0)

    def _loop(self) -> None:
        while not self._stop.is_set():
            target = self._next_run_time()
            wait_s = max(1.0, (target - datetime.utcnow()).total_seconds())
            log.info("next CAMS fetch at %s UTC (sleep %.0fs)",
                     target.isoformat(timespec="seconds"), wait_s)

            # sleep in small chunks so stop() responds quickly
            while wait_s > 0 and not self._stop.is_set():
                slice_s = min(30.0, wait_s)
                time.sleep(slice_s)
                wait_s -= slice_s
            if self._stop.is_set():
                break

            try:
                results = run_once(config_path=self.config_path, dry_run=False)
                if self.on_complete:
                    self.on_complete(results)
            except Exception as exc:
                log.error("CAMS fetch failed: %s", exc)


# ── Cron setup ────────────────────────────────────────────────────────────

def setup_cron(
    config_path: Optional[str] = None,
    log_dir: str = "logs",
    install: bool = True,
) -> str:
    """Generate (and optionally install) crontab entries for the fetcher.

    Returns the crontab fragment as a string. Calling with `install=False`
    is safe in tests (no system mutation).
    """
    cfg = load_config(config_path)
    sched = cfg["schedule"]
    project_root = Path(__file__).resolve().parents[2]

    log_path = (project_root / log_dir).resolve()

    config_arg = f" --config {config_path}" if config_path else ""
    lines = ["# cams-fetcher (auto-generated)"]
    for hour in sorted({int(h) for h in sched.values()}):
        line = (
            f"{15} {hour} * * * cd {project_root} && "
            f"python -m solar_forecast.cams_fetcher{config_arg} "
            f">> {log_path}/cams-$(date +\\%Y-\\%m-\\%d).log 2>&1 # cams-fetcher"
        )
        lines.append(line)
    fragment = "\n".join(lines) + "\n"

    if not install:
        return fragment

    if not shutil.which("crontab"):
        raise RuntimeError("crontab binary not found — set install=False")

    log_path.mkdir(parents=True, exist_ok=True)

    # Replace any existing cams-fetcher cron lines
    import subprocess
    existing = subprocess.run(
        ["crontab", "-l"], capture_output=True, text=True
    ).stdout
    keep = [ln for ln in existing.splitlines() if "cams-fetcher" not in ln]
    new_cron = "\n".join(keep + lines) + "\n"
    subprocess.run(["crontab", "-"], input=new_cron, text=True, check=True)
    log.info("installed cron entries:\n%s", fragment)
    return fragment


__all__ = ["CamsScheduler", "setup_cron"]
