"""CAMS ingestion scheduler — daemon thread + cron installer.

Usage (daemon thread, e.g. in app startup):
    from solar_forecast.ingestion.cams.scheduler import CamsIngestionScheduler
    sched = CamsIngestionScheduler(location_ids=[1, 2])
    sched.start()          # background daemon thread

Usage (cron):
    from solar_forecast.ingestion.cams.scheduler import setup_cron
    setup_cron()           # installs two crontab entries (03:15 and 15:15 UTC)
"""

from __future__ import annotations
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Sequence

log = logging.getLogger(__name__)

# CAMS runs are available ~3 hours after the nominal run time (00Z → ~03Z, 12Z → ~15Z)
# We schedule slightly after to avoid hitting a not-yet-ready dataset.
_CRON_ENTRIES = [
    ("15", "3"),   # 03:15 UTC  — 00Z run
    ("15", "15"),  # 15:15 UTC  — 12Z run
]
_CHECK_INTERVAL_S = 300   # poll every 5 min; skip if not time yet


class CamsIngestionScheduler(threading.Thread):
    """Background daemon that fetches live CAMS data for all registered locations.

    Wakes up every _CHECK_INTERVAL_S seconds and fires run_live() if
    the current UTC time is within the fetch window (03:10–04:00 or 15:10–16:00).
    """

    def __init__(self, location_ids: Sequence[int], hours: int = 48):
        super().__init__(name="CamsIngestionScheduler", daemon=True)
        self.location_ids = list(location_ids)
        self.hours = hours
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def _in_fetch_window(self) -> bool:
        now = datetime.now(timezone.utc)
        h, m = now.hour, now.minute
        return (
            (h == 3 and 10 <= m <= 59) or
            (h == 4 and m <= 0) or
            (h == 15 and 10 <= m <= 59) or
            (h == 16 and m <= 0)
        )

    def run(self):
        log.info("CamsIngestionScheduler started (locations=%s, horizon=%dh)",
                 self.location_ids, self.hours)
        while not self._stop_event.is_set():
            if self._in_fetch_window():
                self._run_all()
            self._stop_event.wait(_CHECK_INTERVAL_S)
        log.info("CamsIngestionScheduler stopped")

    def _run_all(self):
        from .live import run_live
        for loc_id in self.location_ids:
            try:
                status = run_live(location_id=loc_id, hours=self.hours)
                log.info("scheduler: loc %d → %s", loc_id, status)
            except Exception as exc:
                log.error("scheduler: loc %d failed: %s", loc_id, exc)


def setup_cron(python_path: str = "python", module: str = "solar_forecast.ingestion.cams.live") -> None:
    """Install crontab entries to run live CAMS fetch at 03:15 and 15:15 UTC.

    Existing entries for the same module are removed first.
    """
    import subprocess
    import shlex

    try:
        existing = subprocess.check_output(["crontab", "-l"], stderr=subprocess.DEVNULL).decode()
    except subprocess.CalledProcessError:
        existing = ""

    # Remove stale lines referencing this module
    lines = [l for l in existing.splitlines() if module not in l]

    for minute, hour in _CRON_ENTRIES:
        cmd = f"{python_path} -m {module} --location-id ALL --hours 48"
        lines.append(f"{minute} {hour} * * * {cmd}")

    new_crontab = "\n".join(lines) + "\n"
    proc = subprocess.run(
        ["crontab", "-"],
        input=new_crontab.encode(),
        capture_output=True,
    )
    if proc.returncode == 0:
        log.info("crontab updated with CAMS fetch entries")
    else:
        log.error("crontab update failed: %s", proc.stderr.decode())
