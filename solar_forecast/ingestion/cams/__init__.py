"""
solar_forecast.ingestion.cams — CAMS atmospheric data ingestion.

Entry points
------------
    python -m solar_forecast.ingestion.cams.backfill --location-id 1 --days 365
    python -m solar_forecast.ingestion.cams.live     --location-id 1 --hours 12

Imports
-------
    from solar_forecast.ingestion.cams.variables import CAMS_VARIABLES, map_row
    from solar_forecast.ingestion.cams.client   import get_cams_client
    from solar_forecast.ingestion.cams.fetcher  import fetch_cams_window
    from solar_forecast.ingestion.cams.backfill import run_backfill
    from solar_forecast.ingestion.cams.live     import run_live
    from solar_forecast.ingestion.cams.scheduler import CamsIngestionScheduler
"""
