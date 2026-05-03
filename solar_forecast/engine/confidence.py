"""Confidence model — estimate forecast quality from data availability.

Returns confidence_pct (0-100), a label (Low/Medium/High/Very High),
and a list of human-readable reasons.

Confidence factors
------------------
  +30  CAMS atmospheric data available
  +20  Open-Meteo data available
  +15  AI Kt model applied
  +10  Historical trainer model applied
  +10  Multiple forecast days (≥3)
  - 5  Demo / climatology fallback active
  - 5  Poor recent forecast accuracy (placeholder)
  Base: 35 (pure climatology demo mode)
"""

from __future__ import annotations
from typing import Optional


def compute_confidence(
    atmosphere_source: str = "climatology",
    has_openmeteo: bool = True,
    use_ai: bool = False,
    has_historical_model: bool = False,
    horizon_days: int = 7,
    technology: str = "mono_si",
    sr_csv: Optional[str] = None,
) -> dict:
    """Compute forecast confidence.

    Parameters
    ----------
    atmosphere_source : 'cams' or 'climatology'
    has_openmeteo     : True if Open-Meteo data was fetched successfully
    use_ai            : True if AI Kt model was applied
    has_historical_model : True if HistoricalGHITrainer was applied
    horizon_days      : Forecast horizon
    technology        : PV technology string
    sr_csv            : Custom SR CSV path (if any)

    Returns
    -------
    dict with keys: confidence_pct, confidence_label, confidence_reasons
    """
    score = 35
    reasons: list[str] = []

    if atmosphere_source == "cams":
        score += 30
        reasons.append("CAMS atmospheric data used (aerosols, ozone, water vapour)")
    else:
        score -= 5
        reasons.append("Climatological aerosol fallback (no live CAMS data)")

    if has_openmeteo:
        score += 20
        reasons.append("Live Open-Meteo weather forecast integrated")
    else:
        reasons.append("Open-Meteo unavailable; clear-sky + physics fallback")

    if use_ai:
        score += 15
        reasons.append("AI Kt correction model applied (XGBoost)")

    if has_historical_model:
        score += 10
        reasons.append("Historical GHI trainer correction applied")

    if horizon_days >= 3:
        score += 10
        reasons.append(f"Multi-day forecast ({horizon_days}d) with SPECTRL2 physics")

    if sr_csv is not None:
        reasons.append("Custom spectral response curve applied")

    if technology not in ("mono_si", "poly_si"):
        reasons.append(f"Non-standard technology ({technology}) — spectral mismatch computed")

    score = max(0, min(100, score))

    if score >= 85:
        label = "Very High"
    elif score >= 70:
        label = "High"
    elif score >= 50:
        label = "Medium"
    else:
        label = "Low"

    return {
        "confidence_pct":    score,
        "confidence_label":  label,
        "confidence_reasons": reasons,
    }
