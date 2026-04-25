"""
Financial performance calculator for PV systems.
Computes daily/weekly/monthly/annual revenue, payback period, CO2 savings.

Revenue model
-------------
Each kWh of generated energy is split between:
  - Self-consumption (``self_consumption_ratio``): valued at the retail
    electricity price, because it displaces grid purchases.
  - Export to grid (``1 - self_consumption_ratio``): valued at the
    feed-in tariff, which is typically lower than the retail price.

Degradation model
-----------------
Module output declines by ``DEGRADATION_RATE`` (0.5 %/year) each year
from year 1.  The first-year yield is taken as the supplied ``annual_kwh``.
Years 2-25 are computed as::

    kwh_year_n = annual_kwh × (1 - DEGRADATION_RATE) ** (n - 1)

NPV calculation
---------------
Cash flows are discounted at 3 % per annum.  The initial investment
(``system_cost_eur``) is treated as a negative cash flow at t = 0.

IRR
---
Computed via bisection on the NPV function over the range [−50 %, +200 %].
Returns ``None`` if no root is found within 200 iterations.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants (exported for use by other modules)
# ---------------------------------------------------------------------------

CO2_FACTOR_KG_PER_KWH: float = 0.4  # European grid average (kg CO₂ / kWh)
SYSTEM_LIFETIME_YEARS: int = 25
DEGRADATION_RATE: float = 0.005  # 0.5 % per year

_DISCOUNT_RATE: float = 0.03  # NPV discount rate
_TREE_CO2_KG_PER_YEAR: float = 22.0  # kg CO₂ absorbed by one tree per year


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def calculate_revenue(
    energy_df: pd.DataFrame,
    electricity_price_eur: float,
    feedin_tariff_eur: float,
    self_consumption_ratio: float = 0.3,
) -> dict[str, Any]:
    """
    Compute revenue over multiple time windows from a metered energy DataFrame.

    Self-consumed electricity is valued at the retail ``electricity_price_eur``
    because it offsets grid purchases.  Exported electricity is valued at the
    lower ``feedin_tariff_eur``.

    Parameters
    ----------
    energy_df:
        DataFrame with at least an ``energy_kwh`` column and a UTC-aware
        :class:`~pandas.DatetimeIndex`.  Rows typically represent hourly or
        daily measurements.
    electricity_price_eur:
        Retail electricity price in EUR/kWh (e.g. 0.25).
    feedin_tariff_eur:
        Feed-in tariff in EUR/kWh (e.g. 0.08).
    self_consumption_ratio:
        Fraction of generated energy that is self-consumed on-site [0, 1].
        Default is 0.3 (30 %).

    Returns
    -------
    dict
        ``today_eur``, ``week_eur``, ``month_eur``, ``year_eur`` – revenue in
        EUR for each window; ``blended_rate_eur_kwh`` – effective revenue per
        kWh; ``self_consumption_ratio`` – echoed back for the caller.
        Returns zeroes on any error.
    """
    _zero: dict[str, Any] = {
        "today_eur": 0.0,
        "week_eur": 0.0,
        "month_eur": 0.0,
        "year_eur": 0.0,
        "blended_rate_eur_kwh": 0.0,
        "self_consumption_ratio": self_consumption_ratio,
    }
    try:
        if energy_df is None or energy_df.empty or "energy_kwh" not in energy_df.columns:
            log.warning("calculate_revenue: energy_df is empty or missing 'energy_kwh'.")
            return _zero

        df = energy_df.copy()

        # Ensure UTC-aware DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        elif df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

        df = df.sort_index()
        now = pd.Timestamp.now(tz="UTC")

        blended = _blended_rate(electricity_price_eur, feedin_tariff_eur, self_consumption_ratio)

        def _sum_window(start: pd.Timestamp) -> float:
            mask = df.index >= start
            kwh = df.loc[mask, "energy_kwh"].sum()
            return round(float(kwh) * blended, 4)

        today_start = now.normalize()  # midnight UTC
        week_start = today_start - pd.Timedelta(days=6)
        month_start = today_start - pd.Timedelta(days=29)
        year_start = today_start - pd.Timedelta(days=364)

        return {
            "today_eur": _sum_window(today_start),
            "week_eur": _sum_window(week_start),
            "month_eur": _sum_window(month_start),
            "year_eur": _sum_window(year_start),
            "blended_rate_eur_kwh": round(blended, 4),
            "self_consumption_ratio": self_consumption_ratio,
        }
    except Exception as exc:
        log.error("calculate_revenue failed: %s", exc)
        return _zero


def calculate_roi(
    capacity_kw: float,
    annual_kwh: float,
    electricity_price_eur: float,
    feedin_tariff_eur: float,
    system_cost_eur: float,
    self_consumption_ratio: float = 0.3,
) -> dict[str, Any]:
    """
    Compute key return-on-investment metrics for a PV system over 25 years.

    Includes module degradation, NPV at 3 % discount rate, and IRR.

    Parameters
    ----------
    capacity_kw:
        DC nameplate capacity of the system (kW-peak).
    annual_kwh:
        First-year energy yield (kWh).
    electricity_price_eur:
        Retail electricity price (EUR/kWh).
    feedin_tariff_eur:
        Feed-in tariff (EUR/kWh).
    system_cost_eur:
        Total installed cost of the system (EUR, positive number).
    self_consumption_ratio:
        Fraction consumed on-site [0, 1].  Default 0.3.

    Returns
    -------
    dict
        ``annual_revenue_eur`` – year-1 revenue,
        ``payback_years`` – simple payback period,
        ``lifetime_revenue_eur`` – undiscounted 25-year revenue,
        ``npv_eur`` – net present value at 3 % discount,
        ``irr_pct`` – internal rate of return in percent (or ``None``),
        ``specific_yield_kwh_kwp`` – kWh per kWp,
        ``capacity_kw`` – echoed back.
    """
    _zero: dict[str, Any] = {
        "annual_revenue_eur": 0.0,
        "payback_years": None,
        "lifetime_revenue_eur": 0.0,
        "npv_eur": 0.0,
        "irr_pct": None,
        "specific_yield_kwh_kwp": 0.0,
        "capacity_kw": capacity_kw,
    }
    try:
        blended = _blended_rate(electricity_price_eur, feedin_tariff_eur, self_consumption_ratio)
        cashflows = _annual_cashflows(annual_kwh, blended)

        annual_rev = cashflows[0]
        lifetime_rev = sum(cashflows)
        payback = _simple_payback(system_cost_eur, cashflows)

        # NPV: year-0 cash flow is −system_cost; years 1-25 are revenues
        npv = -system_cost_eur + sum(
            cf / (1.0 + _DISCOUNT_RATE) ** yr
            for yr, cf in enumerate(cashflows, start=1)
        )

        # IRR
        all_cfs = [-system_cost_eur] + cashflows
        irr = _compute_irr(all_cfs)

        specific_yield = (annual_kwh / capacity_kw) if capacity_kw > 0 else 0.0

        return {
            "annual_revenue_eur": round(annual_rev, 2),
            "payback_years": round(payback, 1) if payback is not None else None,
            "lifetime_revenue_eur": round(lifetime_rev, 2),
            "npv_eur": round(npv, 2),
            "irr_pct": round(irr * 100.0, 2) if irr is not None else None,
            "specific_yield_kwh_kwp": round(specific_yield, 1),
            "capacity_kw": capacity_kw,
        }
    except Exception as exc:
        log.error("calculate_roi failed: %s", exc)
        return _zero


def calculate_co2_savings(annual_kwh: float) -> dict[str, Any]:
    """
    Estimate CO₂ emissions avoided by a PV system over its lifetime.

    Uses the European grid average emission factor of
    :data:`CO2_FACTOR_KG_PER_KWH` = 0.4 kg CO₂/kWh.

    Parameters
    ----------
    annual_kwh:
        Annual energy yield (kWh) in the first year.

    Returns
    -------
    dict
        ``annual_kg`` – kg CO₂ avoided in year 1,
        ``lifetime_kg`` – total kg CO₂ avoided over 25 years (with
        degradation),
        ``trees_equivalent`` – equivalent number of trees planted for one year,
        based on 22 kg CO₂/tree/year.
    """
    try:
        annual_kg = float(annual_kwh) * CO2_FACTOR_KG_PER_KWH
        # Lifetime savings account for annual yield degradation
        lifetime_kwh = sum(
            annual_kwh * (1.0 - DEGRADATION_RATE) ** yr
            for yr in range(SYSTEM_LIFETIME_YEARS)
        )
        lifetime_kg = lifetime_kwh * CO2_FACTOR_KG_PER_KWH
        trees = lifetime_kg / _TREE_CO2_KG_PER_YEAR

        return {
            "annual_kg": round(annual_kg, 1),
            "lifetime_kg": round(lifetime_kg, 1),
            "trees_equivalent": round(trees, 0),
        }
    except Exception as exc:
        log.error("calculate_co2_savings failed: %s", exc)
        return {"annual_kg": 0.0, "lifetime_kg": 0.0, "trees_equivalent": 0.0}


def lifetime_cashflows(
    capacity_kw: float,
    annual_kwh: float,
    electricity_price_eur: float,
    feedin_tariff_eur: float,
    system_cost_eur: float,
    self_consumption_ratio: float = 0.3,
) -> pd.DataFrame:
    """
    Build a year-by-year cashflow table for the full 25-year system lifetime.

    Intended as the data source for the lifetime earnings chart in the
    dashboard and HTML report.

    Parameters
    ----------
    capacity_kw:
        DC nameplate capacity (kW-peak).
    annual_kwh:
        First-year energy yield (kWh).
    electricity_price_eur:
        Retail electricity price (EUR/kWh).
    feedin_tariff_eur:
        Feed-in tariff (EUR/kWh).
    system_cost_eur:
        Total installed system cost (EUR).
    self_consumption_ratio:
        Fraction consumed on-site [0, 1].

    Returns
    -------
    pd.DataFrame
        Columns: ``year`` (1-25), ``energy_kwh``, ``revenue_eur``,
        ``cumulative_eur`` (cumulative net revenue after subtracting
        ``system_cost_eur`` in year 0), ``degradation_factor``.
        Returns an empty DataFrame on error.
    """
    try:
        blended = _blended_rate(electricity_price_eur, feedin_tariff_eur, self_consumption_ratio)
        rows = []
        cumulative = -system_cost_eur  # start after investment
        for yr in range(1, SYSTEM_LIFETIME_YEARS + 1):
            deg_factor = (1.0 - DEGRADATION_RATE) ** (yr - 1)
            kwh = annual_kwh * deg_factor
            revenue = kwh * blended
            cumulative += revenue
            rows.append(
                {
                    "year": yr,
                    "energy_kwh": round(kwh, 1),
                    "revenue_eur": round(revenue, 2),
                    "cumulative_eur": round(cumulative, 2),
                    "degradation_factor": round(deg_factor, 6),
                }
            )
        return pd.DataFrame(rows)
    except Exception as exc:
        log.error("lifetime_cashflows failed: %s", exc)
        return pd.DataFrame(
            columns=["year", "energy_kwh", "revenue_eur", "cumulative_eur", "degradation_factor"]
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _blended_rate(
    electricity_price_eur: float,
    feedin_tariff_eur: float,
    self_consumption_ratio: float,
) -> float:
    """Return the blended EUR/kWh revenue rate."""
    sc = max(0.0, min(1.0, self_consumption_ratio))
    return sc * electricity_price_eur + (1.0 - sc) * feedin_tariff_eur


def _annual_cashflows(annual_kwh: float, blended_rate: float) -> list[float]:
    """Return a list of annual revenue cash flows for years 1-25."""
    return [
        annual_kwh * (1.0 - DEGRADATION_RATE) ** (yr - 1) * blended_rate
        for yr in range(1, SYSTEM_LIFETIME_YEARS + 1)
    ]


def _simple_payback(system_cost_eur: float, cashflows: list[float]) -> float | None:
    """
    Return the simple payback period in fractional years.

    Iterates through annual cash flows until cumulative revenue exceeds the
    system cost.  Returns ``None`` if payback is not achieved within the
    lifetime.
    """
    cumulative = 0.0
    for yr, cf in enumerate(cashflows, start=1):
        cumulative += cf
        if cumulative >= system_cost_eur:
            # Linear interpolation within the payback year
            overshoot = cumulative - system_cost_eur
            return float(yr) - overshoot / cf
    return None


def _compute_irr(cashflows: list[float], max_iter: int = 200) -> float | None:
    """
    Estimate the internal rate of return via bisection.

    Parameters
    ----------
    cashflows:
        List of cash flows starting at t = 0.  Typically the first entry is a
        negative investment cost.
    max_iter:
        Maximum bisection iterations.

    Returns
    -------
    float or None
        IRR as a decimal fraction (e.g. 0.12 for 12 %).  Returns ``None`` if
        no root is found.
    """

    def _npv(rate: float) -> float:
        return sum(cf / (1.0 + rate) ** t for t, cf in enumerate(cashflows))

    lo, hi = -0.5, 2.0
    npv_lo = _npv(lo)
    npv_hi = _npv(hi)

    # Require a sign change for bisection to work
    if npv_lo * npv_hi > 0:
        return None

    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        npv_mid = _npv(mid)
        if abs(npv_mid) < 1e-4 or (hi - lo) < 1e-8:
            return mid
        if npv_lo * npv_mid < 0:
            hi = mid
            npv_hi = npv_mid
        else:
            lo = mid
            npv_lo = npv_mid

    return (lo + hi) / 2.0
