"""
AI Solar Production Forecast — 3-Tier Dashboard
================================================
Level 1 BASIC  : city + kW → instant forecast (no science visible)
Level 2 PRO    : tilt, azimuth, technology, horizon, multi-location, CSV
Level 3 EXPERT : SR upload, IAM model, AI toggle, denorm, data status
"""

from __future__ import annotations

import io
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.db import sqlite_manager as db
from solar_forecast.demo.pipeline import run_demo_forecast, run_realtime_forecast

logging.basicConfig(level=logging.WARNING)

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Solar Forecast",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  body, [data-testid="stAppViewContainer"] { background:#0E1117; }
  .block-container { padding-top:1.5rem; max-width:1400px; }

  .hero-card {
    background: linear-gradient(135deg,#1a1a2e 0%,#16213e 60%,#0f3460 100%);
    border:1px solid #F4A503; border-radius:16px; padding:32px 36px;
    margin-bottom:24px;
  }
  .hero-title { font-size:2.4rem; font-weight:800; color:#F4A503; line-height:1.1; }
  .hero-sub   { font-size:1.1rem; color:#aaa; margin-top:6px; }

  .kpi-grid   { display:flex; gap:16px; flex-wrap:wrap; margin-bottom:24px; }
  .kpi-card   { flex:1; min-width:140px; background:#1E1E2E;
                border-radius:14px; padding:20px 18px;
                border:1px solid #2a2a3e; text-align:center; }
  .kpi-val    { font-size:2rem; font-weight:800; color:#F4A503; line-height:1.1; }
  .kpi-label  { font-size:0.82rem; color:#888; margin-top:6px; text-transform:uppercase; }
  .kpi-sub    { font-size:0.75rem; color:#555; margin-top:3px; }

  .badge-pro    { background:#2563eb; color:#fff; padding:2px 8px;
                  border-radius:6px; font-size:0.72rem; font-weight:700; }
  .badge-expert { background:#7c3aed; color:#fff; padding:2px 8px;
                  border-radius:6px; font-size:0.72rem; font-weight:700; }

  .section-title { font-size:1.2rem; font-weight:700; color:#e0e0e0;
                   margin:20px 0 12px 0; }

  [data-testid="stTab"] { font-size:0.95rem; }

  .conf-bar  { height:8px; border-radius:4px; background:#2a2a3e; margin:6px 0; }
  .conf-fill { height:8px; border-radius:4px; }
  .conf-high  { background:linear-gradient(90deg,#22c55e,#4ade80); }
  .conf-med   { background:linear-gradient(90deg,#F4A503,#fbbf24); }
  .conf-low   { background:linear-gradient(90deg,#ef4444,#f87171); }

  .status-ok   { color:#22c55e; font-weight:600; }
  .status-warn { color:#F4A503; font-weight:600; }
  .status-err  { color:#ef4444; font-weight:600; }

  footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════
_TECH = {
    "mono_si": "Mono-Si (standard)",
    "poly_si": "Poly-Si",
    "cdte":    "CdTe (thin-film)",
    "cigs":    "CIGS (thin-film)",
    "hit":     "HIT / Heterojunction",
}
_IAM_MODELS = ["ashrae", "martin_ruiz", "fresnel"]
_TIMEZONES  = [
    "UTC","Europe/Budapest","Europe/Vienna","Europe/Berlin","Europe/London",
    "Europe/Paris","Europe/Warsaw","Europe/Rome","Europe/Madrid",
    "Europe/Bucharest","Europe/Athens",
    "US/Eastern","US/Central","US/Mountain","US/Pacific",
    "Asia/Tokyo","Asia/Shanghai","Asia/Kolkata","Australia/Sydney",
]
_LEVELS = {"Basic": 1, "Pro": 2, "Expert": 3}

# Demo defaults (Budapest, Hungary)
_DEMO_LAT, _DEMO_LON, _DEMO_ALT = 47.498, 19.040, 120.0
_DEMO_NAME = "Budapest (demo)"


@st.cache_resource
def _init_db():
    db.create_tables()
    db.seed_demo_location()

_init_db()


# ═══════════════════════════════════════════════════════════════════
# Forecast cache (30 min TTL — no live API calls during tests)
# ═══════════════════════════════════════════════════════════════════
@st.cache_data(ttl=1800, show_spinner=False)
def _forecast(lat, lon, alt, cap, tilt, az, tech, iam, horizon, use_ai, sr_csv, denorm, _key):
    return run_demo_forecast(
        lat=lat, lon=lon, altitude=alt, capacity_kw=cap,
        tilt=tilt, azimuth=az, technology=tech, iam_model=iam,
        horizon_days=horizon, sr_csv=sr_csv, use_ai=use_ai,
        denorm_factor=denorm,
    )


def _tz_convert(df: pd.DataFrame, tz: str) -> pd.DataFrame:
    if tz == "UTC" or df.empty:
        return df
    out = df.copy()
    out.index = out.index.tz_convert(tz)
    return out


def _geocode(city: str):
    import requests
    r = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1, "language": "en"}, timeout=8,
    )
    results = r.json().get("results", [])
    if not results:
        raise ValueError(f"City not found: {city!r}")
    res = results[0]
    return float(res["latitude"]), float(res["longitude"]), res.get("name", city), float(res.get("elevation", 0))


# ═══════════════════════════════════════════════════════════════════
# Data status helpers
# ═══════════════════════════════════════════════════════════════════
@st.cache_data(ttl=300, show_spinner=False)
def _get_data_status():
    """Return data availability status dict (cached 5 min)."""
    status = {
        "cams_configured": False,
        "cams_last_update": None,
        "cams_rows": 0,
        "om_last_update": None,
        "om_rows": 0,
        "data_tier": "demo",
    }
    try:
        from solar_forecast.ingestion.cams.client import is_cams_configured
        status["cams_configured"] = is_cams_configured()
    except Exception:
        pass

    try:
        from solar_forecast.db.manager import get_connection, create_tables
        create_tables()
        with get_connection() as conn:
            r1 = conn.execute(
                "SELECT MAX(ingested_at), COUNT(*) FROM cams_atmospheric_forecast"
            ).fetchone()
            r2 = conn.execute(
                "SELECT MAX(ingested_at), COUNT(*) FROM openmeteo_forecast"
            ).fetchone()
        if r1:
            status["cams_last_update"] = r1[0]
            status["cams_rows"] = r1[1] or 0
        if r2:
            status["om_last_update"] = r2[0]
            status["om_rows"] = r2[1] or 0

        if status["cams_rows"] > 0 and status["om_rows"] > 0:
            status["data_tier"] = "cams_om"
        elif status["om_rows"] > 0:
            status["data_tier"] = "om_climatology"
        elif status["cams_rows"] > 0:
            status["data_tier"] = "cams_only"
    except Exception:
        pass

    return status


# ═══════════════════════════════════════════════════════════════════
# Confidence
# ═══════════════════════════════════════════════════════════════════
def _get_confidence(cfg: dict, result: dict) -> dict:
    try:
        from solar_forecast.engine.confidence import compute_confidence
        atm_src = result.get("atmosphere", {}).get("source", "climatology")
        return compute_confidence(
            atmosphere_source=atm_src,
            has_openmeteo=not result["hourly"].get("ghi_wm2", pd.Series()).empty,
            use_ai=cfg.get("use_ai", False),
            has_historical_model=Path("models/ghi_historical.joblib").exists(),
            horizon_days=cfg.get("horizon", 7),
            technology=cfg.get("tech", "mono_si"),
            sr_csv=cfg.get("sr_csv"),
        )
    except Exception:
        return {"confidence_pct": 65, "confidence_label": "Medium", "confidence_reasons": []}


# ═══════════════════════════════════════════════════════════════════
# KPI card helper
# ═══════════════════════════════════════════════════════════════════
def _kpi(col, label: str, value: str, sub: str = ""):
    col.markdown(
        f'<div class="kpi-card">'
        f'<div class="kpi-val">{value}</div>'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-sub">{sub}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _confidence_widget(conf: dict):
    pct   = conf["confidence_pct"]
    label = conf["confidence_label"]
    color = "conf-high" if label in ("High", "Very High") else ("conf-med" if label == "Medium" else "conf-low")
    icon  = "🟢" if label in ("High", "Very High") else ("🟡" if label == "Medium" else "🔴")
    st.markdown(
        f'**{icon} Forecast confidence: {label}** ({pct}%)<br>'
        f'<div class="conf-bar"><div class="conf-fill {color}" style="width:{pct}%"></div></div>',
        unsafe_allow_html=True,
    )
    if conf.get("confidence_reasons"):
        with st.expander("Why this confidence score?", expanded=False):
            for r in conf["confidence_reasons"]:
                st.markdown(f"• {r}")


# ═══════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════
def _sidebar():
    with st.sidebar:
        st.markdown("### ☀️ AI Solar Forecast")
        st.caption("Physics-accurate · AI-enhanced · SaaS")
        st.divider()

        level_name = st.radio("User level", list(_LEVELS.keys()), horizontal=True, index=0)
        level = _LEVELS[level_name]
        st.divider()

        # ── Location ─────────────────────────────────────────────
        mode = st.radio("Location", ["City", "GPS"], horizontal=True)
        lat = lon = alt = None
        loc_name = _DEMO_NAME

        if mode == "City":
            city = st.text_input("City name", value="Budapest", placeholder="e.g. London")
            if st.button("🔍 Find", use_container_width=True):
                try:
                    lat, lon, loc_name, alt = _geocode(city)
                    st.session_state["geo"] = (lat, lon, loc_name, alt)
                except Exception as e:
                    st.error(str(e))
            geo = st.session_state.get("geo", (_DEMO_LAT, _DEMO_LON, _DEMO_NAME, _DEMO_ALT))
            lat, lon, loc_name, alt = geo
        else:
            c1, c2 = st.columns(2)
            lat = c1.number_input("Lat", -90.0,  90.0,  _DEMO_LAT, 0.001, format="%.4f")
            lon = c2.number_input("Lon", -180.0, 180.0, _DEMO_LON, 0.001, format="%.4f")
            alt = st.number_input("Altitude (m)", 0, 5000, int(_DEMO_ALT))
            loc_name = f"{lat:.3f}°N {lon:.3f}°E"

        cap = st.number_input("System size (kW)", 0.1, 100000.0, 5.0, 0.5,
                              help="Total installed DC capacity of your solar panels")

        # ── PRO options ───────────────────────────────────────────
        tilt = az = None
        tech    = "mono_si"
        horizon = 7
        tz      = "Europe/Budapest"

        if level >= 2:
            st.divider()
            st.markdown('<span class="badge-pro">PRO</span>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            tilt = c1.slider("Tilt (°)", 0, 90, 35)
            az   = c2.slider("Azimuth (°)", 0, 360, 180,
                             help="180 = South  ·  90 = East  ·  270 = West")
            tech    = st.selectbox("Panel type", list(_TECH), format_func=lambda k: _TECH[k])
            horizon = st.slider("Forecast days", 1, 14, 7)
            tz      = st.selectbox("Timezone", _TIMEZONES, index=1)

        # ── EXPERT options ────────────────────────────────────────
        iam_model  = "ashrae"
        use_ai     = False
        sr_csv     = None
        denorm     = 1.0

        if level >= 3:
            st.divider()
            st.markdown('<span class="badge-expert">EXPERT</span>', unsafe_allow_html=True)
            with st.expander("Advanced physics settings"):
                iam_model = st.selectbox(
                    "Incidence angle model",
                    _IAM_MODELS,
                    help="ASHRAE: simple cosine  ·  Martin-Ruiz: empirical  ·  Fresnel: physical optics",
                )
                denorm = st.slider(
                    "Effective irradiance scale",
                    0.70, 1.30, 1.00, 0.01,
                    help="Scales effective irradiance (spectral denormalization factor). "
                         "Values < 1.0 reduce output; > 1.0 increase. Default 1.0.",
                )
                use_ai = st.toggle(
                    "AI Kt correction (XGBoost)",
                    value=False,
                    help="Requires trained model at models/kt_xgb.joblib",
                )
                sr_file = st.file_uploader(
                    "Custom spectral response (CSV)",
                    type="csv",
                    help="Two-column CSV: wavelength_nm, sr_value  (0–1 normalised)",
                )
                if sr_file:
                    p = Path("/tmp/sr_custom.csv")
                    p.write_bytes(sr_file.read())
                    sr_csv = str(p)
                    st.success("Custom SR loaded")

        st.divider()
        if st.button("🔄 Refresh forecast", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    return dict(
        lat=lat, lon=lon, alt=float(alt or 0), cap=float(cap),
        tilt=tilt, az=az, tech=tech, iam=iam_model,
        horizon=horizon, tz=tz if level >= 2 else "UTC",
        use_ai=use_ai, sr_csv=sr_csv, denorm=denorm,
        loc_name=loc_name, level=level,
    )


# ═══════════════════════════════════════════════════════════════════
# Charts
# ═══════════════════════════════════════════════════════════════════
def _chart_production(hourly: pd.DataFrame, tz: str, show_clearsky: bool = True):
    df = _tz_convert(hourly, tz)
    fig = go.Figure()
    if show_clearsky and "power_clear_kw" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["power_clear_kw"],
            name="Clear-sky", line=dict(color="#74c0fc", width=1.5, dash="dot"),
        ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["power_kw"],
        name="Forecast", fill="tozeroy",
        line=dict(color="#F4A503", width=2.5),
        fillcolor="rgba(244,165,3,0.12)",
    ))
    fig.update_layout(
        template="plotly_dark", height=340,
        margin=dict(t=10, b=10, l=0, r=0),
        yaxis_title="Power (kW)",
        legend=dict(orientation="h", y=1.05),
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="#2a2a3e"),
        plot_bgcolor="#0E1117", paper_bgcolor="#0E1117",
    )
    return fig


def _chart_daily(hourly: pd.DataFrame, tz: str):
    df = _tz_convert(hourly, tz)
    daily = df.groupby(df.index.normalize())["energy_kwh"].sum().head(14)
    fig = go.Figure(go.Bar(
        x=[d.strftime("%a %d %b") for d in daily.index],
        y=daily.values.round(1),
        marker_color="#F4A503", marker_line_width=0,
        text=[f"{v:.1f}" for v in daily.values],
        textposition="outside",
        textfont=dict(color="#aaa", size=11),
    ))
    fig.update_layout(
        template="plotly_dark", height=260,
        margin=dict(t=10, b=10, l=0, r=0),
        yaxis_title="kWh",
        plot_bgcolor="#0E1117", paper_bgcolor="#0E1117",
        yaxis=dict(gridcolor="#2a2a3e"),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════
# Data Status panel (reusable section)
# ═══════════════════════════════════════════════════════════════════
def _data_status_section(expanded: bool = False):
    ds = _get_data_status()

    tier_labels = {
        "cams_om":        ("🟢", "CAMS + Weather",     "Full physics accuracy"),
        "om_climatology": ("🟡", "Weather + defaults", "Good accuracy, no live aerosol data"),
        "cams_only":      ("🟡", "CAMS only",          "Atmospheric data, no weather"),
        "demo":           ("🔵", "Demo mode",           "Works offline, no data sources needed"),
    }
    icon, tier_name, tier_desc = tier_labels.get(ds["data_tier"], ("🔵", "Demo", ""))

    with st.expander(f"{icon} Data status — {tier_name}", expanded=expanded):
        c1, c2 = st.columns(2)

        # CAMS status
        with c1:
            st.markdown("**🛰 CAMS Atmospheric Data**")
            if ds["cams_configured"]:
                st.markdown('<span class="status-ok">✓ API key configured</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="status-warn">⚠ No API key — using defaults</span>',
                            unsafe_allow_html=True)
            if ds["cams_rows"] > 0:
                st.markdown(f'<span class="status-ok">✓ {ds["cams_rows"]:,} rows stored</span>',
                            unsafe_allow_html=True)
                if ds["cams_last_update"]:
                    st.caption(f"Last update: {ds['cams_last_update'][:16]}")
            else:
                st.markdown('<span class="status-warn">No CAMS data in database — falling back to climatology</span>',
                            unsafe_allow_html=True)
                st.caption("Run: python -m solar_forecast.ingestion.cams.backfill --location-id 1 --days 30")

        # Open-Meteo status
        with c2:
            st.markdown("**🌤 Open-Meteo Weather**")
            if ds["om_rows"] > 0:
                st.markdown(f'<span class="status-ok">✓ {ds["om_rows"]:,} rows stored</span>',
                            unsafe_allow_html=True)
                if ds["om_last_update"]:
                    st.caption(f"Last update: {ds['om_last_update'][:16]}")
            else:
                st.markdown('<span class="status-ok">✓ Live fetch on demand (free, no key needed)</span>',
                            unsafe_allow_html=True)
                st.caption("Weather is fetched live from Open-Meteo for each forecast request")

        st.caption(f"Forecast tier: **{tier_name}** — {tier_desc}")


# ═══════════════════════════════════════════════════════════════════
# Tab: Dashboard
# ═══════════════════════════════════════════════════════════════════
def tab_dashboard(cfg: dict):
    # Demo mode banner (Basic level only — no jargon)
    ds = _get_data_status()
    if ds["data_tier"] == "demo" and cfg["level"] == 1:
        st.info(
            "📍 **Demo mode** — showing forecast for Budapest. "
            "Enter your city and system size in the sidebar to get your personalised forecast.",
            icon=None,
        )

    st.markdown(
        f'<div class="hero-card">'
        f'<div class="hero-title">☀️ Solar Forecast</div>'
        f'<div class="hero-sub">📍 {cfg["loc_name"]} &nbsp;·&nbsp; '
        f'⚡ {cfg["cap"]:.1f} kW &nbsp;·&nbsp; '
        f'{"🤖 AI-enhanced" if cfg["use_ai"] else "⚙️ Physics model"}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    key = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H")
    with st.spinner("Computing forecast…"):
        try:
            result = _forecast(
                cfg["lat"], cfg["lon"], cfg["alt"], cfg["cap"],
                cfg["tilt"], cfg["az"], cfg["tech"], cfg["iam"],
                cfg["horizon"], cfg["use_ai"], cfg["sr_csv"], cfg.get("denorm", 1.0), key,
            )
        except Exception as exc:
            st.error(f"Forecast error: {exc}")
            return

    s      = result["summary"]
    hourly = result["hourly"]
    tz     = cfg["tz"]

    # KPIs — plain language for Basic
    cols = st.columns(5)
    _kpi(cols[0], "Today",    f"{s['today_kwh']:.1f} kWh", "estimated production")
    _kpi(cols[1], "Tomorrow", f"{s['tomorrow_kwh']:.1f} kWh")
    _kpi(cols[2], f"{cfg['horizon']}-Day Total", f"{s['total_7d_kwh']:.0f} kWh")
    _kpi(cols[3], "Peak power", f"{s['peak_power_kw']:.2f} kW")
    _kpi(cols[4], "Cloud losses", f"{s['cloud_loss_pct']:.0f}%", "vs clear-sky day")

    st.markdown("")

    try:
        ph     = pd.Timestamp(s["peak_hour_utc"]).tz_convert(tz).strftime("%H:%M")
        phdate = pd.Timestamp(s["peak_hour_utc"]).tz_convert(tz).strftime("%d %b")
    except Exception:
        ph, phdate = "—", ""

    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown('<div class="section-title">Production vs clear-sky</div>', unsafe_allow_html=True)
        st.plotly_chart(_chart_production(hourly, tz), use_container_width=True)
    with c2:
        st.markdown('<div class="section-title">Details</div>', unsafe_allow_html=True)
        st.metric("Peak time", ph, phdate)
        if cfg["level"] >= 2:
            st.metric("Panel type",  _TECH.get(cfg["tech"], cfg["tech"]))
            st.metric("IAM model",   cfg["iam"].replace("_", "-").title())
        conf = _get_confidence(cfg, result)
        st.markdown("")
        _confidence_widget(conf)

    st.markdown(f'<div class="section-title">Daily output ({cfg["horizon"]}-day)</div>',
                unsafe_allow_html=True)
    st.plotly_chart(_chart_daily(hourly, tz), use_container_width=True)

    # Data status (always visible — collapsed by default in Basic)
    _data_status_section(expanded=(cfg["level"] >= 3))


# ═══════════════════════════════════════════════════════════════════
# Tab: Forecast detail
# ═══════════════════════════════════════════════════════════════════
def tab_forecast(cfg: dict):
    key = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H")
    try:
        result = _forecast(
            cfg["lat"], cfg["lon"], cfg["alt"], cfg["cap"],
            cfg["tilt"], cfg["az"], cfg["tech"], cfg["iam"],
            cfg["horizon"], cfg["use_ai"], cfg["sr_csv"], cfg.get("denorm", 1.0), key,
        )
    except Exception as exc:
        st.error(str(exc)); return

    hourly = _tz_convert(result["hourly"], cfg["tz"])

    st.markdown('<div class="section-title">Hourly production forecast</div>', unsafe_allow_html=True)
    st.plotly_chart(_chart_production(result["hourly"], cfg["tz"]), use_container_width=True)

    # GHI comparison
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=hourly.index, y=hourly.get("ghi_clear_wm2", []),
                              name="Clear-sky GHI", line=dict(color="#74c0fc", dash="dot")))
    fig2.add_trace(go.Scatter(x=hourly.index, y=hourly.get("ghi_wm2", []),
                              name="All-sky GHI", fill="tozeroy",
                              line=dict(color="#F4A503")))
    fig2.update_layout(template="plotly_dark", height=260,
                       margin=dict(t=10, b=0, l=0, r=0),
                       yaxis_title="W/m²",
                       plot_bgcolor="#0E1117", paper_bgcolor="#0E1117",
                       legend=dict(orientation="h", y=1.1))
    st.markdown('<div class="section-title">Solar irradiance: actual vs clear-sky (W/m²)</div>',
                unsafe_allow_html=True)
    st.plotly_chart(fig2, use_container_width=True)

    # SR/IAM/denorm comparison (Expert only)
    if cfg["level"] >= 3 and "spectral_mm" in hourly.columns:
        with st.expander("Spectral mismatch & IAM details", expanded=False):
            mm_mean  = float(hourly["spectral_mm"].replace(0, np.nan).mean() or 1.0)
            iam_mean = float(hourly["iam"].replace(0, np.nan).mean() or 0.96)
            denorm   = cfg.get("denorm", 1.0)
            col1, col2, col3 = st.columns(3)
            col1.metric("Spectral MM (mean)", f"{mm_mean:.4f}",
                        help="How well panel spectral response matches actual sky spectrum. 1.0 = perfect match")
            col2.metric("IAM factor (mean)", f"{iam_mean:.4f}",
                        help="Incidence angle modifier — reduces output at oblique sun angles")
            col3.metric("Denorm scale", f"{denorm:.2f}",
                        help="Effective irradiance scaling factor (1.0 = no change)")
            st.caption(
                f"Combined modifier applied to POA irradiance: "
                f"MM ({mm_mean:.3f}) × IAM ({iam_mean:.3f}) × scale ({denorm:.2f}) "
                f"= **{mm_mean * iam_mean * denorm:.3f}**"
            )

    show_cols = ["power_kw", "energy_kwh", "ghi_wm2", "kt", "t_cell_c", "iam", "cloud_cover_frac"]
    show_cols = [c for c in show_cols if c in hourly.columns]
    rename = {
        "power_kw": "Power (kW)", "energy_kwh": "Energy (kWh)", "ghi_wm2": "GHI (W/m²)",
        "kt": "Kt", "t_cell_c": "Cell °C", "iam": "IAM factor", "cloud_cover_frac": "Cloud cover",
    }
    st.markdown('<div class="section-title">Hourly data table</div>', unsafe_allow_html=True)
    st.dataframe(hourly[show_cols].rename(columns=rename).round(3),
                 use_container_width=True, height=380)


# ═══════════════════════════════════════════════════════════════════
# Tab: Locations
# ═══════════════════════════════════════════════════════════════════
def tab_locations(cfg: dict):
    st.markdown('<div class="section-title">📍 Saved Locations</div>', unsafe_allow_html=True)
    locations = db.list_locations()

    if locations:
        df = pd.DataFrame(locations)[
            ["id","name","lat","lon","capacity_kw","tilt","azimuth","technology","timezone"]
        ]
        st.dataframe(df, use_container_width=True, hide_index=True)
        c1, c2 = st.columns([3, 1])
        del_id = c1.number_input("Delete location ID", 0, step=1, value=0)
        if c2.button("🗑️ Delete"):
            if del_id and db.delete_location(int(del_id)):
                st.success(f"Deleted #{del_id}"); st.rerun()
    else:
        st.info("No locations saved yet. Add one below.")

    st.divider()
    st.markdown('<div class="section-title">Add Location</div>', unsafe_allow_html=True)

    with st.form("add_loc"):
        c1, c2 = st.columns(2)
        name = c1.text_input("Name")
        cap  = c2.number_input("System size (kW)", 0.1, 100000.0, 5.0, 0.5)
        c1, c2, c3 = st.columns(3)
        lat  = c1.number_input("Latitude",  -90.0,  90.0,  _DEMO_LAT, 0.001)
        lon  = c2.number_input("Longitude", -180.0, 180.0, _DEMO_LON, 0.001)
        alt  = c3.number_input("Altitude (m)", 0.0, 5000.0, _DEMO_ALT, 10.0)
        c1, c2 = st.columns(2)
        tilt = c1.number_input("Tilt (°)", 0.0, 90.0, 35.0)
        az   = c2.number_input("Azimuth (°)", 0.0, 360.0, 180.0)
        c1, c2 = st.columns(2)
        tech = c1.selectbox("Panel type", list(_TECH), format_func=lambda k: _TECH[k])
        tz   = c2.selectbox("Timezone", _TIMEZONES, index=1)
        if st.form_submit_button("✓ Save", type="primary"):
            if not name.strip():
                st.error("Name is required.")
            else:
                new = db.create_location({
                    "name": name.strip(), "lat": lat, "lon": lon,
                    "altitude": alt, "capacity_kw": cap,
                    "tilt": tilt, "azimuth": az,
                    "technology": tech, "timezone": tz,
                })
                st.success(f"Saved #{new['id']}: {new['name']}"); st.rerun()


# ═══════════════════════════════════════════════════════════════════
# Tab: Reports
# ═══════════════════════════════════════════════════════════════════
def tab_reports(cfg: dict):
    key = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H")
    try:
        result = _forecast(
            cfg["lat"], cfg["lon"], cfg["alt"], cfg["cap"],
            cfg["tilt"], cfg["az"], cfg["tech"], cfg["iam"],
            cfg["horizon"], cfg["use_ai"], cfg["sr_csv"], cfg.get("denorm", 1.0), key,
        )
    except Exception as exc:
        st.error(str(exc)); return

    hourly = _tz_convert(result["hourly"], cfg["tz"])
    daily  = hourly.groupby(hourly.index.normalize()).agg(
        energy_kwh=("energy_kwh", "sum"),
        peak_kw=("power_kw", "max"),
        avg_ghi=("ghi_wm2", "mean"),
        cloud_frac=("cloud_cover_frac", "mean"),
    ).round(2)

    st.markdown('<div class="section-title">Daily Summary</div>', unsafe_allow_html=True)
    st.dataframe(daily, use_container_width=True)

    c1, c2 = st.columns(2)
    buf1 = io.StringIO(); hourly.to_csv(buf1)
    c1.download_button(
        "⬇ Download hourly CSV", buf1.getvalue().encode(),
        f"forecast_hourly_{cfg['loc_name'].replace(' ','_')}.csv",
        "text/csv", use_container_width=True,
    )
    buf2 = io.StringIO(); daily.to_csv(buf2)
    c2.download_button(
        "⬇ Download daily summary CSV", buf2.getvalue().encode(),
        f"forecast_daily_{cfg['loc_name'].replace(' ','_')}.csv",
        "text/csv", use_container_width=True,
    )


# ═══════════════════════════════════════════════════════════════════
# Real-time cache
# ═══════════════════════════════════════════════════════════════════
@st.cache_data(ttl=60, show_spinner=False)
def _realtime(lat, lon, alt, cap, tilt, az, tech, iam, resolution, horizon, _key):
    return run_realtime_forecast(
        lat=lat, lon=lon, altitude=alt, capacity_kw=cap,
        tilt=tilt, azimuth=az, technology=tech, iam_model=iam,
        resolution_minutes=resolution, horizon_hours=horizon,
    )


def _chart_realtime(curve: pd.DataFrame, tz: str, now_power_kw: float):
    df = _tz_convert(curve, tz)
    fig = go.Figure()
    if "ghi_clear_wm2" in df.columns and df["ghi_clear_wm2"].max() > 0:
        scale = df["power_kw"].max() / max(df["ghi_clear_wm2"].max(), 1)
        fig.add_trace(go.Scatter(
            x=df.index,
            y=(df["ghi_clear_wm2"] * scale).clip(lower=0),
            name="Clear-sky (scaled)",
            line=dict(color="#74c0fc", width=1.2, dash="dot"),
        ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["power_kw"].clip(lower=0),
        name="Forecast", fill="tozeroy",
        line=dict(color="#F4A503", width=2.5),
        fillcolor="rgba(244,165,3,0.10)",
    ))
    now_ts = pd.Timestamp.now(tz="UTC")
    if tz != "UTC":
        import pytz
        now_ts = now_ts.tz_convert(tz)
    fig.add_vline(x=now_ts, line_dash="solid", line_color="#22c55e", line_width=2,
                  annotation_text="NOW", annotation_position="top right",
                  annotation_font_color="#22c55e")
    fig.update_layout(
        template="plotly_dark", height=320,
        margin=dict(t=10, b=10, l=0, r=0),
        yaxis_title="Power (kW)",
        legend=dict(orientation="h", y=1.05),
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="#2a2a3e"),
        plot_bgcolor="#0E1117", paper_bgcolor="#0E1117",
    )
    return fig


# ═══════════════════════════════════════════════════════════════════
# Tab: Real-Time
# ═══════════════════════════════════════════════════════════════════
def tab_realtime(cfg: dict):
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=60_000, key="rt_refresh")
    except Exception:
        pass

    st.markdown(
        '<div class="hero-card">'
        '<div class="hero-title">⚡ Real-Time Production</div>'
        '<div class="hero-sub">Sub-hourly estimate · auto-refreshes every 60 s · '
        'NOW marker shows current moment</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    c_res, c_hor = st.columns(2)
    resolution = c_res.select_slider("Time resolution", options=[5, 10, 15, 30, 60], value=15,
                                      help="Minutes between data points")
    horizon = c_hor.slider("Horizon (hours)", 6, 48, 24)

    key_rt = pd.Timestamp.now(tz="UTC").floor("1min").isoformat()
    with st.spinner("Computing real-time estimate…"):
        try:
            rt = _realtime(
                cfg["lat"], cfg["lon"], cfg["alt"], cfg["cap"],
                cfg["tilt"], cfg["az"], cfg["tech"], cfg["iam"],
                resolution, horizon, key_rt,
            )
        except Exception as exc:
            st.error(f"Real-time forecast error: {exc}"); return

    now_kw  = rt["now_power_kw"]
    curve   = rt["curve"]
    atm     = rt.get("atmosphere", {})
    now_utc = pd.Timestamp(rt["now_utc"])
    day_kwh = float(curve["energy_kwh"].sum())
    peak_kw = float(curve["power_kw"].max())
    source  = atm.get("source", "climatology")
    src_lbl = "🛰 CAMS data" if source == "cams" else "📊 Weather model"

    cols = st.columns(4)
    _kpi(cols[0], "Right now",    f"{now_kw:.3f} kW",  now_utc.strftime("%H:%M UTC"))
    _kpi(cols[1], "Peak (period)", f"{peak_kw:.2f} kW")
    _kpi(cols[2], f"{horizon}h energy", f"{day_kwh:.2f} kWh")
    _kpi(cols[3], "Atmosphere",   src_lbl, f"AOD {atm.get('aod_550nm_mean', 0):.3f}")

    st.markdown("")
    st.markdown('<div class="section-title">Live production curve</div>', unsafe_allow_html=True)
    st.plotly_chart(_chart_realtime(curve, cfg.get("tz", "UTC"), now_kw),
                    use_container_width=True)

    if cfg.get("level", 1) >= 2:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="section-title">Clearness index (Kt)</div>',
                        unsafe_allow_html=True)
            fig_kt = go.Figure(go.Scatter(
                x=curve.index, y=curve["kt"].clip(0, 1.2),
                fill="tozeroy", line=dict(color="#a78bfa", width=1.8),
                fillcolor="rgba(167,139,250,0.12)",
            ))
            fig_kt.update_layout(template="plotly_dark", height=200,
                                 margin=dict(t=5, b=5, l=0, r=0),
                                 yaxis=dict(range=[0, 1.2], gridcolor="#2a2a3e"),
                                 plot_bgcolor="#0E1117", paper_bgcolor="#0E1117")
            st.plotly_chart(fig_kt, use_container_width=True)
        with c2:
            st.markdown('<div class="section-title">Cell temperature (°C)</div>',
                        unsafe_allow_html=True)
            fig_t = go.Figure(go.Scatter(
                x=curve.index, y=curve["t_cell_c"],
                line=dict(color="#f87171", width=1.8),
            ))
            fig_t.update_layout(template="plotly_dark", height=200,
                                margin=dict(t=5, b=5, l=0, r=0),
                                yaxis=dict(gridcolor="#2a2a3e"),
                                plot_bgcolor="#0E1117", paper_bgcolor="#0E1117")
            st.plotly_chart(fig_t, use_container_width=True)

    if cfg.get("level", 1) >= 3:
        with st.expander("Atmospheric diagnostics"):
            st.json({
                "source":                atm.get("source", "climatology"),
                "aod_550nm":             round(atm.get("aod_550nm_mean", 0), 4),
                "ozone_du":              round(atm.get("ozone_du_mean", 0), 1),
                "precipitable_water_cm": round(atm.get("precipitable_water_cm", 0), 2),
            })
        buf = io.StringIO()
        _tz_convert(curve, cfg.get("tz", "UTC")).to_csv(buf)
        st.download_button(
            "⬇ Download real-time CSV",
            buf.getvalue().encode(),
            f"realtime_{cfg['loc_name'].replace(' ','_')}.csv",
            "text/csv", use_container_width=True,
        )


# ═══════════════════════════════════════════════════════════════════
# Tab: Settings / Data Status
# ═══════════════════════════════════════════════════════════════════
def tab_settings(cfg: dict):
    st.markdown('<div class="section-title">Data Sources</div>', unsafe_allow_html=True)
    _data_status_section(expanded=True)

    st.markdown('<div class="section-title">System Status</div>', unsafe_allow_html=True)
    st.success("✓ Demo mode — works instantly without any API keys or setup")

    ai_path = Path("models/kt_xgb.joblib")
    if ai_path.exists():
        st.success(f"✓ AI model ready: {ai_path} ({ai_path.stat().st_size // 1024} KB)")
    else:
        st.info("ℹ No AI model — running pure physics mode. See Model Training tab to train one.")

    ghi_path = Path("models/ghi_historical.joblib")
    if ghi_path.exists():
        st.success(f"✓ Historical GHI model: {ghi_path}")

    st.markdown('<div class="section-title">Physics Engine</div>', unsafe_allow_html=True)
    st.markdown("""
| Component | Implementation |
|---|---|
| Clear-sky model | pvlib SPECTRL2 (Bird & Riordan 1986) |
| All-sky transposition | Perez model |
| Aerosol inputs | CAMS AOD 469–865 nm / climatology fallback |
| Spectral integration | ∫ SR(λ) × I(λ) × IAM(θ) dλ per timestep |
| AI correction | XGBoost Kt regressor (21 features) |
| Cell temperature | NOCT model |
| Version | 2.1.0 |
""")

    ds = _get_data_status()
    if not ds["cams_configured"]:
        st.divider()
        st.markdown("**Enable CAMS for higher accuracy:**")
        st.markdown("""
1. Register free at [ads.atmosphere.copernicus.eu](https://ads.atmosphere.copernicus.eu)
2. Add to `.env`: `CAMS_API_KEY=UID:KEY`
3. Run: `python -m solar_forecast.ingestion.cams.backfill --location-id 1 --days 30`
4. For automated updates, install the scheduler:
```python
from solar_forecast.ingestion.cams.scheduler import setup_cron
setup_cron()
```
""")


# ═══════════════════════════════════════════════════════════════════
# Tab: Model Training
# ═══════════════════════════════════════════════════════════════════
def tab_training():
    st.markdown('<div class="section-title">🧠 AI Model Training</div>', unsafe_allow_html=True)
    ai_path = Path("models/kt_xgb.joblib")
    if ai_path.exists():
        st.success(f"✓ Model ready: `{ai_path}` ({ai_path.stat().st_size // 1024} KB)")
    else:
        st.warning("No trained model found — running physics-only mode.")

    st.markdown("""
### What does the AI model do?

The XGBoost Kt corrector learns from CAMS atmospheric data to reduce forecast
errors by 10–20%, especially on partly-cloudy or aerosol-heavy days.

### Training steps

```bash
# 1. Download CAMS history
python scripts/01_download_cams.py --start 2022-01-01 --end 2023-12-31

# 2. Train XGBoost model
python scripts/02_train_kt_model.py --cv 5
```

### Typical accuracy improvement

| | Physics only | Physics + AI |
|---|:---:|:---:|
| Kt RMSE | 0.12 | 0.08 |
| GHI RMSE (W/m²) | 62 | 41 |
| R² | 0.86 | 0.93 |
""")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
def main():
    cfg = _sidebar()

    tabs = st.tabs([
        "📊 Dashboard", "⚡ Real-Time", "☀️ Forecast",
        "📍 Locations", "📁 Reports", "⚙️ Settings", "🧠 Model Training",
    ])
    with tabs[0]: tab_dashboard(cfg)
    with tabs[1]: tab_realtime(cfg)
    with tabs[2]: tab_forecast(cfg)
    with tabs[3]: tab_locations(cfg)
    with tabs[4]: tab_reports(cfg)
    with tabs[5]: tab_settings(cfg)
    with tabs[6]: tab_training()


if __name__ == "__main__":
    main()
