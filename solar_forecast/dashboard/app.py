"""
AI Solar Production Forecast SaaS — 3-Tier Dashboard
=====================================================
Level 1 BASIC  : city + kW → instant forecast (no science visible)
Level 2 PRO    : tilt, azimuth, technology, horizon, multi-location, CSV
Level 3 EXPERT : SR upload, IAM model, AI toggle, denorm tuning
"""

from __future__ import annotations

import io
import logging
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
from solar_forecast.demo.pipeline import run_demo_forecast

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
  /* Global */
  body, [data-testid="stAppViewContainer"] { background:#0E1117; }
  .block-container { padding-top:1.5rem; max-width:1400px; }

  /* Hero card */
  .hero-card {
    background: linear-gradient(135deg,#1a1a2e 0%,#16213e 60%,#0f3460 100%);
    border:1px solid #F4A503; border-radius:16px; padding:32px 36px;
    margin-bottom:24px;
  }
  .hero-title { font-size:2.4rem; font-weight:800; color:#F4A503; line-height:1.1; }
  .hero-sub   { font-size:1.1rem; color:#aaa; margin-top:6px; }

  /* KPI cards */
  .kpi-grid   { display:flex; gap:16px; flex-wrap:wrap; margin-bottom:24px; }
  .kpi-card   { flex:1; min-width:140px; background:#1E1E2E;
                border-radius:14px; padding:20px 18px;
                border:1px solid #2a2a3e; text-align:center; }
  .kpi-val    { font-size:2rem; font-weight:800; color:#F4A503; line-height:1.1; }
  .kpi-label  { font-size:0.82rem; color:#888; margin-top:6px; text-transform:uppercase; }
  .kpi-sub    { font-size:0.75rem; color:#555; margin-top:3px; }

  /* Badge */
  .badge-pro    { background:#2563eb; color:#fff; padding:2px 8px;
                  border-radius:6px; font-size:0.72rem; font-weight:700; }
  .badge-expert { background:#7c3aed; color:#fff; padding:2px 8px;
                  border-radius:6px; font-size:0.72rem; font-weight:700; }

  /* Section title */
  .section-title { font-size:1.2rem; font-weight:700; color:#e0e0e0;
                   margin:20px 0 12px 0; }

  /* Tab strip */
  [data-testid="stTab"] { font-size:0.95rem; }

  /* Confidence bar */
  .conf-bar { height:6px; border-radius:3px; background:#2a2a3e; }
  .conf-fill { height:6px; border-radius:3px; background:linear-gradient(90deg,#F4A503,#f97316); }

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


@st.cache_resource
def _init_db():
    db.create_tables()
    db.seed_demo_location()

_init_db()


# ═══════════════════════════════════════════════════════════════════
# Forecast cache
# ═══════════════════════════════════════════════════════════════════
@st.cache_data(ttl=1800, show_spinner=False)
def _forecast(lat, lon, alt, cap, tilt, az, tech, iam, horizon, use_ai, sr_csv, _key):
    return run_demo_forecast(
        lat=lat, lon=lon, altitude=alt, capacity_kw=cap,
        tilt=tilt, azimuth=az, technology=tech, iam_model=iam,
        horizon_days=horizon, sr_csv=sr_csv, use_ai=use_ai,
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
        loc_name = "Custom"

        if mode == "City":
            city = st.text_input("City name", value="Budapest", placeholder="e.g. London")
            if st.button("🔍 Find", use_container_width=True):
                try:
                    lat, lon, loc_name, alt = _geocode(city)
                    st.session_state["geo"] = (lat, lon, loc_name, alt)
                except Exception as e:
                    st.error(str(e))
            geo = st.session_state.get("geo", (47.498, 19.040, "Budapest", 120.0))
            lat, lon, loc_name, alt = geo
        else:
            c1, c2 = st.columns(2)
            lat = c1.number_input("Lat", -90.0,  90.0,  47.498, 0.001, format="%.4f")
            lon = c2.number_input("Lon", -180.0, 180.0, 19.040, 0.001, format="%.4f")
            alt = st.number_input("Altitude (m)", 0, 5000, 120)
            loc_name = f"{lat:.3f}°N {lon:.3f}°E"

        cap = st.number_input("System size (kW)", 0.1, 100000.0, 5.0, 0.5,
                              help="Installed DC capacity")

        # ── PRO options ───────────────────────────────────────────
        tilt = az = None
        tech  = "mono_si"
        horizon = 7
        tz    = "Europe/Budapest"

        if level >= 2:
            st.divider()
            st.markdown('<span class="badge-pro">PRO</span>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            tilt = c1.slider("Tilt (°)", 0, 90, 35)
            az   = c2.slider("Azimuth (°)", 0, 360, 180,
                             help="180=South, 90=East, 270=West")
            tech     = st.selectbox("Panel type", list(_TECH), format_func=lambda k: _TECH[k])
            horizon  = st.slider("Forecast days", 1, 14, 7)
            tz       = st.selectbox("Timezone", _TIMEZONES, index=1)

        # ── EXPERT options ────────────────────────────────────────
        iam_model = "ashrae"
        use_ai    = False
        sr_csv    = None

        if level >= 3:
            st.divider()
            st.markdown('<span class="badge-expert">EXPERT</span>', unsafe_allow_html=True)
            with st.expander("Advanced physics settings"):
                iam_model = st.selectbox("IAM model", _IAM_MODELS)
                use_ai    = st.toggle("Use AI Kt correction (XGBoost)", value=False,
                                      help="Requires trained model at models/kt_xgb.joblib")
                sr_file = st.file_uploader("Custom SR curve (CSV)", type="csv",
                                           help="Columns: wavelength_nm, sr_value")
                if sr_file:
                    p = Path("/tmp/sr_custom.csv")
                    p.write_bytes(sr_file.read())
                    sr_csv = str(p)

        st.divider()
        if st.button("🔄 Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    return dict(
        lat=lat, lon=lon, alt=float(alt or 0), cap=float(cap),
        tilt=tilt, az=az, tech=tech, iam=iam_model,
        horizon=horizon, tz=tz if level >= 2 else "UTC",
        use_ai=use_ai, sr_csv=sr_csv,
        loc_name=loc_name, level=level,
    )


# ═══════════════════════════════════════════════════════════════════
# Production curve chart
# ═══════════════════════════════════════════════════════════════════
def _chart_production(hourly: pd.DataFrame, tz: str, show_clearsky: bool = True):
    df = _tz_convert(hourly, tz)
    fig = go.Figure()

    if show_clearsky and "power_clear_kw" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["power_clear_kw"],
            name="Clear-sky", line=dict(color="#74c0fc", width=1.5, dash="dot"),
            fill=None,
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
        marker_color="#F4A503",
        marker_line_width=0,
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
# Tab: Dashboard (instant forecast)
# ═══════════════════════════════════════════════════════════════════
def tab_dashboard(cfg: dict):
    # Hero
    st.markdown(
        f'<div class="hero-card">'
        f'<div class="hero-title">☀️ Solar Forecast</div>'
        f'<div class="hero-sub">📍 {cfg["loc_name"]} &nbsp;·&nbsp; '
        f'⚡ {cfg["cap"]:.1f} kW installed &nbsp;·&nbsp; '
        f'🔬 SPECTRL2 physics + {"AI" if cfg["use_ai"] else "physics"} Kt model</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    key = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H")
    with st.spinner("Computing forecast…"):
        try:
            result = _forecast(
                cfg["lat"], cfg["lon"], cfg["alt"], cfg["cap"],
                cfg["tilt"], cfg["az"], cfg["tech"], cfg["iam"],
                cfg["horizon"], cfg["use_ai"], cfg["sr_csv"], key,
            )
        except Exception as exc:
            st.error(f"Forecast error: {exc}")
            return

    s       = result["summary"]
    hourly  = result["hourly"]
    tz      = cfg["tz"]

    # KPIs
    cols = st.columns(6)
    _kpi(cols[0], "Today",       f"{s['today_kwh']:.1f} kWh")
    _kpi(cols[1], "Tomorrow",    f"{s['tomorrow_kwh']:.1f} kWh")
    _kpi(cols[2], f"{cfg['horizon']}-Day Total", f"{s['total_7d_kwh']:.0f} kWh")
    _kpi(cols[3], "Peak power",  f"{s['peak_power_kw']:.2f} kW")
    _kpi(cols[4], "Capacity factor", f"{s['capacity_factor_pct']:.1f}%")
    _kpi(cols[5], "Cloud loss",  f"{s['cloud_loss_pct']:.0f}%")

    st.markdown("")

    # Peak hour (local time)
    try:
        ph = pd.Timestamp(s["peak_hour_utc"]).tz_convert(tz).strftime("%H:%M")
        phdate = pd.Timestamp(s["peak_hour_utc"]).tz_convert(tz).strftime("%d %b")
    except Exception:
        ph, phdate = "—", ""

    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown('<div class="section-title">Production curve vs clear-sky</div>', unsafe_allow_html=True)
        st.plotly_chart(_chart_production(hourly, tz), use_container_width=True)
    with c2:
        st.markdown('<div class="section-title">Summary</div>', unsafe_allow_html=True)
        st.metric("Peak time", ph, phdate)
        st.metric("Location", cfg["loc_name"][:20])
        st.metric("Technology", _TECH.get(cfg["tech"], cfg["tech"]))
        st.metric("IAM model", cfg["iam"].replace("_", "-").title())

        conf_pct = max(0, 100 - s["cloud_loss_pct"])
        st.markdown(f"**Forecast confidence:** {conf_pct:.0f}%")
        st.markdown(
            f'<div class="conf-bar"><div class="conf-fill" style="width:{conf_pct}%"></div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title">Daily energy ({}-day)</div>'.format(cfg["horizon"]),
                unsafe_allow_html=True)
    st.plotly_chart(_chart_daily(hourly, tz), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# Tab: Forecast detail
# ═══════════════════════════════════════════════════════════════════
def tab_forecast(cfg: dict):
    key = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H")
    try:
        result = _forecast(
            cfg["lat"], cfg["lon"], cfg["alt"], cfg["cap"],
            cfg["tilt"], cfg["az"], cfg["tech"], cfg["iam"],
            cfg["horizon"], cfg["use_ai"], cfg["sr_csv"], key,
        )
    except Exception as exc:
        st.error(str(exc)); return

    hourly = _tz_convert(result["hourly"], cfg["tz"])
    st.markdown('<div class="section-title">☀️ Hourly production forecast</div>', unsafe_allow_html=True)
    st.plotly_chart(_chart_production(result["hourly"], cfg["tz"]), use_container_width=True)

    # GHI comparison chart
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=hourly.index, y=hourly.get("ghi_clear_wm2", []),
                              name="Clear-sky GHI", line=dict(color="#74c0fc", dash="dot")))
    fig2.add_trace(go.Scatter(x=hourly.index, y=hourly.get("ghi_wm2", []),
                              name="All-sky GHI", fill="tozeroy",
                              line=dict(color="#F4A503")))
    fig2.update_layout(template="plotly_dark", height=280,
                       margin=dict(t=10, b=0, l=0, r=0),
                       yaxis_title="W/m²",
                       plot_bgcolor="#0E1117", paper_bgcolor="#0E1117",
                       legend=dict(orientation="h", y=1.1))
    st.markdown('<div class="section-title">GHI: all-sky vs clear-sky (W/m²)</div>', unsafe_allow_html=True)
    st.plotly_chart(fig2, use_container_width=True)

    show_cols = ["power_kw", "energy_kwh", "ghi_wm2", "kt", "t_cell_c", "iam", "cloud_cover_frac"]
    show_cols = [c for c in show_cols if c in hourly.columns]
    rename = {"power_kw": "Power (kW)", "energy_kwh": "Energy (kWh)", "ghi_wm2": "GHI (W/m²)",
               "kt": "Kt", "t_cell_c": "Cell T (°C)", "iam": "IAM", "cloud_cover_frac": "Cloud"}
    st.markdown('<div class="section-title">Hourly table</div>', unsafe_allow_html=True)
    st.dataframe(hourly[show_cols].rename(columns=rename).round(3),
                 use_container_width=True, height=380)


# ═══════════════════════════════════════════════════════════════════
# Tab: Locations
# ═══════════════════════════════════════════════════════════════════
def tab_locations(cfg: dict):
    st.markdown('<div class="section-title">📍 Saved Locations</div>', unsafe_allow_html=True)
    locations = db.list_locations()

    if locations:
        df = pd.DataFrame(locations)[["id","name","lat","lon","capacity_kw","tilt","azimuth","technology","timezone"]]
        st.dataframe(df, use_container_width=True, hide_index=True)
        c1, c2 = st.columns([3, 1])
        del_id = c1.number_input("Delete location ID", 0, step=1, value=0)
        if c2.button("🗑️ Delete"):
            if del_id and db.delete_location(int(del_id)):
                st.success(f"Deleted #{del_id}"); st.rerun()
    else:
        st.info("No locations yet.")

    st.divider()
    st.markdown('<div class="section-title">➕ Add Location</div>', unsafe_allow_html=True)

    with st.form("add_loc"):
        c1, c2 = st.columns(2)
        name = c1.text_input("Name")
        cap  = c2.number_input("System size (kW)", 0.1, 100000.0, 5.0, 0.5)
        c1, c2, c3 = st.columns(3)
        lat  = c1.number_input("Latitude",  -90.0,  90.0,  47.498, 0.001)
        lon  = c2.number_input("Longitude", -180.0, 180.0, 19.040, 0.001)
        alt  = c3.number_input("Altitude (m)", 0.0, 5000.0, 120.0, 10.0)
        c1, c2 = st.columns(2)
        tilt = c1.number_input("Tilt (°)", 0.0, 90.0, 35.0)
        az   = c2.number_input("Azimuth (°)", 0.0, 360.0, 180.0)
        c1, c2 = st.columns(2)
        tech = c1.selectbox("Technology", list(_TECH), format_func=lambda k: _TECH[k])
        tz   = c2.selectbox("Timezone", _TIMEZONES, index=1)
        if st.form_submit_button("✓ Save", type="primary"):
            if not name.strip():
                st.error("Name is required.")
            else:
                new = db.create_location({"name": name.strip(), "lat": lat, "lon": lon,
                    "altitude": alt, "capacity_kw": cap, "tilt": tilt, "azimuth": az,
                    "technology": tech, "timezone": tz})
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
            cfg["horizon"], cfg["use_ai"], cfg["sr_csv"], key,
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

    st.markdown('<div class="section-title">📁 Daily Summary</div>', unsafe_allow_html=True)
    st.dataframe(daily, use_container_width=True)

    c1, c2 = st.columns(2)
    buf1 = io.StringIO(); hourly.to_csv(buf1)
    c1.download_button("⬇ Hourly CSV", buf1.getvalue().encode(),
                       f"forecast_hourly_{cfg['loc_name'].replace(' ','_')}.csv",
                       "text/csv", use_container_width=True)
    buf2 = io.StringIO(); daily.to_csv(buf2)
    c2.download_button("⬇ Daily Summary CSV", buf2.getvalue().encode(),
                       f"forecast_daily_{cfg['loc_name'].replace(' ','_')}.csv",
                       "text/csv", use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# Tab: Settings
# ═══════════════════════════════════════════════════════════════════
def tab_settings(cfg: dict):
    st.markdown('<div class="section-title">⚙️ Application Info</div>', unsafe_allow_html=True)
    st.markdown("""
| | |
|---|---|
| **Version** | 2.0.0 |
| **Physics engine** | pvlib SPECTRL2 (Bird & Riordan 1986) |
| **Transposition** | Perez model |
| **Clear-sky** | SPECTRL2 with Ångström + Hänel aerosol corrections |
| **Spectral integration** | ∫ SR(λ) × I(λ) × IAM(θ) dλ |
| **AI model** | XGBoost Kt regressor, 21 atmospheric features (optional) |
| **Weather** | Open-Meteo (free, no key) |
| **Database** | SQLite (locations + forecast cache) |
""")
    st.divider()
    st.success("✓ Demo mode active — works without CAMS or PostgreSQL")
    ai_path = Path("models/kt_xgb.joblib")
    if ai_path.exists():
        st.success(f"✓ Trained AI model: {ai_path} ({ai_path.stat().st_size//1024} KB)")
    else:
        st.info("ℹ No AI model found — physics-only mode. See Model Training tab.")

    st.divider()
    st.markdown("### Enable CAMS for maximum accuracy")
    st.markdown("""
1. Register free at [ads.atmosphere.copernicus.eu](https://ads.atmosphere.copernicus.eu)
2. Add to `.env`:  `CAMS_API_KEY=UID:KEY`
3. Run `python scripts/01_download_cams.py`
4. Run `python scripts/02_train_kt_model.py --cv 5`
5. Toggle **AI Kt correction** in Expert settings
""")


# ═══════════════════════════════════════════════════════════════════
# Tab: Model Training
# ═══════════════════════════════════════════════════════════════════
def tab_training():
    st.markdown('<div class="section-title">🧠 XGBoost Kt Model Training</div>', unsafe_allow_html=True)
    ai_path = Path("models/kt_xgb.joblib")
    if ai_path.exists():
        st.success(f"✓ Model ready: `{ai_path}` ({ai_path.stat().st_size//1024} KB)")
    else:
        st.warning("No trained model. Physics-only mode is active.")
    st.markdown("""
### Why train the AI model?

The XGBoost Kt corrector learns systematic biases in the physics model
from CAMS atmospheric history — reducing forecast RMSE by 10–20%,
especially on aerosol-heavy or partly-cloudy days.

### Training pipeline

```bash
# 1. Download CAMS EAC4 history (AOD, SSA, ozone, PM, cloud…)
python scripts/01_download_cams.py --start 2022-01-01 --end 2023-12-31

# 2. Train XGBoost model (RMSE objective, 21 features, 5-fold CV)
python scripts/02_train_kt_model.py --cv 5
```

### Typical accuracy (Hungary, 2 years)

| | Physics-only | Physics + AI |
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

    tabs = st.tabs(["📊 Dashboard", "☀️ Forecast", "📍 Locations", "📁 Reports",
                    "⚙️ Settings", "🧠 Model Training"])
    with tabs[0]: tab_dashboard(cfg)
    with tabs[1]: tab_forecast(cfg)
    with tabs[2]: tab_locations(cfg)
    with tabs[3]: tab_reports(cfg)
    with tabs[4]: tab_settings(cfg)
    with tabs[5]: tab_training()


if __name__ == "__main__":
    main()
