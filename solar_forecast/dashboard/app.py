"""
Solar Forecast Pro — Productized SaaS Dashboard.

Buyer-friendly multi-tab interface:
  📊 Dashboard       — KPIs, today's curve, weekly summary
  📍 Locations       — Multi-site CRUD (SQLite)
  ☀️  Forecast        — 7-day hourly forecast for selected location
  📁 Reports         — CSV/Excel export, monthly summary
  ⚙️  Settings        — Defaults, units, theme
  🧠 Model Training  — (Advanced) XGBoost Kt training (optional)

Demo mode: works with no CAMS key, no PostgreSQL, no trained model.
"""

from __future__ import annotations

import io
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.db import sqlite_manager as db
from solar_forecast.demo.pipeline import run_demo_forecast

logging.basicConfig(level=logging.WARNING)

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Solar Forecast Pro",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-title { font-size:2rem; font-weight:700; color:#F4A503; margin-bottom:8px; }
    .sub-title  { color:#aaa; font-size:0.95rem; margin-bottom:24px; }
    .metric-box { background:#1E1E2E; border-radius:12px; padding:18px;
                  text-align:center; border:1px solid #2a2a3e; }
    .metric-val { font-size:2rem; font-weight:800; color:#F4A503; line-height:1.1; }
    .metric-lbl { font-size:0.85rem; color:#888; margin-top:4px; }
    .badge      { background:#F4A503; color:#0E1117; padding:2px 8px;
                  border-radius:6px; font-size:0.75rem; font-weight:700; }
    footer { visibility:hidden; }
    [data-testid="stMetricValue"] { font-size: 1.6rem; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# State + DB initialization
# ══════════════════════════════════════════════════════════════════════════

@st.cache_resource
def _init_db():
    db.create_tables()
    db.seed_demo_location()
    return True


_init_db()


_TECH_LABELS = {
    "mono_si": "Mono-crystalline Si",
    "poly_si": "Poly-crystalline Si",
    "cdte":    "CdTe (thin-film)",
    "cigs":    "CIGS (thin-film)",
    "hit":     "HIT / Heterojunction",
}

_TZ_OPTIONS = [
    "UTC", "Europe/Budapest", "Europe/Vienna", "Europe/Berlin",
    "Europe/London", "Europe/Paris", "Europe/Warsaw", "Europe/Bucharest",
    "Europe/Athens", "Europe/Madrid", "Europe/Rome",
    "US/Eastern", "US/Central", "US/Mountain", "US/Pacific",
    "Asia/Tokyo", "Asia/Shanghai", "Asia/Kolkata", "Australia/Sydney",
]


def _metric_card(col, label: str, value: str, sub: str = ""):
    col.markdown(
        f'<div class="metric-box"><div class="metric-val">{value}</div>'
        f'<div class="metric-lbl">{label}</div>'
        f'<div style="color:#666;font-size:0.75rem;margin-top:4px;">{sub}</div></div>',
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=1800, show_spinner=False)
def _cached_forecast(lat: float, lon: float, alt: float, cap_kw: float,
                     tilt: Optional[float], az: Optional[float],
                     tech: str, horizon: int, _refresh_key: str):
    return run_demo_forecast(
        lat=lat, lon=lon, altitude=alt, capacity_kw=cap_kw,
        tilt=tilt, azimuth=az, technology=tech, horizon_days=horizon,
    )


def _to_local(df: pd.DataFrame, tz: str) -> pd.DataFrame:
    if df.empty or tz == "UTC":
        return df
    out = df.copy()
    out.index = out.index.tz_convert(tz)
    return out


# ══════════════════════════════════════════════════════════════════════════
# Sidebar — location picker (compact, buyer-friendly)
# ══════════════════════════════════════════════════════════════════════════

def _sidebar() -> dict:
    st.sidebar.markdown("### ☀️ Solar Forecast Pro")
    st.sidebar.caption("v2.0 — Physics + AI hybrid")

    locations = db.list_locations()
    if not locations:
        st.sidebar.warning("No locations. Add one in the **Locations** tab.")
        return {"location": None}

    options = {f"{l['name']} ({l['lat']:.2f}°, {l['lon']:.2f}°)": l for l in locations}
    label = st.sidebar.selectbox("Active location", list(options.keys()))
    loc = options[label]

    st.sidebar.markdown("---")
    st.sidebar.markdown("**System**")
    st.sidebar.text(f"Capacity: {loc['capacity_kw']:.1f} kW")
    st.sidebar.text(f"Tech:     {_TECH_LABELS.get(loc['technology'], loc['technology'])}")
    if loc.get("tilt") is not None:
        st.sidebar.text(f"Tilt:     {loc['tilt']:.1f}°")
    if loc.get("azimuth") is not None:
        st.sidebar.text(f"Azimuth:  {loc['azimuth']:.1f}°")
    st.sidebar.text(f"TZ:       {loc['timezone']}")

    st.sidebar.markdown("---")
    horizon = st.sidebar.slider("Forecast horizon (days)", 1, 14, 7)

    if st.sidebar.button("🔄 Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.caption("Data: Open-Meteo (live), pvlib spectrl2 physics. "
                       "AOD/SSA fallback to climatology when CAMS unavailable.")

    return {"location": loc, "horizon": horizon}


# ══════════════════════════════════════════════════════════════════════════
# Tab 1 — Dashboard
# ══════════════════════════════════════════════════════════════════════════

def tab_dashboard(state: dict):
    st.markdown('<div class="main-title">📊 Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Live overview of your solar production</div>',
                unsafe_allow_html=True)

    loc = state.get("location")
    if not loc:
        st.info("👈 Add a location in the **Locations** tab to get started.")
        return

    refresh_key = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H")
    with st.spinner("Computing forecast…"):
        try:
            result = _cached_forecast(
                loc["lat"], loc["lon"], loc.get("altitude", 0.0),
                loc["capacity_kw"], loc.get("tilt"), loc.get("azimuth"),
                loc.get("technology", "mono_si"),
                state.get("horizon", 7),
                refresh_key,
            )
        except Exception as exc:
            st.error(f"Forecast failed: {exc}")
            return

    s = result["summary"]
    hourly = _to_local(result["hourly"], loc["timezone"])

    # KPI row
    cols = st.columns(4)
    _metric_card(cols[0], "Today",      f"{s['today_kwh']:.1f} kWh")
    _metric_card(cols[1], "Tomorrow",   f"{s['tomorrow_kwh']:.1f} kWh")
    _metric_card(cols[2], "7-day total",f"{s['total_7d_kwh']:.0f} kWh")
    _metric_card(cols[3], "Peak power", f"{s['peak_power_kw']:.2f} kW")

    cols = st.columns(4)
    _metric_card(cols[0], "Capacity factor", f"{s['capacity_factor_pct']:.1f}%")
    _metric_card(cols[1], "Cloud loss",      f"{s['cloud_loss_pct']:.0f}%")
    peak_hour = s.get('peak_hour_utc', '')
    try:
        peak_local = pd.Timestamp(peak_hour).tz_convert(loc["timezone"]).strftime("%H:%M")
    except Exception:
        peak_local = "—"
    _metric_card(cols[2], "Peak hour", peak_local, sub=loc["timezone"])
    _metric_card(cols[3], "Capacity",  f"{loc['capacity_kw']:.1f} kW")

    st.markdown("### Today's production curve")
    today_local = pd.Timestamp.now(tz=loc["timezone"]).normalize()
    today_mask = (hourly.index >= today_local) & (hourly.index < today_local + pd.Timedelta(days=1))
    today_df = hourly.loc[today_mask]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=today_df.index, y=today_df["power_kw"],
        mode="lines", fill="tozeroy", name="Power (kW)",
        line=dict(color="#F4A503", width=3),
    ))
    fig.update_layout(
        height=320, template="plotly_dark", margin=dict(t=10, b=10, l=10, r=10),
        xaxis_title="", yaxis_title="kW",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 7-day daily energy")
    daily = hourly.groupby(hourly.index.normalize())["energy_kwh"].sum().head(7)
    fig2 = go.Figure(go.Bar(
        x=[d.strftime("%a %d") for d in daily.index],
        y=daily.values,
        marker_color="#F4A503",
        text=[f"{v:.1f}" for v in daily.values],
        textposition="outside",
    ))
    fig2.update_layout(
        height=280, template="plotly_dark", margin=dict(t=10, b=10, l=10, r=10),
        yaxis_title="kWh",
    )
    st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# Tab 2 — Locations
# ══════════════════════════════════════════════════════════════════════════

def tab_locations(state: dict):
    st.markdown('<div class="main-title">📍 Locations</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Manage your solar installations</div>',
                unsafe_allow_html=True)

    locations = db.list_locations()

    if locations:
        df = pd.DataFrame(locations)
        display_df = df[["id", "name", "lat", "lon", "capacity_kw",
                         "tilt", "azimuth", "technology", "timezone"]]
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        c1, c2 = st.columns([3, 1])
        del_id = c1.number_input("Location ID to delete", min_value=0, value=0, step=1)
        if c2.button("🗑️ Delete", type="secondary"):
            if del_id and db.delete_location(int(del_id)):
                st.success(f"Deleted location #{del_id}")
                st.rerun()
            else:
                st.warning("No matching location.")
    else:
        st.info("No locations yet. Add your first one below.")

    st.divider()
    st.subheader("➕ Add new location")

    with st.form("add_location"):
        c1, c2 = st.columns(2)
        name = c1.text_input("Name", placeholder="e.g. Office Roof Array")
        capacity_kw = c2.number_input("System size (kW)", 0.1, 10000.0, 5.0, 0.5)

        c1, c2, c3 = st.columns(3)
        lat = c1.number_input("Latitude",  -90.0,  90.0,  47.498, 0.001)
        lon = c2.number_input("Longitude", -180.0, 180.0, 19.040, 0.001)
        alt = c3.number_input("Altitude (m)", 0.0, 5000.0, 120.0, 10.0)

        c1, c2 = st.columns(2)
        tilt    = c1.number_input("Tilt (°)",    0.0, 90.0,  35.0, 1.0)
        azimuth = c2.number_input("Azimuth (°, 180=South)", 0.0, 360.0, 180.0, 5.0)

        c1, c2 = st.columns(2)
        tech = c1.selectbox("Cell technology",
                            list(_TECH_LABELS.keys()),
                            format_func=lambda k: _TECH_LABELS[k])
        tz   = c2.selectbox("Timezone", _TZ_OPTIONS, index=1)

        submitted = st.form_submit_button("✓ Save location", type="primary")
        if submitted:
            if not name.strip():
                st.error("Name is required.")
            else:
                try:
                    new = db.create_location({
                        "name": name.strip(), "lat": lat, "lon": lon,
                        "altitude": alt, "capacity_kw": capacity_kw,
                        "tilt": tilt, "azimuth": azimuth,
                        "technology": tech, "timezone": tz,
                    })
                    st.success(f"Created location #{new['id']}: {new['name']}")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Failed: {exc}")


# ══════════════════════════════════════════════════════════════════════════
# Tab 3 — Forecast (detailed)
# ══════════════════════════════════════════════════════════════════════════

def tab_forecast(state: dict):
    st.markdown('<div class="main-title">☀️ Forecast</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Hourly production forecast</div>',
                unsafe_allow_html=True)

    loc = state.get("location")
    if not loc:
        st.info("Select a location in the sidebar.")
        return

    refresh_key = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H")
    try:
        result = _cached_forecast(
            loc["lat"], loc["lon"], loc.get("altitude", 0.0),
            loc["capacity_kw"], loc.get("tilt"), loc.get("azimuth"),
            loc.get("technology", "mono_si"),
            state.get("horizon", 7),
            refresh_key,
        )
    except Exception as exc:
        st.error(f"Forecast failed: {exc}")
        return

    hourly = _to_local(result["hourly"], loc["timezone"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hourly.index, y=hourly["power_kw"],
        mode="lines", fill="tozeroy", name="Power (kW)",
        line=dict(color="#F4A503", width=2),
    ))
    if "ghi_wm2" in hourly.columns:
        fig.add_trace(go.Scatter(
            x=hourly.index, y=hourly["ghi_wm2"],
            name="GHI (W/m²)", yaxis="y2",
            line=dict(color="#74c0fc", width=1.5, dash="dot"),
        ))
    fig.update_layout(
        height=420, template="plotly_dark",
        margin=dict(t=20, b=10, l=10, r=10),
        yaxis=dict(title="Power (kW)"),
        yaxis2=dict(title="GHI (W/m²)", overlaying="y", side="right"),
        legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Hourly table")
    show = hourly[["power_kw", "energy_kwh", "ghi_wm2", "kt", "t_cell_c"]].copy()
    show.columns = ["Power (kW)", "Energy (kWh)", "GHI (W/m²)",
                    "Clearness Kt", "Cell T (°C)"]
    st.dataframe(show.round(3), use_container_width=True, height=400)


# ══════════════════════════════════════════════════════════════════════════
# Tab 4 — Reports
# ══════════════════════════════════════════════════════════════════════════

def tab_reports(state: dict):
    st.markdown('<div class="main-title">📁 Reports</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Export production data and summaries</div>',
                unsafe_allow_html=True)

    loc = state.get("location")
    if not loc:
        st.info("Select a location in the sidebar.")
        return

    refresh_key = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H")
    try:
        result = _cached_forecast(
            loc["lat"], loc["lon"], loc.get("altitude", 0.0),
            loc["capacity_kw"], loc.get("tilt"), loc.get("azimuth"),
            loc.get("technology", "mono_si"),
            state.get("horizon", 7),
            refresh_key,
        )
    except Exception as exc:
        st.error(f"Forecast failed: {exc}")
        return

    hourly = _to_local(result["hourly"], loc["timezone"])
    daily = hourly.groupby(hourly.index.normalize()).agg(
        energy_kwh=("energy_kwh", "sum"),
        peak_kw=("power_kw", "max"),
        avg_ghi=("ghi_wm2", "mean"),
    ).round(2)
    daily.index.name = "date"

    st.subheader("Daily summary")
    st.dataframe(daily, use_container_width=True)

    csv_buf = io.StringIO()
    hourly.to_csv(csv_buf)
    daily_buf = io.StringIO()
    daily.to_csv(daily_buf)

    c1, c2 = st.columns(2)
    c1.download_button(
        "⬇ Download hourly CSV",
        csv_buf.getvalue().encode("utf-8"),
        file_name=f"forecast_hourly_{loc['name'].replace(' ', '_')}.csv",
        mime="text/csv",
        use_container_width=True,
    )
    c2.download_button(
        "⬇ Download daily summary CSV",
        daily_buf.getvalue().encode("utf-8"),
        file_name=f"forecast_daily_{loc['name'].replace(' ', '_')}.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ══════════════════════════════════════════════════════════════════════════
# Tab 5 — Settings
# ══════════════════════════════════════════════════════════════════════════

def tab_settings(state: dict):
    st.markdown('<div class="main-title">⚙️ Settings</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Application preferences and defaults</div>',
                unsafe_allow_html=True)

    st.subheader("Application info")
    st.markdown("""
    | Item | Value |
    |---|---|
    | Version | **2.0.0** |
    | Database | SQLite (`data/solar_forecast.db`) |
    | Weather provider | Open-Meteo (free, no API key) |
    | Physics engine | pvlib spectrl2 + Perez transposition |
    | AI model (optional) | XGBoost regressor on 21 atmospheric features |
    """)

    st.subheader("Demo mode")
    st.success("✓ Demo mode is **active** — works without CAMS or PostgreSQL.")
    st.markdown("""
    - Uses **Open-Meteo** for live weather (free, no key)
    - Uses **climatological aerosol fallbacks** (continental Europe defaults)
    - Falls back to **physics-only** if no XGBoost model is trained
    """)

    st.subheader("Optional: enable CAMS for higher accuracy")
    with st.expander("CAMS / Copernicus Atmosphere Service setup"):
        st.markdown("""
        1. Register a free account at [ads.atmosphere.copernicus.eu](https://ads.atmosphere.copernicus.eu)
        2. Copy your UID:KEY from your profile page
        3. Add to `.env`:  `CAMS_API_KEY=00000:xxxxxxxx-...`
        4. Restart the app
        5. Use the **Model Training** tab to download data and train the AI model
        """)


# ══════════════════════════════════════════════════════════════════════════
# Tab 6 — Model Training (Advanced)
# ══════════════════════════════════════════════════════════════════════════

def tab_training(state: dict):
    st.markdown('<div class="main-title">🧠 Model Training</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Advanced — XGBoost Kt model (optional)</div>',
                unsafe_allow_html=True)

    model_path = Path("models/kt_xgb.joblib")
    if model_path.exists():
        size_kb = model_path.stat().st_size / 1024
        st.success(f"✓ Trained model present: `{model_path}` ({size_kb:.1f} KB)")
    else:
        st.info("No trained model yet. The app uses physics-only forecasts.")

    st.markdown("""
    ### What this does
    Training the XGBoost Kt model improves accuracy by learning a clearness
    index correction from CAMS atmospheric features (AOD, SSA, PM, ozone, …).

    ### Requirements
    - **CAMS API key** (free) — set `CAMS_API_KEY` in your `.env`
    - **PostgreSQL** running (configured in `config.yaml`)
    - **Historical CAMS data** downloaded (≥ 1 year recommended)

    ### Training pipeline
    ```bash
    # 1. Download CAMS atmospheric + radiation history
    python scripts/01_download_cams.py --start 2022-01-01 --end 2023-12-31

    # 2. Train the XGBoost Kt model
    python scripts/02_train_kt_model.py --cv 5
    ```

    The model is saved to `models/kt_xgb.joblib` and is automatically loaded
    by the forecast pipeline once present.
    """)


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    state = _sidebar()

    tabs = st.tabs([
        "📊 Dashboard",
        "📍 Locations",
        "☀️ Forecast",
        "📁 Reports",
        "⚙️ Settings",
        "🧠 Model Training",
    ])

    with tabs[0]: tab_dashboard(state)
    with tabs[1]: tab_locations(state)
    with tabs[2]: tab_forecast(state)
    with tabs[3]: tab_reports(state)
    with tabs[4]: tab_settings(state)
    with tabs[5]: tab_training(state)


if __name__ == "__main__":
    main()
