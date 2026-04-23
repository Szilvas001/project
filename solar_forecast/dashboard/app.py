"""
Solar Forecast — Live Dashboard

Run:
    streamlit run solar_forecast/dashboard/app.py

UI flow:
  1. User enters location (city name or GPS) + system size (kW)
  2. Optional: tilt/azimuth, custom SR curve CSV, IAM model
  3. App fetches Open-Meteo live weather → computes clear-sky → hybrid Kt
     → production forecast for next 7 days
  4. Displays:
     – Hourly production curve (interactive Plotly chart)
     – Daily energy totals table
     – Key stats: peak power, total energy, capacity factor
"""

import io
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Configure logging before imports so sub-modules respect it
logging.basicConfig(level=logging.WARNING)

# ── Streamlit page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Solar Forecast",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS (minimal, professional) ────────────────────────────────────
st.markdown("""
<style>
    .main-title  { font-size: 2rem; font-weight: 700; color: #F4A503; }
    .metric-box  { background: #1E1E2E; border-radius: 12px; padding: 16px;
                   text-align: center; }
    .metric-val  { font-size: 2rem; font-weight: 800; color: #F4A503; }
    .metric-lbl  { font-size: 0.85rem; color: #888; }
    footer       { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# Helpers / caching
# ══════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def geocode(city: str) -> tuple[float, float, str]:
    from solar_forecast.utils import geocode_city
    return geocode_city(city)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_weather(lat: float, lon: float) -> pd.DataFrame:
    from solar_forecast.data_ingestion.openmeteo_live import OpenMeteoClient
    cfg = _minimal_cfg(lat, lon)
    client = OpenMeteoClient(cfg)
    return client.get_forecast(lat, lon, days=7)


@st.cache_data(ttl=3600, show_spinner=False)
def compute_clearsky_cached(
    lat: float, lon: float, altitude: float,
    tilt: float, azimuth: float,
    times_key: str,            # cache key = ISO range string
    _times: object,            # actual DatetimeIndex (not hashable → prefixed _)
    aod_arr: tuple,
    pw_arr:  tuple,
    pres_arr: tuple,
) -> pd.DataFrame:
    from solar_forecast.clearsky.spectrl2_model import compute_clearsky
    return compute_clearsky(
        times=_times,
        lat=lat, lon=lon, altitude=altitude,
        tilt=tilt, azimuth=azimuth,
        aod_550nm=np.array(aod_arr),
        precipitable_water=np.array(pw_arr),
        surface_pressure=np.array(pres_arr),
    )


def run_forecast(
    lat: float,
    lon: float,
    altitude: float,
    capacity_kw: float,
    tilt: float,
    azimuth: float,
    sr_csv_bytes: bytes | None,
    iam_type: str,
    kt_model_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full forecast pipeline (no Streamlit cache — result changes with model).
    Returns (hourly_df, daily_df).
    """
    from solar_forecast.allsky.hybrid_model import AllSkyModel
    from solar_forecast.production.pv_output import PVOutputModel

    weather = fetch_weather(lat, lon)
    times = weather.index

    # Cache key for clear-sky (depends on atmospheric arrays)
    aod_arr  = tuple(weather["aod_550nm"].fillna(0.1).round(4).tolist())
    pw_arr   = tuple(weather["precipitable_water"].fillna(1.5).round(3).tolist())
    pres_arr = tuple(weather.get("surface_pressure",
                     pd.Series(1013.25, index=times)).fillna(1013.25).round(1).tolist())
    times_key = f"{times[0].isoformat()}_{times[-1].isoformat()}"

    cs_df = compute_clearsky_cached(
        lat, lon, altitude, tilt, azimuth,
        times_key, times,
        aod_arr, pw_arr, pres_arr,
    )

    cfg = _minimal_cfg(lat, lon, altitude, capacity_kw, tilt, azimuth)
    cfg["model"]["kt_model_path"] = kt_model_path

    # All-sky irradiance
    allsky_model = AllSkyModel(cfg)
    allsky_model.load_kt_model()
    allsky_df = allsky_model.forecast(times, weather, cs_df)

    # PV power
    sr_csv_path = None
    if sr_csv_bytes:
        tmp = Path("/tmp/user_sr.csv")
        tmp.write_bytes(sr_csv_bytes)
        sr_csv_path = str(tmp)

    pv = PVOutputModel(cfg, sr_csv=sr_csv_path, iam_model=iam_type)
    prod_df = pv.run_from_live(allsky_df, weather, lat, lon, altitude)

    hourly = pd.concat([prod_df, allsky_df[["ghi", "kt", "ghi_clear"]]], axis=1)
    hourly.index = hourly.index.tz_convert("UTC")

    # Daily aggregates
    daily = hourly["power_kw"].resample("D").mean()  # mean kW → kWh if × 1h
    daily_kwh = hourly["power_kw"].resample("D").sum()  # sum of hourly kW ≈ kWh
    daily_df = pd.DataFrame({
        "Date":          daily.index.strftime("%Y-%m-%d"),
        "Energy (kWh)":  daily_kwh.round(2),
        "Avg Power (kW)": daily.round(2),
        "Peak Power (kW)": hourly["power_kw"].resample("D").max().round(2),
    }).reset_index(drop=True)

    return hourly, daily_df


def _minimal_cfg(
    lat: float, lon: float, altitude: float = 0.0,
    capacity_kw: float = 10.0, tilt: float = 30.0, azimuth: float = 180.0,
) -> dict:
    return {
        "location": {"lat": lat, "lon": lon, "altitude": altitude},
        "system": {
            "capacity_kw": capacity_kw,
            "tilt": tilt,
            "azimuth": azimuth,
            "module_efficiency": 0.205,
            "temperature_coefficient": -0.0040,
            "noct": 44.0,
            "ground_albedo": 0.20,
            "inverter_efficiency": 0.97,
            "wiring_loss": 0.02,
            "soiling_loss": 0.02,
        },
        "model": {
            "kt_model_path": "models/kt_xgb.joblib",
            "physics_weight": 0.40,
            "min_train_samples": 500,
        },
        "openmeteo": {
            "forecast_url": "https://api.open-meteo.com/v1/forecast",
            "historical_url": "https://archive-api.open-meteo.com/v1/archive",
            "cache_expire_hours": 1,
        },
        "cams": {},
        "database": {},
    }


# ══════════════════════════════════════════════════════════════════════════
# Sidebar — user inputs
# ══════════════════════════════════════════════════════════════════════════

def _sidebar() -> dict:
    st.sidebar.markdown("## ☀️ System Configuration")

    # Location input
    loc_mode = st.sidebar.radio("Location input", ["City name", "GPS coordinates"],
                                 horizontal=True)
    if loc_mode == "City name":
        city = st.sidebar.text_input("City", value="Budapest")
        lat_in = lon_in = None
    else:
        city = None
        col1, col2 = st.sidebar.columns(2)
        lat_in = col1.number_input("Latitude", -90.0, 90.0, 47.498, 0.001)
        lon_in = col2.number_input("Longitude", -180.0, 180.0, 19.040, 0.001)

    # System size
    capacity_kw = st.sidebar.number_input(
        "System size (kW)", min_value=0.1, max_value=10000.0,
        value=10.0, step=0.5,
    )

    altitude = st.sidebar.number_input(
        "Altitude (m)", min_value=0, max_value=5000, value=120, step=10,
    )

    # Panel configuration
    st.sidebar.markdown("---")
    profile = st.sidebar.selectbox(
        "Panel profile",
        ["🏠 Residential roof", "🏢 Flat roof", "🌍 Optimised (latitude-based)"],
    )

    show_advanced = st.sidebar.checkbox("Advanced settings")
    if show_advanced:
        tilt_val   = st.sidebar.slider("Tilt (°)", 0, 90, 30)
        azimuth_val = st.sidebar.slider("Azimuth (°, 180=south)", 0, 360, 180)
        iam_type   = st.sidebar.selectbox("IAM model", ["ashrae", "martin_ruiz", "fresnel"])
    else:
        iam_type = "ashrae"
        # Derive tilt/azimuth from profile
        if loc_mode == "City name":
            ref_lat = 47.5   # will be updated after geocode
        else:
            ref_lat = lat_in or 47.5
        tilt_val, azimuth_val = _profile_defaults(profile, ref_lat)

    # Custom spectral response CSV
    st.sidebar.markdown("---")
    sr_file = st.sidebar.file_uploader(
        "Custom SR curve (CSV: wavelength_nm, sr_value)", type="csv"
    )
    sr_bytes = sr_file.read() if sr_file else None

    kt_model_path = st.sidebar.text_input("Kt model path", "models/kt_xgb.joblib")

    return {
        "loc_mode": loc_mode,
        "city": city,
        "lat_in": lat_in,
        "lon_in": lon_in,
        "capacity_kw": capacity_kw,
        "altitude": altitude,
        "profile": profile,
        "tilt": tilt_val,
        "azimuth": azimuth_val,
        "iam_type": iam_type,
        "sr_bytes": sr_bytes,
        "kt_model_path": kt_model_path,
        "show_advanced": show_advanced,
    }


def _profile_defaults(profile: str, lat: float) -> tuple[float, float]:
    azimuth = 180.0 if lat >= 0 else 0.0
    if "Flat" in profile:
        return 12.0, azimuth
    elif "Optimised" in profile:
        return round(abs(lat) * 0.76, 1), azimuth
    else:
        return 30.0, azimuth  # Residential


# ══════════════════════════════════════════════════════════════════════════
# Plotly charts
# ══════════════════════════════════════════════════════════════════════════

def _hourly_chart(hourly: pd.DataFrame, capacity_kw: float) -> go.Figure:
    fig = go.Figure()

    # GHI clear-sky (reference)
    fig.add_trace(go.Scatter(
        x=hourly.index,
        y=hourly["ghi_clear"].clip(0) / 1000,
        name="Clear-sky GHI (kW/m²)",
        line=dict(color="#FFD700", dash="dot", width=1),
        fill=None,
        opacity=0.5,
    ))

    # All-sky GHI
    fig.add_trace(go.Scatter(
        x=hourly.index,
        y=hourly["ghi"].clip(0) / 1000,
        name="GHI (kW/m²)",
        line=dict(color="#FFA500", width=1.5),
        opacity=0.7,
    ))

    # AC power output (primary)
    fig.add_trace(go.Scatter(
        x=hourly.index,
        y=hourly["power_kw"],
        name=f"AC Power (kW) — {capacity_kw} kW system",
        line=dict(color="#00D4FF", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(0,212,255,0.12)",
    ))

    fig.update_layout(
        title="7-Day Solar Production Forecast",
        xaxis_title="Time (UTC)",
        yaxis_title="Power / Irradiance",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        hovermode="x unified",
        template="plotly_dark",
        height=440,
        margin=dict(l=0, r=0, t=50, b=0),
        yaxis=dict(rangemode="tozero"),
    )
    return fig


def _kt_chart(hourly: pd.DataFrame) -> go.Figure:
    mask = hourly["ghi_clear"] > 10
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hourly.index[mask],
        y=hourly["kt"][mask],
        name="Clearness index Kt",
        line=dict(color="#7CFC00", width=1.5),
    ))
    fig.add_hline(y=1.0, line_dash="dot", line_color="white", opacity=0.3,
                  annotation_text="Clear sky (Kt=1)")
    fig.update_layout(
        title="Clearness Index Kt",
        xaxis_title="Time (UTC)",
        yaxis_title="Kt",
        template="plotly_dark",
        height=260,
        margin=dict(l=0, r=0, t=40, b=0),
        yaxis=dict(range=[0, 1.1]),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    st.markdown('<div class="main-title">☀️ Solar Production Forecast</div>',
                unsafe_allow_html=True)
    st.caption("Physics × AI hybrid | spectrl2 clear-sky | CAMS-trained Kt | Open-Meteo live")

    inputs = _sidebar()

    # Resolve location
    with st.spinner("Resolving location…"):
        try:
            if inputs["loc_mode"] == "City name":
                lat, lon, display_name = geocode(inputs["city"])
            else:
                lat = float(inputs["lat_in"])
                lon = float(inputs["lon_in"])
                display_name = f"{lat:.4f}°, {lon:.4f}°"
        except Exception as exc:
            st.error(f"Could not geocode location: {exc}")
            st.stop()

    altitude = float(inputs["altitude"])

    # Update tilt/azimuth if not advanced mode
    if not inputs["show_advanced"]:
        tilt, azimuth = _profile_defaults(inputs["profile"], lat)
    else:
        tilt    = float(inputs["tilt"])
        azimuth = float(inputs["azimuth"])

    capacity_kw = float(inputs["capacity_kw"])

    # Header row
    col1, col2, col3 = st.columns([3, 2, 2])
    col1.info(f"📍 **{display_name}**  |  {lat:.4f}°, {lon:.4f}°  |  {altitude} m")
    col2.info(f"⚡ **{capacity_kw} kW**  |  Tilt {tilt}°  |  Az {azimuth}°")
    col3.info(f"🔬 IAM: {inputs['iam_type']}  |  SR: {'custom' if inputs['sr_bytes'] else 'c-Si standard'}")

    st.markdown("---")

    # Run forecast
    with st.spinner("Computing solar forecast…"):
        try:
            hourly, daily_df = run_forecast(
                lat=lat, lon=lon, altitude=altitude,
                capacity_kw=capacity_kw,
                tilt=tilt, azimuth=azimuth,
                sr_csv_bytes=inputs["sr_bytes"],
                iam_type=inputs["iam_type"],
                kt_model_path=inputs["kt_model_path"],
            )
        except Exception as exc:
            st.error(f"Forecast failed: {exc}")
            st.exception(exc)
            st.stop()

    # ── Key metrics ──────────────────────────────────────────────────────
    total_energy  = float(hourly["power_kw"].sum())       # kWh (sum of hourly kW)
    peak_power    = float(hourly["power_kw"].max())
    cap_factor    = peak_power / capacity_kw if capacity_kw > 0 else 0
    avg_daily_kwh = total_energy / 7

    m1, m2, m3, m4 = st.columns(4)
    _metric(m1, "7-Day Energy",  f"{total_energy:,.0f} kWh", "Total production")
    _metric(m2, "Daily Average", f"{avg_daily_kwh:,.1f} kWh", "Per day")
    _metric(m3, "Peak Power",    f"{peak_power:,.1f} kW",   "Instantaneous max")
    _metric(m4, "Capacity Factor", f"{cap_factor*100:.1f} %", "Peak / installed")

    st.markdown("---")

    # ── Charts ────────────────────────────────────────────────────────────
    st.plotly_chart(_hourly_chart(hourly, capacity_kw), use_container_width=True)

    c1, c2 = st.columns([2, 1])
    with c1:
        st.plotly_chart(_kt_chart(hourly), use_container_width=True)
    with c2:
        st.markdown("### Daily Summary")
        st.dataframe(daily_df, use_container_width=True, hide_index=True)

    # ── Download ──────────────────────────────────────────────────────────
    st.markdown("---")
    csv_buf = io.StringIO()
    hourly[["power_kw", "ghi", "ghi_clear", "kt"]].to_csv(csv_buf)
    st.download_button(
        "⬇️ Download hourly data (CSV)",
        data=csv_buf.getvalue(),
        file_name=f"solar_forecast_{display_name.replace(' ','_')}.csv",
        mime="text/csv",
    )

    # ── Info footer ───────────────────────────────────────────────────────
    with st.expander("ℹ️ Model details"):
        st.markdown("""
**Clear-sky model:** pvlib `spectrl2` (Bird & Riordan 1986) — spectral irradiance
integrated 300–4000 nm with site-specific atmospheric inputs (AOD, ozone, water vapour, pressure).

**All-sky Kt model:** Physics + XGBoost hybrid.
- *Physics component*: Delta-Eddington two-stream cloud transmittance
  with Beer-Lambert direct beam and backscattered diffuse.
  Kt_cloud = (1−fc) + fc × [Rd + Rn × (ω_c + (1−ω_c)·exp(−COD/cos θ))]
- *AI component*: XGBoost trained on CAMS EAC4 atmospheric reanalysis
  + CAMS radiation service (GHI truth). Features: cloud cover, COD, AOD,
  precipitable water, ozone, zenith, season.
- *Blend*: α × Kt_phys + (1−α) × Kt_AI  (α=0.40 default)

**Decomposition:** Erbs-extended with clear-sky beam fraction prior.

**Transposition (GHI → POA):** pvlib Perez anisotropic model.

**IAM:** ASHRAE / Martin-Ruiz / Fresnel (user choice).

**Temperature:** NOCT model (IEC 61215) with wind correction.

**Live data:** Open-Meteo forecast API (cloud, T, RH, radiation); AOD from
monthly MERRA-2 climatology scaled by relative humidity (Hänel 1976).
        """)


def _metric(col, label: str, value: str, sublabel: str):
    col.markdown(f"""
    <div class="metric-box">
      <div class="metric-val">{value}</div>
      <div class="metric-lbl">{label}</div>
      <div style="font-size:0.75rem;color:#666">{sublabel}</div>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
