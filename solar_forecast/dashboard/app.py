"""
Solar Forecast — SaaS Dashboard

Three-module layout:
  📥 Data        — Download CAMS historical data, view atmospheric variables
  🤖 Train       — Train / retrain the XGBoost Kt model, evaluate metrics
  ☀️  Live Forecast — Real-time 7-day production estimate with live curve drawing

UTC / Timezone:
  All internal data (CAMS, Open-Meteo, clearsky) is stored and processed in UTC.
  The Live Forecast tab displays times in the user's chosen local timezone.
  CAMS returns UTC; Open-Meteo is also requested in UTC mode.

Run:
    streamlit run solar_forecast/dashboard/app.py
"""

from __future__ import annotations

import io
import logging
import time as _time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

logging.basicConfig(level=logging.WARNING)

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Solar Forecast",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-title { font-size:2rem; font-weight:700; color:#F4A503; }
    .metric-box { background:#1E1E2E; border-radius:12px; padding:16px;
                  text-align:center; }
    .metric-val { font-size:2rem; font-weight:800; color:#F4A503; }
    .metric-lbl { font-size:0.85rem; color:#888; }
    .tab-note   { font-size:0.80rem; color:#aaa; margin-top:4px; }
    footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# Known timezones (user-selectable)
# ══════════════════════════════════════════════════════════════════════════
_TZ_OPTIONS = [
    "UTC",
    "Europe/Budapest",
    "Europe/Vienna",
    "Europe/Berlin",
    "Europe/London",
    "Europe/Paris",
    "Europe/Warsaw",
    "Europe/Bucharest",
    "Europe/Athens",
    "US/Eastern",
    "US/Central",
    "US/Mountain",
    "US/Pacific",
    "Asia/Tokyo",
    "Asia/Shanghai",
    "Asia/Kolkata",
    "Australia/Sydney",
]

_IAM_OPTIONS  = ["ashrae", "martin_ruiz", "fresnel"]
_TECH_OPTIONS = {
    "mono_si": "Mono-Si (standard c-Si) — default",
    "poly_si": "Poly-Si (multi-crystalline)",
    "cdte":    "CdTe (thin-film)",
    "cigs":    "CIGS / CIS (thin-film)",
    "hit":     "HIT / HJT (heterojunction)",
    "custom":  "Custom (upload CSV)",
}


# ══════════════════════════════════════════════════════════════════════════
# Sidebar — system configuration (shared across all tabs)
# ══════════════════════════════════════════════════════════════════════════

def _sidebar() -> dict:
    st.sidebar.markdown("## ☀️ System Configuration")

    # ── Location ─────────────────────────────────────────────────────────
    loc_mode = st.sidebar.radio(
        "Location input", ["City name", "GPS coordinates"], horizontal=True
    )
    if loc_mode == "City name":
        city   = st.sidebar.text_input("City", value="Budapest")
        lat_in = lon_in = None
    else:
        city   = None
        c1, c2 = st.sidebar.columns(2)
        lat_in = c1.number_input("Latitude",  -90.0,  90.0,  47.498, 0.001)
        lon_in = c2.number_input("Longitude", -180.0, 180.0, 19.040, 0.001)

    capacity_kw = st.sidebar.number_input(
        "System size (kW)", min_value=0.1, max_value=100000.0, value=10.0, step=0.5,
    )
    altitude = st.sidebar.number_input(
        "Altitude (m)", min_value=0, max_value=5000, value=120, step=10,
    )

    # ── Panel orientation ──────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Panel orientation**")
    orient_mode = st.sidebar.radio(
        "Tilt/Azimuth mode",
        ["Auto (from latitude)", "Standard profile", "Manual input"],
        horizontal=False,
    )

    tilt_val    = 30.0
    azimuth_val = 180.0

    if orient_mode == "Manual input":
        tilt_val    = st.sidebar.slider("Tilt (°)",           0,  90, 30)
        azimuth_val = st.sidebar.slider("Azimuth (°, 180=S)", 0, 360, 180)
    elif orient_mode == "Standard profile":
        profile = st.sidebar.selectbox(
            "Profile",
            ["Residential roof (30°)", "Flat roof (12°)", "Optimised (lat×0.76)"],
        )
        ref_lat = lat_in if loc_mode == "GPS coordinates" else 47.5
        tilt_val, azimuth_val = _profile_tilt_az(profile, ref_lat)
    # else: auto — will be set after geocode in run_forecast

    # ── Module technology & SR ────────────────────────────────────────
    st.sidebar.markdown("---")
    tech_key = st.sidebar.selectbox(
        "Cell technology",
        list(_TECH_OPTIONS.keys()),
        format_func=lambda k: _TECH_OPTIONS[k],
    )
    sr_bytes: bytes | None = None
    if tech_key == "custom":
        sr_file = st.sidebar.file_uploader(
            "SR curve CSV (wavelength_nm, sr_value)", type="csv"
        )
        sr_bytes = sr_file.read() if sr_file else None

    # ── IAM model ─────────────────────────────────────────────────────
    iam_type = st.sidebar.selectbox("IAM model", _IAM_OPTIONS)

    # ── Timezone ──────────────────────────────────────────────────────
    st.sidebar.markdown("---")
    tz = st.sidebar.selectbox("Display timezone", _TZ_OPTIONS, index=1)

    # ── Model path ────────────────────────────────────────────────────
    kt_model_path = st.sidebar.text_input("Kt model path", "models/kt_xgb.joblib")

    return {
        "loc_mode":     loc_mode,
        "city":         city,
        "lat_in":       lat_in,
        "lon_in":       lon_in,
        "capacity_kw":  capacity_kw,
        "altitude":     altitude,
        "orient_mode":  orient_mode,
        "tilt":         tilt_val,
        "azimuth":      azimuth_val,
        "tech_key":     tech_key,
        "sr_bytes":     sr_bytes,
        "iam_type":     iam_type,
        "timezone":     tz,
        "kt_model_path": kt_model_path,
    }


def _profile_tilt_az(profile: str, lat: float) -> tuple[float, float]:
    az = 180.0 if lat >= 0 else 0.0
    if "Flat" in profile:
        return 12.0, az
    elif "Optimised" in profile:
        return round(abs(lat) * 0.76, 1), az
    else:
        return 30.0, az


# ══════════════════════════════════════════════════════════════════════════
# Helper — resolve location
# ══════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def _resolve_location(loc_mode: str, city: str, lat_in, lon_in) -> tuple[float, float, str, float]:
    """Return (lat, lon, display_name, altitude_m)."""
    if loc_mode == "City name":
        from solar_forecast.data_ingestion.openmeteo_live import OpenMeteoClient
        client = OpenMeteoClient({"openmeteo": {}})
        lat, lon, name, elev = client.geocode(city)
        return lat, lon, name, elev or 0.0
    else:
        return float(lat_in), float(lon_in), f"{lat_in:.3f}°N {lon_in:.3f}°E", 0.0


# ══════════════════════════════════════════════════════════════════════════
# Tab 1 — Data Download
# ══════════════════════════════════════════════════════════════════════════

def tab_data(cfg_ui: dict) -> None:
    st.header("📥 Data — CAMS Historical Download")
    st.markdown(
        "Download CAMS EAC4 atmospheric reanalysis (AOD, ozone, water vapour, "
        "PM, SSA, BLH, …) and CAMS solar radiation service (hourly GHI truth) "
        "to PostgreSQL for model training."
    )

    col1, col2 = st.columns(2)
    start_date = col1.date_input("Start date", value=datetime(2022, 1, 1))
    end_date   = col2.date_input("End date",   value=datetime(2023, 12, 31))

    with st.expander("CAMS API settings"):
        api_key = st.text_input("CAMS API key (UID:key)", type="password",
                                help="From https://ads.atmosphere.copernicus.eu")
        fetch_spectral = st.checkbox("Download multi-wavelength AOD (469/670/865/1240 nm)", True)
        fetch_species  = st.checkbox("Download speciated AOD (dust/BC/OM/SS/SO4)", True)
        fetch_optional = st.checkbox("Download PM2.5, PM10, CO, NO2", True)

    with st.expander("Database connection"):
        db_host = st.text_input("Host",     "localhost")
        db_port = st.number_input("Port",   value=5432, step=1)
        db_name = st.text_input("Database", "solar_forecast")
        db_user = st.text_input("User",     "postgres")
        db_pass = st.text_input("Password", type="password")

    if st.button("▶ Start CAMS download", type="primary"):
        try:
            lat, lon, name, alt = _resolve_location(
                cfg_ui["loc_mode"], cfg_ui["city"],
                cfg_ui["lat_in"], cfg_ui["lon_in"],
            )
        except Exception as e:
            st.error(f"Location error: {e}")
            return

        cfg = _make_cfg(lat, lon, alt, cfg_ui["capacity_kw"],
                        cfg_ui["tilt"], cfg_ui["azimuth"])
        cfg["cams"]["api_key"]          = api_key
        cfg["cams"]["fetch_spectral_aod"] = fetch_spectral
        cfg["cams"]["fetch_species_aod"]  = fetch_species
        cfg["cams"]["fetch_optional"]     = fetch_optional
        cfg["database"] = {
            "host": db_host, "port": db_port,
            "name": db_name, "user": db_user, "password": db_pass,
        }

        from solar_forecast.data_ingestion.db_manager import DBManager
        from solar_forecast.data_ingestion.cams_loader import CamsLoader

        try:
            db = DBManager(cfg)
            loader = CamsLoader(cfg, db)
            progress = st.progress(0, text="Downloading CAMS data…")
            with st.spinner("Running backfill…"):
                result = loader.run_backfill(
                    str(start_date), str(end_date)
                )
            progress.progress(100, text="Done!")
            st.success(
                f"Downloaded: {result['cams_atmo']} atmospheric rows, "
                f"{result['cams_radiation']} radiation rows."
            )
        except Exception as exc:
            st.error(f"Download failed: {exc}")

    st.divider()
    st.subheader("CAMS variable reference")
    st.markdown("""
| Variable | Source | Unit | Used for |
|---|---|---|---|
| AOD 550nm | EAC4 | — | Aerosol extinction |
| AOD 469/670/865/1240nm | EAC4 | — | Ångström exponent α1, α2 |
| Dust/BC/OM/SS/SO4 AOD | EAC4 | — | SSA, asymmetry factor |
| Total ozone | EAC4 | DU | UV absorption |
| Precipitable water | EAC4 | cm | Water vapour absorption |
| Surface pressure | EAC4 | hPa | Air mass correction |
| Total cloud cover | EAC4 | fraction | Cloud optical depth |
| BLH | EAC4 | m | Mixing depth feature |
| PM2.5, PM10 | EAC4 | µg/m³ | Pollution proxy |
| GHI, DNI, DHI (all-sky) | CAMS Radiation | W/m² | Training target Kt |
| GHI_clear (McClear) | CAMS Radiation | W/m² | Clear-sky reference |
""")


# ══════════════════════════════════════════════════════════════════════════
# Tab 2 — Model Training
# ══════════════════════════════════════════════════════════════════════════

def tab_train(cfg_ui: dict) -> None:
    st.header("🤖 Training — XGBoost Kt Model")
    st.markdown(
        "Train the XGBoost clearness-index model on CAMS historical data. "
        "The model uses **physics Kt** as a feature and learns residual corrections "
        "from SSA, GG, PM, BLH, cloud layer heights, and cyclical time encoding."
    )

    col1, col2 = st.columns(2)
    train_start = col1.date_input("Training start", value=datetime(2022, 1, 1))
    train_end   = col2.date_input("Training end",   value=datetime(2023, 12, 31))

    with st.expander("Training options"):
        physics_weight = st.slider("Physics weight α (0=full AI, 1=full physics)", 0.0, 1.0, 0.40, 0.05)
        n_cv_folds = st.selectbox("Cross-validation folds", [0, 3, 5, 10], index=1,
                                  help="0 = no CV, just train/val split")
        min_samples = st.number_input("Min training samples", value=500, step=100)

    with st.expander("Database connection"):
        db_host = st.text_input("Host",     "localhost",        key="train_db_host")
        db_port = st.number_input("Port",   value=5432, step=1, key="train_db_port")
        db_name = st.text_input("Database", "solar_forecast",   key="train_db_name")
        db_user = st.text_input("User",     "postgres",         key="train_db_user")
        db_pass = st.text_input("Password", type="password",    key="train_db_pass")

    kt_save_path = st.text_input("Model save path", cfg_ui["kt_model_path"])

    if st.button("▶ Start training", type="primary"):
        try:
            lat, lon, name, alt = _resolve_location(
                cfg_ui["loc_mode"], cfg_ui["city"],
                cfg_ui["lat_in"], cfg_ui["lon_in"],
            )
        except Exception as e:
            st.error(f"Location error: {e}")
            return

        cfg = _make_cfg(lat, lon, alt, cfg_ui["capacity_kw"],
                        cfg_ui["tilt"], cfg_ui["azimuth"])
        cfg["model"]["physics_weight"]   = physics_weight
        cfg["model"]["min_train_samples"] = min_samples
        cfg["model"]["kt_model_path"]     = kt_save_path
        cfg["database"] = {
            "host": db_host, "port": db_port,
            "name": db_name, "user": db_user, "password": db_pass,
        }

        progress_bar = st.progress(0, "Loading data…")
        status_text  = st.empty()

        try:
            from solar_forecast.data_ingestion.db_manager import DBManager
            from solar_forecast.data_ingestion.cams_loader import CamsLoader
            from solar_forecast.clearsky.spectrl2_model import compute_clearsky_from_weather
            from solar_forecast.allsky.ai_trainer import KtTrainer

            db = DBManager(cfg)
            loader = CamsLoader(cfg, db)

            status_text.text("Loading atmospheric data from DB…")
            df_atmo, df_rad = loader.load_training_data(
                str(train_start), str(train_end)
            )

            if df_atmo.empty or df_rad.empty:
                st.warning("No data found in DB for the selected range. "
                           "Download CAMS data first (Data tab).")
                return

            progress_bar.progress(30, "Computing clear-sky…")
            status_text.text("Computing clear-sky reference (spectrl2)…")
            df_cs = compute_clearsky_from_weather(
                df_atmo, lat, lon, alt,
                cfg_ui["tilt"], cfg_ui["azimuth"],
            )

            progress_bar.progress(50, "Building training set…")
            trainer = KtTrainer(cfg)
            df_train = trainer.build_training_set(df_atmo, df_rad, df_cs)
            status_text.text(f"Training on {len(df_train)} samples…")

            progress_bar.progress(60, "Training XGBoost…")
            metrics = trainer.train(df_train, n_cv_folds=n_cv_folds)
            trainer.save(kt_save_path)

            progress_bar.progress(100, "Done!")
            status_text.empty()

            # ── Results display ───────────────────────────────────────────
            st.success("Model trained and saved.")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("MAE",  f"{metrics['mae']:.4f}")
            m2.metric("RMSE", f"{metrics['rmse']:.4f}")
            m3.metric("R²",   f"{metrics['r2']:.4f}")
            m4.metric("Best iter", str(metrics["best_iteration"]))

            if f"cv_{n_cv_folds}fold_rmse_mean" in metrics:
                st.info(
                    f"{n_cv_folds}-fold CV RMSE: "
                    f"{metrics[f'cv_{n_cv_folds}fold_rmse_mean']:.4f} "
                    f"± {metrics[f'cv_{n_cv_folds}fold_rmse_std']:.4f}"
                )

            # Feature importance chart
            model_obj = trainer.pipeline["model"]
            feat_imp = model_obj.feature_importances_
            from solar_forecast.allsky.ai_trainer import _FEATURE_COLS
            fig = go.Figure(go.Bar(
                x=feat_imp,
                y=_FEATURE_COLS,
                orientation="h",
                marker_color="#F4A503",
            ))
            fig.update_layout(
                title="Feature Importances",
                xaxis_title="Importance",
                height=500,
                margin=dict(l=180),
                paper_bgcolor="#0E1117",
                plot_bgcolor="#0E1117",
                font_color="white",
            )
            st.plotly_chart(fig, use_container_width=True)

        except Exception as exc:
            st.error(f"Training failed: {exc}")
            import traceback
            st.code(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════
# Tab 3 — Live Forecast (SaaS)
# ══════════════════════════════════════════════════════════════════════════

def tab_live(cfg_ui: dict) -> None:
    st.header("☀️ Live Forecast — 7-Day PV Production")

    # Auto-refresh toggle
    col_r1, col_r2 = st.columns([3, 1])
    auto_refresh  = col_r2.checkbox("Auto-refresh (30 min)", value=False)
    if auto_refresh:
        col_r2.markdown('<p class="tab-note">Next refresh in 30 min</p>',
                        unsafe_allow_html=True)
        st_autorefresh_available = False
        try:
            from streamlit_autorefresh import st_autorefresh
            st_autorefresh(interval=30 * 60 * 1000, key="live_refresh")
            st_autorefresh_available = True
        except ImportError:
            col_r2.caption("Install streamlit-autorefresh for auto-refresh")

    run_btn = col_r1.button("▶ Run forecast now", type="primary", use_container_width=True)

    if not run_btn:
        st.info("Configure the system in the sidebar, then click 'Run forecast now'.")
        return

    # Resolve location
    try:
        with st.spinner("Resolving location…"):
            lat, lon, name, alt_geo = _resolve_location(
                cfg_ui["loc_mode"], cfg_ui["city"],
                cfg_ui["lat_in"], cfg_ui["lon_in"],
            )
    except Exception as e:
        st.error(f"Location error: {e}")
        return

    alt = cfg_ui["altitude"] if cfg_ui["altitude"] > 0 else (alt_geo or 0.0)
    tilt, azimuth = cfg_ui["tilt"], cfg_ui["azimuth"]

    # Auto-compute tilt from latitude if orientation is "Auto"
    if cfg_ui["orient_mode"] == "Auto (from latitude)":
        tilt    = round(abs(lat) * 0.76, 1)
        azimuth = 180.0 if lat >= 0 else 0.0

    tz = cfg_ui["timezone"]

    st.markdown(
        f"**Location:** {name} | "
        f"**Lat/Lon:** {lat:.3f}°, {lon:.3f}° | "
        f"**Alt:** {alt:.0f} m | "
        f"**Tilt:** {tilt}° | **Azimuth:** {azimuth}° | "
        f"**Timezone:** {tz}"
    )

    # SR curve
    sr_csv_path = None
    if cfg_ui["sr_bytes"]:
        tmp = Path("/tmp/user_sr_dash.csv")
        tmp.write_bytes(cfg_ui["sr_bytes"])
        sr_csv_path = str(tmp)

    cfg = _make_cfg(lat, lon, alt, cfg_ui["capacity_kw"], tilt, azimuth)
    cfg["model"]["kt_model_path"] = cfg_ui["kt_model_path"]

    try:
        with st.spinner("Fetching Open-Meteo weather (UTC)…"):
            hourly, daily_df = _run_full_forecast(
                lat=lat, lon=lon, altitude=alt,
                capacity_kw=cfg_ui["capacity_kw"],
                tilt=tilt, azimuth=azimuth,
                tech_key=cfg_ui["tech_key"],
                sr_csv_path=sr_csv_path,
                iam_type=cfg_ui["iam_type"],
                kt_model_path=cfg_ui["kt_model_path"],
                cfg=cfg,
                tz=tz,
            )
    except Exception as exc:
        st.error(f"Forecast failed: {exc}")
        import traceback
        st.code(traceback.format_exc())
        return

    # ── Convert UTC index to display timezone ─────────────────────────────
    hourly_local = hourly.copy()
    hourly_local.index = hourly_local.index.tz_convert(tz)

    # ── KPI metrics ───────────────────────────────────────────────────────
    peak_kw     = hourly["power_kw"].max()
    total_7d    = hourly["power_kw"].sum()  # sum of hourly kW ≈ kWh
    cf          = hourly["power_kw"].mean() / cfg_ui["capacity_kw"] * 100
    avg_kt      = hourly["kt"].mean()

    k1, k2, k3, k4 = st.columns(4)
    _kpi(k1, f"{peak_kw:.2f} kW",    "Peak Power")
    _kpi(k2, f"{total_7d:.1f} kWh",  "7-Day Energy")
    _kpi(k3, f"{cf:.1f} %",          "Capacity Factor")
    _kpi(k4, f"{avg_kt:.3f}",        "Avg Clearness Kt")

    # ── Hourly production chart ───────────────────────────────────────────
    st.subheader("Hourly Production Forecast")
    fig = _production_chart(hourly_local, cfg_ui["capacity_kw"], tz)
    st.plotly_chart(fig, use_container_width=True)

    # ── Daily summary table ───────────────────────────────────────────────
    st.subheader("Daily Summary")
    st.dataframe(
        daily_df.set_index("Date").style.format("{:.2f}"),
        use_container_width=True,
    )

    # ── Atmospheric detail chart ──────────────────────────────────────────
    with st.expander("Atmospheric variables (UTC)"):
        _atmo_chart(hourly_local)

    # ── Spectral mismatch ─────────────────────────────────────────────────
    if "mm" in hourly.columns:
        with st.expander("Spectral mismatch factor MM"):
            fig_mm = go.Figure()
            fig_mm.add_trace(go.Scatter(
                x=hourly_local.index,
                y=hourly_local["mm"],
                name="Mismatch MM",
                line=dict(color="#00BFFF", width=1.5),
            ))
            fig_mm.add_hline(y=1.0, line_dash="dot", line_color="gray",
                             annotation_text="AM1.5G reference")
            fig_mm.update_layout(
                title="Spectral Mismatch Factor",
                yaxis_title="MM",
                yaxis_range=[0.85, 1.15],
                paper_bgcolor="#0E1117",
                plot_bgcolor="#0E1117",
                font_color="white",
                height=300,
            )
            st.plotly_chart(fig_mm, use_container_width=True)

    # ── Raw data download ─────────────────────────────────────────────────
    st.divider()
    csv_bytes = hourly.reset_index().to_csv(index=False).encode()
    st.download_button(
        "⬇ Download forecast CSV (UTC)",
        data=csv_bytes,
        file_name=f"solar_forecast_{lat:.2f}_{lon:.2f}.csv",
        mime="text/csv",
    )


# ══════════════════════════════════════════════════════════════════════════
# Forecast pipeline
# ══════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=1800, show_spinner=False)
def _run_full_forecast(
    lat, lon, altitude, capacity_kw, tilt, azimuth,
    tech_key, sr_csv_path, iam_type, kt_model_path,
    cfg: dict, tz: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full 7-day forecast pipeline.  Results cached for 30 min.

    UTC throughout — caller converts index to display timezone.
    """
    from solar_forecast.data_ingestion.openmeteo_live import OpenMeteoClient
    from solar_forecast.clearsky.spectrl2_model import compute_clearsky_from_weather
    from solar_forecast.allsky.hybrid_model import AllSkyModel
    from solar_forecast.production.pv_output import PVOutputModel

    client  = OpenMeteoClient(cfg)
    weather = client.get_forecast(lat, lon, days=7, tilt=tilt, azimuth=azimuth)
    # weather.index is UTC-aware

    cs_df = compute_clearsky_from_weather(
        weather, lat, lon, altitude, tilt, azimuth,
        return_spectra=False,   # fast mode for live dashboard
    )

    allsky = AllSkyModel(cfg)
    allsky.load_kt_model()
    allsky_df = allsky.forecast(weather.index, weather, cs_df)

    pv = PVOutputModel(
        cfg,
        technology=tech_key if tech_key != "custom" else "mono_si",
        sr_csv=sr_csv_path,
        iam_model=iam_type,
    )
    prod_df = pv.run_from_live(allsky_df, weather, lat, lon, altitude)

    hourly = pd.concat([
        prod_df,
        allsky_df[["ghi", "kt", "ghi_clear"]],
        weather[["cloud_cover", "aod_550nm", "precipitable_water",
                 "temperature"]].rename(columns={"temperature": "t_air"}),
    ], axis=1)
    hourly.index = hourly.index.tz_convert("UTC")   # ensure UTC

    # Daily aggregates
    daily_kwh  = hourly["power_kw"].resample("D").sum()
    daily_peak = hourly["power_kw"].resample("D").max()
    daily_avg  = hourly["power_kw"].resample("D").mean()

    # Convert daily index to display timezone for table
    daily_idx = daily_kwh.index.tz_convert(tz)
    daily_df = pd.DataFrame({
        "Date":             daily_idx.strftime("%Y-%m-%d"),
        "Energy (kWh)":     daily_kwh.round(2).values,
        "Peak Power (kW)":  daily_peak.round(2).values,
        "Avg Power (kW)":   daily_avg.round(2).values,
    })

    return hourly, daily_df


# ══════════════════════════════════════════════════════════════════════════
# Chart builders
# ══════════════════════════════════════════════════════════════════════════

def _production_chart(hourly_local: pd.DataFrame, capacity_kw: float, tz: str) -> go.Figure:
    fig = go.Figure()

    # Clear-sky GHI reference
    if "ghi_clear" in hourly_local.columns:
        fig.add_trace(go.Scatter(
            x=hourly_local.index, y=hourly_local["ghi_clear"].clip(0) / 1000,
            name="Clear-sky GHI (kW/m²)",
            line=dict(color="#FFD700", dash="dot", width=1),
            opacity=0.5,
        ))

    # All-sky GHI
    if "ghi" in hourly_local.columns:
        fig.add_trace(go.Scatter(
            x=hourly_local.index, y=hourly_local["ghi"].clip(0) / 1000,
            name="GHI (kW/m²)",
            line=dict(color="#FFA500", width=1.5),
        ))

    # PV production (filled)
    fig.add_trace(go.Scatter(
        x=hourly_local.index, y=hourly_local["power_kw"].clip(0),
        name="PV Power (kW)",
        line=dict(color="#00CC96", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(0,204,150,0.15)",
    ))

    # Capacity reference line
    fig.add_hline(y=capacity_kw, line_dash="dash", line_color="gray",
                  annotation_text=f"Capacity {capacity_kw:.1f} kW", opacity=0.5)

    fig.update_layout(
        title=f"7-Day Production Forecast — {tz}",
        xaxis_title=f"Time ({tz})",
        yaxis_title="Power / GHI",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02, x=0),
        paper_bgcolor="#0E1117",
        plot_bgcolor="#1E1E2E",
        font_color="white",
        height=420,
    )
    return fig


def _atmo_chart(hourly_local: pd.DataFrame) -> None:
    fig = go.Figure()
    if "cloud_cover" in hourly_local.columns:
        fig.add_trace(go.Scatter(
            x=hourly_local.index,
            y=hourly_local["cloud_cover"] * 100,
            name="Cloud cover (%)",
            line=dict(color="#7EC8E3"),
        ))
    if "aod_550nm" in hourly_local.columns:
        fig.add_trace(go.Scatter(
            x=hourly_local.index,
            y=hourly_local["aod_550nm"],
            name="AOD 550nm",
            yaxis="y2",
            line=dict(color="#FF7F50", dash="dot"),
        ))
    fig.update_layout(
        yaxis=dict(title="Cloud cover (%)", range=[0, 105]),
        yaxis2=dict(title="AOD", overlaying="y", side="right", range=[0, 1.5]),
        paper_bgcolor="#0E1117",
        plot_bgcolor="#1E1E2E",
        font_color="white",
        height=280,
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, use_container_width=True)


def _kpi(col, value: str, label: str) -> None:
    col.markdown(
        f'<div class="metric-box">'
        f'<div class="metric-val">{value}</div>'
        f'<div class="metric-lbl">{label}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════
# Config factory
# ══════════════════════════════════════════════════════════════════════════

def _make_cfg(
    lat: float, lon: float, altitude: float = 0.0,
    capacity_kw: float = 10.0, tilt: float = 30.0, azimuth: float = 180.0,
) -> dict:
    return {
        "location": {"lat": lat, "lon": lon, "altitude": altitude},
        "system": {
            "capacity_kw":         capacity_kw,
            "tilt":                tilt,
            "azimuth":             azimuth,
            "module_efficiency":   0.205,
            "temperature_coefficient": -0.0040,
            "noct":                44.0,
            "ground_albedo":       0.20,
            "inverter_efficiency": 0.97,
            "wiring_loss":         0.02,
            "soiling_loss":        0.02,
        },
        "model": {
            "kt_model_path":     "models/kt_xgb.joblib",
            "physics_weight":    0.40,
            "min_train_samples": 500,
        },
        "openmeteo": {
            "forecast_url":   "https://api.open-meteo.com/v1/forecast",
            "historical_url": "https://archive-api.open-meteo.com/v1/archive",
            "cache_expire_hours": 1,
        },
        "cams":     {},
        "database": {},
    }


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main() -> None:
    st.markdown('<div class="main-title">☀️ Solar Forecast</div>', unsafe_allow_html=True)
    st.markdown(
        "Physics + AI hybrid PV production forecasting | "
        "CAMS EAC4 + Open-Meteo | spectrl2 clear-sky | XGBoost Kt"
    )

    cfg_ui = _sidebar()

    tab1, tab2, tab3 = st.tabs(["📥 Data", "🤖 Train", "☀️ Live Forecast"])

    with tab1:
        tab_data(cfg_ui)

    with tab2:
        tab_train(cfg_ui)

    with tab3:
        tab_live(cfg_ui)


if __name__ == "__main__":
    main()
