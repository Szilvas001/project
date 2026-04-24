"""
Solar Forecast — physics-based + AI hybrid PV production forecasting system.

Modules:
    data_ingestion  — CAMS historical download, Open-Meteo live, PostgreSQL storage
    clearsky        — pvlib spectrl2-based clear-sky irradiance
    allsky          — physics Kt model + XGBoost AI trainer + hybrid combiner
    production      — spectral response, IAM, power conversion
    physics         — aerosol optics: Ångström, Hänel, SSA/GG
    dashboard       — Streamlit SaaS dashboard (Data / Train / Live)
"""

__version__ = "2.0.0"
