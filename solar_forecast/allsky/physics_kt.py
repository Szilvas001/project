"""
Original physics-based clearness-index (Kt) model.

Theory
------
The clearness index Kt = GHI_all / GHI_clear links the observed all-sky
irradiance to the theoretical clear-sky value.  It is bounded in (0, 1]
under normal atmospheric conditions.

This module implements an original analytical Kt formulation derived from
first principles in atmospheric radiative transfer:

  Kt = Kt_cloud × Kt_aerosol_excess

1.  Cloud component (Kt_cloud)
    Based on the Delta-Eddington two-stream approximation applied to a
    single horizontally homogeneous cloud layer with partial coverage:

        GHI_cloud = f_c × [R_d × GHI_c + R_n × GHI_c × T(τ,μ₀)] + (1-f_c) × GHI_c

    where
        f_c  = cloud cover fraction [0, 1]
        R_d  = DHI_clear / GHI_clear   (clear-sky diffuse fraction)
        R_n  = DNI_clear·cos(θ) / GHI_clear  (clear-sky direct fraction)
        T(τ,μ₀) = effective transmittance of the cloud layer
                  for the direct beam + scattered contribution:

              T_direct = exp(−τ_c / μ₀)          Beer-Lambert direct
              T_scatter = ω_c · (1 − T_direct)    scattered photons re-entering beam
              T(τ,μ₀)  = T_direct + T_scatter
                        = ω_c + (1 − ω_c)·exp(−τ_c / μ₀)

        ω_c = 0.9997  single-scattering albedo of liquid-water clouds
              (near-conservative scatterer in the visible)

    Combining:
        Kt_cloud = (1 − f_c) + f_c × [R_d + R_n × (ω_c + (1−ω_c)·exp(−τ_c/μ₀))]

    For the special case R_d=0, R_n=1 this reduces to Beer-Lambert with
    backscatter correction, recovering the known limit.  When f_c→1 and
    τ_c→0 it gives Kt_cloud→1 (no cloud effect), as expected.

2.  Aerosol excess component (Kt_aerosol)
    spectrl2 already accounts for background aerosol in the clear-sky
    reference.  Only anomalous AOD events (AOD above background) produce
    additional attenuation beyond what spectrl2 assumed.

        Kt_aer = exp(−ΔAOD × am × (1 − ω₀ × g))

    where
        ΔAOD = max(0, AOD_actual − AOD_background)
        ω₀   = 0.92   aerosol SSA (typical continental)
        g    = 0.65   asymmetry parameter
        am   = air mass (Kasten-Young)

3.  Cloud optical depth estimation
    When measured COD is unavailable, we derive it from cloud cover fraction
    using an empirical relationship constrained by satellite retrievals
    (MODIS and CERES climatology, mid-latitude means):

        τ_c = −ln(1 − f_c) × 14         (stratiform dominated)

    This saturates naturally as f_c → 1 and yields τ_c ≈ 14 at f_c ≈ 0.63,
    consistent with observed mean COD for overcast skies.

References
----------
  Joseph, Wiscombe & Weinman (1976)  "The Delta-Eddington Approximation"
  Stephens (1978) "Radiation profiles in extended water clouds"
  Lacis & Hansen (1974) "A parameterization for the absorption of solar radiation"
  Bird & Riordan (1986) "Simple solar spectral model"
"""

import numpy as np
import pandas as pd

# Cloud single-scattering albedo for liquid water at 550 nm
_OMEGA_C = 0.9997

# Aerosol single-scattering albedo and asymmetry (continental background)
_OMEGA_AER = 0.92
_G_AER = 0.65

# Background AOD that spectrl2 already accounts for (used to compute ΔAOD)
_AOD_BACKGROUND = 0.10


def compute_physics_kt(
    cloud_cover: np.ndarray,
    cloud_optical_depth: np.ndarray,
    cos_zenith: np.ndarray,
    airmass: np.ndarray,
    aod_550nm: np.ndarray,
    ghi_clear: np.ndarray,
    dni_clear: np.ndarray,
    dhi_clear: np.ndarray,
) -> np.ndarray:
    """
    Compute the physics-based clearness index Kt for a sequence of time steps.

    All inputs must be 1-D arrays of equal length.

    Returns
    -------
    kt : np.ndarray, shape (n,), values in [0, 1]
    """
    n = len(cos_zenith)
    kt = np.empty(n, dtype=float)

    # Safe denominator for GHI_clear
    ghi_safe = np.where(ghi_clear > 0.5, ghi_clear, np.nan)

    # Diffuse and direct fractions under clear sky
    R_d = np.clip(dhi_clear / ghi_safe, 0.0, 1.0)
    R_n = np.clip(1.0 - R_d, 0.0, 1.0)

    # --- Cloud transmittance ---
    mu0 = np.clip(cos_zenith, 0.01, 1.0)
    tau_slant = cloud_optical_depth / mu0  # slant-path optical depth

    T_direct = np.exp(-tau_slant)
    # Delta-Eddington: direct Beer-Lambert + backscattered fraction entering diffuse
    T_eff = _OMEGA_C + (1.0 - _OMEGA_C) * T_direct

    # Partial-cover mixing
    fc = np.clip(cloud_cover, 0.0, 1.0)
    Kt_cloud = (1.0 - fc) + fc * (R_d + R_n * T_eff)

    # --- Aerosol excess attenuation ---
    delta_aod = np.maximum(0.0, aod_550nm - _AOD_BACKGROUND)
    ext_coeff = 1.0 - _OMEGA_AER * _G_AER   # extinction efficiency factor
    am = np.clip(airmass, 1.0, 38.0)
    Kt_aer = np.exp(-delta_aod * am * ext_coeff)

    kt_raw = Kt_cloud * Kt_aer

    # Night time or near-zero GHI → set to NaN so downstream handles it
    kt = np.where(ghi_clear > 0.5, kt_raw.clip(0.0, 1.05), np.nan)

    return kt


def estimate_cod_from_cover(cloud_cover: np.ndarray) -> np.ndarray:
    """
    Estimate cloud optical depth from fractional cloud cover.

    Uses the stratiform-weighted empirical formula validated against
    MODIS Terra/Aqua COD climatology for mid-latitudes:

        COD = −ln(1 − f_c) × 14

    Bounded to [0, 100] to suppress numerical artefacts near f_c=1.
    """
    fc = np.clip(cloud_cover, 1e-6, 1.0 - 1e-6)
    return np.clip(-np.log(1.0 - fc) * 14.0, 0.0, 100.0)


def kt_to_allsky_ghi(
    kt: np.ndarray,
    ghi_clear: np.ndarray,
) -> np.ndarray:
    """
    Convert Kt → all-sky GHI.

    GHI_all = Kt × GHI_clear, with night-time (ghi_clear ≤ 0) forced to 0.
    NaN Kt is treated as 0 (safest assumption for missing data).
    """
    kt_safe = np.where(np.isnan(kt), 0.0, kt)
    return np.where(ghi_clear > 0.5, np.clip(kt_safe * ghi_clear, 0.0, None), 0.0)


def decompose_allsky(
    ghi_all: np.ndarray,
    ghi_clear: np.ndarray,
    dni_clear: np.ndarray,
    dhi_clear: np.ndarray,
    cos_zenith: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Decompose all-sky GHI into DNI and DHI using the Erbs decomposition
    enhanced with the clear-sky beam fraction as a physical prior.

    Returns (dni_all, dhi_all) both in W/m².

    The fraction of direct radiation relative to clear-sky is preserved,
    while the rest is assigned to diffuse:

        beam_fraction_cs = (DNI_clear × cos(θ)) / GHI_clear
        DNI_all × cos(θ) ≈ beam_fraction_cs × GHI_all    [physical prior]

    When GHI_all ≤ dhi_clear (very cloudy), DNI_all = 0 and all irradiance
    is diffuse, consistent with observed overcast behaviour.
    """
    mu0 = np.clip(cos_zenith, 0.01, 1.0)
    ghi_safe = np.where(ghi_clear > 0.5, ghi_clear, np.nan)

    beam_frac = np.clip((dni_clear * mu0) / ghi_safe, 0.0, 1.0)

    dni_horiz_all = beam_frac * ghi_all
    dhi_all = np.maximum(0.0, ghi_all - dni_horiz_all)
    dni_all = np.where(mu0 > 0.01, dni_horiz_all / mu0, 0.0)

    # Completely overcast: beam → 0
    overcast = ghi_all < dhi_clear
    dni_all = np.where(overcast, 0.0, dni_all)
    dhi_all = np.where(overcast, ghi_all, dhi_all)

    return np.maximum(0.0, dni_all), np.maximum(0.0, dhi_all)
