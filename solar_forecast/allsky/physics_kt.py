"""
Physics-based clearness-index (Kt) model.

Theory
------
The clearness index Kt = GHI_all / GHI_clear links the observed all-sky
irradiance to the theoretical clear-sky value.  It is bounded in (0, 1]
under normal atmospheric conditions.

This module implements an analytical Kt formulation derived from
atmospheric radiative transfer:

  Kt = Kt_cloud × Kt_aerosol_excess

1.  Cloud component (Delta-Eddington two-stream approximation)

    For a single cloud layer with optical depth τ_c, fractional cover f_c,
    and solar zenith angle θ:

        T_direct  = exp(−τ_c / μ₀)                      Beer-Lambert
        T_scatter = ω_c × (1 − T_direct)                 backscatter
        T_eff     = T_direct + T_scatter
                  = ω_c + (1 − ω_c) × exp(−τ_c / μ₀)

        Kt_cloud = (1 − f_c) + f_c × [R_d + R_n × T_eff]

    where R_d = DHI_clear/GHI_clear, R_n = 1 − R_d.

2.  Aerosol excess attenuation

    spectrl2 accounts for a background AOD in the clear-sky reference.
    Only anomalous AOD beyond the background causes additional extinction:

        Kt_aer = exp(−ΔAOD × am × (1 − ω₀ × g))

    where ω₀ = single-scattering albedo, g = asymmetry parameter.
    These are now taken directly from CAMS speciated AOD data (not fixed).

3.  Cloud optical depth from cloud cover (fallback)

    When COD is not measured:
        τ_c = −ln(1 − f_c) × 14   (Stephens 1978)

References
----------
  Joseph, Wiscombe & Weinman (1976) "The Delta-Eddington Approximation"
  Stephens (1978) "Radiation profiles in extended water clouds"
  Hänel (1976) "Hygroscopic growth"
  Shettle & Fenn (1979) "Models for aerosols"
"""

from __future__ import annotations

import numpy as np

# Cloud single-scattering albedo for liquid water at 550 nm (near-conservative)
_OMEGA_C = 0.9997

# Continental background AOD that spectrl2 assumes (clear-sky reference)
_AOD_BACKGROUND = 0.10

# Defaults when CAMS SSA/GG not available
_OMEGA_AER_DEFAULT = 0.92
_G_AER_DEFAULT     = 0.65


def compute_physics_kt(
    cloud_cover:          np.ndarray,
    cloud_optical_depth:  np.ndarray,
    cos_zenith:           np.ndarray,
    airmass:              np.ndarray,
    aod_550nm:            np.ndarray,
    ghi_clear:            np.ndarray,
    dni_clear:            np.ndarray,
    dhi_clear:            np.ndarray,
    ssa:                  np.ndarray | float = _OMEGA_AER_DEFAULT,
    asymmetry:            np.ndarray | float = _G_AER_DEFAULT,
) -> np.ndarray:
    """
    Compute the physics-based clearness index Kt.

    Parameters
    ----------
    cloud_cover           : Fractional cloud cover [0, 1]
    cloud_optical_depth   : Cloud optical depth [0, ∞)
    cos_zenith            : cos(solar zenith) [0, 1]
    airmass               : Relative air mass
    aod_550nm             : Aerosol optical depth at 550 nm
    ghi_clear, dni_clear, dhi_clear : Clear-sky irradiance components (W/m²)
    ssa                   : Aerosol single-scattering albedo ω₀ (CAMS data or default)
    asymmetry             : Aerosol asymmetry parameter g

    Returns
    -------
    kt : np.ndarray, values in [0, 1.05]; NaN for night/sub-horizon
    """
    n    = len(cos_zenith)
    ghi_safe = np.where(ghi_clear > 0.5, ghi_clear, np.nan)

    # ── Diffuse/direct fraction under clear sky ───────────────────────────
    R_d = np.clip(dhi_clear / ghi_safe, 0.0, 1.0)
    R_n = np.clip(1.0 - R_d, 0.0, 1.0)

    # ── Cloud transmittance (Delta-Eddington) ─────────────────────────────
    mu0       = np.clip(cos_zenith, 0.01, 1.0)
    tau_slant = cloud_optical_depth / mu0

    T_direct  = np.exp(-tau_slant)
    T_eff     = _OMEGA_C + (1.0 - _OMEGA_C) * T_direct   # direct + scattered

    fc = np.clip(cloud_cover, 0.0, 1.0)
    Kt_cloud = (1.0 - fc) + fc * (R_d + R_n * T_eff)

    # ── Aerosol excess attenuation ────────────────────────────────────────
    ssa_arr = np.asarray(ssa, dtype=float)
    g_arr   = np.asarray(asymmetry, dtype=float)
    if ssa_arr.ndim == 0:
        ssa_arr = np.full(n, float(ssa_arr))
    if g_arr.ndim == 0:
        g_arr = np.full(n, float(g_arr))

    ssa_arr = np.clip(ssa_arr, 0.50, 1.00)
    g_arr   = np.clip(g_arr,   0.30, 0.90)

    delta_aod = np.maximum(0.0, aod_550nm - _AOD_BACKGROUND)
    # Extinction efficiency (single-scatter + forward scatter correction)
    ext_eff   = 1.0 - ssa_arr * g_arr
    am        = np.clip(airmass, 1.0, 38.0)
    Kt_aer    = np.exp(-delta_aod * am * ext_eff)

    kt_raw = Kt_cloud * Kt_aer

    # Night / below-horizon → NaN
    kt = np.where(ghi_clear > 0.5, np.clip(kt_raw, 0.0, 1.05), np.nan)

    return kt


def estimate_cod_from_cover(cloud_cover: np.ndarray) -> np.ndarray:
    """
    Estimate cloud optical depth from fractional cloud cover.

    Stratiform empirical formula (Stephens 1978, MODIS-constrained):

        COD = −ln(1 − f_c) × 14

    Saturates naturally as f_c → 1; yields COD ≈ 14 at f_c ≈ 0.63.
    """
    fc = np.clip(np.asarray(cloud_cover, dtype=float), 0.0, 0.9999)
    return (-np.log(1.0 - fc) * 14.0).clip(0, 300)


def kt_to_allsky_ghi(kt: np.ndarray, ghi_clear: np.ndarray) -> np.ndarray:
    """Recover all-sky GHI from Kt and clear-sky reference."""
    return np.where(np.isfinite(kt), np.clip(kt * ghi_clear, 0.0, None), 0.0)


def decompose_allsky(
    ghi_all:   np.ndarray,
    ghi_clear: np.ndarray,
    dni_clear: np.ndarray,
    dhi_clear: np.ndarray,
    cos_zenith: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Decompose all-sky GHI into DNI and DHI using the Erbs decomposition
    constrained by the clear-sky ratio.

    The clear-sky Kt ratio is applied proportionally to DNI_clear and DHI_clear.
    Under heavy overcast (Kt → 0) the result approaches full diffuse.

    Returns
    -------
    (dni, dhi) in W/m²
    """
    ghi_safe = np.where(ghi_clear > 0.5, ghi_clear, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        kt_ratio = np.where(np.isfinite(ghi_safe),
                            np.clip(ghi_all / ghi_safe, 0.0, 1.1),
                            0.0)

    # Scale clear-sky components by the ratio
    dni = np.clip(kt_ratio * dni_clear, 0.0, None)
    dhi = np.clip(kt_ratio * dhi_clear, 0.0, None)

    # Sanity check: GHI = DNI × cos(z) + DHI
    cos_z = np.clip(cos_zenith, 0.0, 1.0)
    ghi_check = np.clip(dni * cos_z + dhi, 0.0, None)
    # If reconstructed GHI differs from input, normalise
    with np.errstate(invalid="ignore", divide="ignore"):
        norm = np.where(ghi_check > 0.5, ghi_all / ghi_check, 1.0)
    norm = np.clip(norm, 0.0, 2.0)
    dni  = dni * norm
    dhi  = dhi * norm

    return np.nan_to_num(dni, nan=0.0), np.nan_to_num(dhi, nan=0.0)
