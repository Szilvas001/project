"""
Aerosol optical property computation.

Implements:
  1. Ångström turbidity formula — spectral AOD from two reference wavelengths
  2. Hänel hygroscopic growth factor — AOD scaling with relative humidity
  3. SSA and asymmetry parameter estimation from speciated AOD
  4. Two-wavelength Ångström exponent calculation (α1: 340–500 nm, α2: 500–1064 nm)

References
----------
  Ångström (1929) "On the atmospheric transmission of sun radiation"
  Hänel (1976) "The properties of atmospheric aerosol particles as functions of
                the relative humidity at thermodynamic equilibrium"
  Shettle & Fenn (1979) "Models for the aerosols of the lower atmosphere"
  Kokhanovsky (2004) "Light Scattering Reviews"
"""

from __future__ import annotations

import numpy as np

# ── Hänel hygroscopic parameters by aerosol type ──────────────────────────
# γ exponent from Hänel (1976) Table 2 (continental and maritime aerosols)
# k_rh = hygroscopic amplification constant (AOD scaling)
_HANEL_GAMMA = {
    "continental": 0.37,   # clean continental
    "maritime":    0.48,   # marine boundary layer
    "urban":       0.44,   # urban/polluted
    "default":     0.40,
}

# Aerosol type refractive indices at 550 nm (real, imaginary)
# Used for simplified SSA/GG estimation when CAMS SSA is unavailable
_REFRACTIVE_INDEX = {
    # (m_real, m_imag) at 550 nm
    "dust":    (1.53, 0.006),
    "bc":      (1.75, 0.44),   # black carbon
    "oc":      (1.45, 0.005),  # organic carbon
    "sea_salt": (1.50, 1e-8),
    "sulphate": (1.43, 1e-8),
}

# Typical SSA and asymmetry g per aerosol species (550 nm)
_SPECIES_SSA = {
    "dust":     (0.93, 0.73),
    "bc":       (0.30, 0.45),
    "oc":       (0.92, 0.65),
    "sea_salt": (0.999, 0.72),
    "sulphate": (0.999, 0.64),
}

# ── AM1.5G ETR solar constant (W/m²) at TOA ───────────────────────────────
_E0 = 1361.0   # Solar constant (W/m²)


# ══════════════════════════════════════════════════════════════════════════
# Ångström formula
# ══════════════════════════════════════════════════════════════════════════

def angstrom_aod(
    tau_ref: float | np.ndarray,
    lambda_ref_nm: float,
    lambda_nm: float | np.ndarray,
    alpha: float | np.ndarray,
) -> float | np.ndarray:
    """
    Compute AOD at wavelength λ from a reference AOD via Ångström formula.

        τ(λ) = τ_ref × (λ / λ_ref)^(-α)

    Parameters
    ----------
    tau_ref       : AOD at reference wavelength
    lambda_ref_nm : Reference wavelength (nm)
    lambda_nm     : Target wavelength(s) (nm)
    alpha         : Ångström exponent

    Returns
    -------
    tau : AOD at target wavelength(s)
    """
    tau_ref = np.asarray(tau_ref, dtype=float)
    lambda_nm = np.asarray(lambda_nm, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    ratio = lambda_nm / lambda_ref_nm
    return np.clip(tau_ref * ratio ** (-alpha), 0.0, 10.0)


def angstrom_exponent(
    tau1: float | np.ndarray,
    lambda1_nm: float,
    tau2: float | np.ndarray,
    lambda2_nm: float,
) -> float | np.ndarray:
    """
    Compute Ångström exponent from two-wavelength AOD pair.

        α = -ln(τ₁/τ₂) / ln(λ₁/λ₂)

    Parameters
    ----------
    tau1, tau2       : AOD at wavelengths λ1, λ2
    lambda1_nm, lambda2_nm : Wavelengths in nm

    Returns
    -------
    alpha : Ångström exponent (clipped to [0, 3])
    """
    tau1 = np.asarray(tau1, dtype=float)
    tau2 = np.asarray(tau2, dtype=float)
    eps = 1e-10
    ratio_tau = np.clip(tau1, eps, None) / np.clip(tau2, eps, None)
    alpha = -np.log(ratio_tau) / np.log(lambda1_nm / lambda2_nm)
    return np.clip(alpha, 0.0, 3.0)


def compute_alpha1_alpha2(
    tau340: float | np.ndarray | None,
    tau500: float | np.ndarray | None,
    tau550: float | np.ndarray | None,
    tau670: float | np.ndarray | None,
    tau865: float | np.ndarray | None,
    tau1020: float | np.ndarray | None = None,
) -> tuple[float | np.ndarray, float | np.ndarray, float | np.ndarray]:
    """
    Compute SMARTS-compatible ALPHA1 (340–500 nm) and ALPHA2 (500–1064 nm)
    and best estimate of TAU550.

    Falls back gracefully when some wavelengths are NaN.

    Returns
    -------
    (alpha1, alpha2, tau550_best)
    """
    def safe(v, default=np.nan):
        if v is None:
            return np.full_like(tau550 or np.array([np.nan]), np.nan) \
                   if hasattr(tau550, '__len__') else np.nan
        v = np.asarray(v, dtype=float)
        v = np.where(v > 0, v, np.nan)
        return v

    t340 = safe(tau340)
    t500 = safe(tau500)
    t550 = safe(tau550)
    t670 = safe(tau670)
    t865 = safe(tau865)
    t1020 = safe(tau1020, default=np.nan)

    # ALPHA1: 340→500 nm (fine mode, UV-visible)
    if not _all_nan(t340) and not _all_nan(t500):
        alpha1 = angstrom_exponent(t340, 340.0, t500, 500.0)
    elif not _all_nan(t500) and not _all_nan(t550):
        alpha1 = angstrom_exponent(t500, 500.0, t550, 550.0)
    else:
        alpha1 = np.full_like(t550, 1.30) if hasattr(t550, '__len__') else 1.30

    # ALPHA2: 500→1064 nm (coarse mode, visible-NIR)
    if not _all_nan(t500) and not _all_nan(t1020):
        alpha2 = angstrom_exponent(t500, 500.0, t1020, 1020.0)
    elif not _all_nan(t550) and not _all_nan(t865):
        alpha2 = angstrom_exponent(t550, 550.0, t865, 865.0)
    elif not _all_nan(t670) and not _all_nan(t865):
        alpha2 = angstrom_exponent(t670, 670.0, t865, 865.0)
    else:
        alpha2 = np.full_like(t550, 1.30) if hasattr(t550, '__len__') else 1.30

    # Best TAU550
    if not _all_nan(t550):
        tau_best = np.where(np.isfinite(t550), t550, 0.10)
    elif not _all_nan(t500) and not _all_nan(alpha2):
        tau_best = angstrom_aod(t500, 500.0, 550.0, alpha2)
    elif not _all_nan(t670) and not _all_nan(alpha2):
        tau_best = angstrom_aod(t670, 670.0, 550.0, alpha2)
    else:
        tau_best = np.full_like(alpha1, 0.10) if hasattr(alpha1, '__len__') else 0.10

    alpha1 = np.clip(np.nan_to_num(alpha1, nan=1.30), 0.0, 3.0)
    alpha2 = np.clip(np.nan_to_num(alpha2, nan=1.30), 0.0, 3.0)
    tau_best = np.clip(np.nan_to_num(tau_best, nan=0.10), 0.005, 5.0)

    return alpha1, alpha2, tau_best


def _all_nan(v) -> bool:
    if v is None:
        return True
    v = np.asarray(v)
    return bool(np.all(~np.isfinite(v)))


# ══════════════════════════════════════════════════════════════════════════
# Hänel hygroscopic growth factor
# ══════════════════════════════════════════════════════════════════════════

def hanel_growth_factor(
    rh: float | np.ndarray,
    gamma: float = 0.40,
    rh_ref: float = 0.50,
    rh_delhyg: float = 0.99,
) -> float | np.ndarray:
    """
    Compute the Hänel hygroscopic growth factor f(RH).

    This factor multiplies the dry-air AOD to yield the ambient (humid) AOD:

        AOD_wet = AOD_dry × f(RH)

    Hänel's formula (Hänel 1976, eq. 4):

        f(RH) = ((1 - RH/RH_delhyg) / (1 - RH_ref/RH_delhyg))^(-γ)

    where RH_delhyg ≈ 0.99 is the deliquescence relative humidity.

    Parameters
    ----------
    rh         : Relative humidity [0, 1]
    gamma      : Hänel hygroscopic exponent (default 0.40 for continental)
    rh_ref     : Reference RH at which AOD was measured (default 0.50)
    rh_delhyg  : Deliquescence RH (default 0.99)

    Returns
    -------
    f : Growth factor ≥ 1.0
    """
    rh = np.clip(np.asarray(rh, dtype=float), 0.0, rh_delhyg - 0.001)
    rh_ref_val = min(rh_ref, rh_delhyg - 0.001)

    numerator   = 1.0 - rh / rh_delhyg
    denominator = 1.0 - rh_ref_val / rh_delhyg

    # Avoid divide-by-zero
    denominator = max(denominator, 1e-6)
    growth = (numerator / denominator) ** (-gamma)
    return np.clip(growth, 1.0, 15.0)


def hanel_corrected_aod(
    aod_dry: float | np.ndarray,
    rh: float | np.ndarray,
    aerosol_type: str = "default",
    rh_ref: float = 0.50,
) -> float | np.ndarray:
    """
    Apply Hänel hygroscopic correction to dry-air AOD.

        AOD_wet = AOD_dry × f(RH)

    Parameters
    ----------
    aod_dry      : Dry-air AOD (e.g., from CAMS reanalysis at RH=50%)
    rh           : Ambient relative humidity [0, 1]
    aerosol_type : One of 'continental', 'maritime', 'urban', 'default'
    rh_ref       : Reference RH of the input AOD (default 0.50 for CAMS)

    Returns
    -------
    aod_wet : Ambient AOD corrected for humidity
    """
    gamma = _HANEL_GAMMA.get(aerosol_type, _HANEL_GAMMA["default"])
    f = hanel_growth_factor(rh, gamma=gamma, rh_ref=rh_ref)
    return np.clip(np.asarray(aod_dry) * f, 0.0, 10.0)


# ══════════════════════════════════════════════════════════════════════════
# SSA and asymmetry parameter estimation
# ══════════════════════════════════════════════════════════════════════════

def estimate_ssa_g_from_species(
    aod_dust: float,
    aod_bc: float,
    aod_oc: float,
    aod_sea_salt: float,
    aod_sulphate: float,
) -> tuple[float, float]:
    """
    Estimate bulk single-scattering albedo (SSA/ω₀) and asymmetry parameter (g)
    from speciated aerosol AOD fractions.

    Uses AOD-weighted mixing rule:
        ω_bulk = Σ(ω_i × τ_i) / Σ(τ_i)
        g_bulk  = Σ(g_i × ω_i × τ_i) / Σ(ω_i × τ_i)

    Parameters
    ----------
    aod_* : AOD for each species at 550 nm

    Returns
    -------
    (ssa, g) — bulk SSA and asymmetry parameter, clipped to physical range
    """
    species = {
        "dust":      (aod_dust,     *_SPECIES_SSA["dust"]),
        "bc":        (aod_bc,       *_SPECIES_SSA["bc"]),
        "oc":        (aod_oc,       *_SPECIES_SSA["oc"]),
        "sea_salt":  (aod_sea_salt, *_SPECIES_SSA["sea_salt"]),
        "sulphate":  (aod_sulphate, *_SPECIES_SSA["sulphate"]),
    }

    total_aod = sum(max(0, v[0]) for v in species.values())
    if total_aod < 1e-6:
        return 0.92, 0.65   # continental background defaults

    ssa_sum = 0.0
    g_weight = 0.0
    g_sum = 0.0

    for tau_i, ssa_i, g_i in species.values():
        tau_i = max(0.0, tau_i)
        ssa_sum += ssa_i * tau_i
        g_weight += ssa_i * tau_i
        g_sum += g_i * ssa_i * tau_i

    ssa_bulk = ssa_sum / total_aod
    g_bulk = g_sum / max(g_weight, 1e-10)

    return float(np.clip(ssa_bulk, 0.50, 1.00)), float(np.clip(g_bulk, 0.40, 0.85))


def estimate_ssa_g_from_pm(
    pm25: float,
    pm10: float,
    rh: float = 0.50,
) -> tuple[float, float]:
    """
    Simplified SSA and g estimation from PM2.5 / PM10 ratio.

    The PM2.5/PM10 ratio indicates the fine-mode fraction of aerosols.
    Fine particles (PM2.5) scatter more (higher SSA) while coarse particles
    (dust in PM10) absorb more.

    Parameters
    ----------
    pm25 : PM2.5 surface concentration (μg/m³)
    pm10 : PM10 surface concentration (μg/m³)
    rh   : Relative humidity [0, 1] (humid air swells particles, increasing SSA)

    Returns
    -------
    (ssa, g) estimates
    """
    pm25 = max(0.1, pm25)
    pm10 = max(0.1, pm10)

    # Fine-mode fraction
    fmf = np.clip(pm25 / pm10, 0.0, 1.0)

    # SSA: fine mode ≈ 0.95 (scattering dominated), coarse (dust) ≈ 0.93
    # Humidity correction: humid particles scatter more
    rh_factor = 1.0 + 0.05 * max(0, rh - 0.5)
    ssa = np.clip((0.93 + 0.04 * fmf) * min(rh_factor, 1.05), 0.70, 1.00)

    # Asymmetry: fine particles have lower g (~0.60), coarse higher (~0.76)
    g = np.clip(0.75 - 0.15 * fmf, 0.55, 0.80)

    return float(ssa), float(g)


# ══════════════════════════════════════════════════════════════════════════
# Extraterrestrial irradiance / solar constant
# ══════════════════════════════════════════════════════════════════════════

def extraterrestrial_irradiance(dayofyear: int | np.ndarray) -> float | np.ndarray:
    """
    Spencer (1971) formula for the solar constant corrected for Earth–Sun
    distance variation.

        E₀(doy) = E_sc × [1.00011 + 0.034221·cos(Γ) + 0.001280·sin(Γ)
                          + 0.000719·cos(2Γ) + 0.000077·sin(2Γ)]

    where Γ = 2π(doy−1)/365

    Parameters
    ----------
    dayofyear : Day of year (1–365)

    Returns
    -------
    E0 : Extraterrestrial irradiance (W/m²)
    """
    doy = np.asarray(dayofyear, dtype=float)
    gamma = 2.0 * np.pi * (doy - 1.0) / 365.0
    e0 = _E0 * (
        1.00011
        + 0.034221 * np.cos(gamma)
        + 0.001280 * np.sin(gamma)
        + 0.000719 * np.cos(2 * gamma)
        + 0.000077 * np.sin(2 * gamma)
    )
    return e0


def kt_from_ghi(
    ghi: float | np.ndarray,
    ghi_clear: float | np.ndarray,
    min_ghi_clear: float = 5.0,
) -> float | np.ndarray:
    """
    Compute clearness index Kt = GHI / GHI_clear.

    Returns NaN when GHI_clear is below threshold (nighttime/near-horizon).
    """
    ghi = np.asarray(ghi, dtype=float)
    ghi_clear = np.asarray(ghi_clear, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        kt = np.where(
            ghi_clear > min_ghi_clear,
            np.clip(ghi / ghi_clear, 0.0, 1.1),
            np.nan,
        )
    return kt


def kt_denorm_factor(
    ghi_clear: float | np.ndarray,
    ghi_ref: float | np.ndarray = None,
) -> float | np.ndarray:
    """
    De-normalization factor for converting predicted Kt back to GHI.

    In training we normalize GHI by the clear-sky reference (Kt = GHI/GHI_cs).
    This factor converts predictions back:

        GHI_pred = Kt_pred × GHI_clear × denorm_factor

    For standard use, denorm_factor = 1.0.  It can be set to
    (GHI_clear_live / GHI_clear_training) to correct for systematic bias
    in the clear-sky model between training and forecast periods.

    Parameters
    ----------
    ghi_clear : Clear-sky GHI at forecast time
    ghi_ref   : Clear-sky GHI reference (training period mean); if None,
                returns the clear-sky values directly

    Returns
    -------
    denorm : De-normalization factor (scalar or array)
    """
    ghi_clear = np.asarray(ghi_clear, dtype=float)
    if ghi_ref is None:
        return ghi_clear

    ghi_ref = np.asarray(ghi_ref, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        factor = np.where(ghi_ref > 1.0, ghi_clear / ghi_ref, 1.0)
    return np.clip(factor, 0.0, 5.0)
