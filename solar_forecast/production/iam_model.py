"""
Incidence Angle Modifier (IAM) models.

The IAM describes the reduction in transmitted irradiance through the module
cover (glass) as a function of the angle of incidence (AOI):

    G_transmitted = IAM(AOI) × G_direct

Three models are provided:

1.  ASHRAE (Souka & Safwat 1966, used in NREL SAM)
    Simple one-parameter model; valid for typical flat-plate glass covers:

        IAM = 1 − b₀ × (1/cos(AOI) − 1)

    Recommended b₀: 0.05 for standard antireflection-coated glass.
    Returns 0 for AOI > 85°.

2.  Martin-Ruiz (Martin & Ruiz 2001)
    More physical; captures the angular dependence of Fresnel reflection:

        IAM = exp(−c₁ × (1/cos(AOI) − 1)^c₂)

    Parameters: c₁ = 0.16, c₂ = 1.0  for c-Si with standard AR coating.

3.  Physical Fresnel
    Exact solution from Fresnel equations averaged over S and P polarisations
    assuming glass with n_glass = 1.526 and n_encapsulant = 1.50:

        r_s = (n₁ cos θᵢ − n₂ cos θₜ) / (n₁ cos θᵢ + n₂ cos θₜ)
        r_p = (n₂ cos θᵢ − n₁ cos θₜ) / (n₂ cos θᵢ + n₁ cos θₜ)
        R   = (r_s² + r_p²) / 2
        R₀  = ((n₁ − n₂) / (n₁ + n₂))²   (normal incidence)
        IAM = (1 − R) / (1 − R₀)

    This is the most accurate model for detailed loss analysis.

Diffuse IAM
-----------
A separate function handles the diffuse component using the Scott (1996) /
De Soto (2006) equivalent angle for the isotropic sky dome:

    AOI_diffuse ≈ 59.68° − 0.1388 × tilt + 0.001497 × tilt²

All models are vectorised with NumPy and accept scalars or arrays.
"""

import numpy as np

# Default ASHRAE parameter for antireflection-coated glass
_ASHRAE_B0 = 0.05

# Default Martin-Ruiz parameters
_MR_C1 = 0.16
_MR_C2 = 1.00

# Glass refractive index (typical soda-lime glass)
_N_GLASS = 1.526
# Air
_N_AIR = 1.000


def iam_ashrae(aoi_deg: np.ndarray, b0: float = _ASHRAE_B0) -> np.ndarray:
    """
    ASHRAE incidence angle modifier.

        IAM = 1 − b₀ × (1/cos(θ) − 1)

    Clipped to [0, 1]; returns 0 for AOI ≥ 85°.
    """
    aoi = np.asarray(aoi_deg, dtype=float)
    cos_aoi = np.cos(np.radians(aoi))
    # Avoid division by zero near 90°
    safe_cos = np.where(cos_aoi < 0.01, 0.01, cos_aoi)
    iam = 1.0 - b0 * (1.0 / safe_cos - 1.0)
    iam = np.clip(iam, 0.0, 1.0)
    iam = np.where(aoi >= 85.0, 0.0, iam)
    return iam


def iam_martin_ruiz(
    aoi_deg: np.ndarray,
    c1: float = _MR_C1,
    c2: float = _MR_C2,
) -> np.ndarray:
    """
    Martin-Ruiz IAM — exponential Fresnel-inspired model.

        IAM = exp(−c₁ × (1/cos(θ) − 1)^c₂)

    Returns 0 for AOI ≥ 85°.
    """
    aoi = np.asarray(aoi_deg, dtype=float)
    cos_aoi = np.cos(np.radians(aoi))
    safe_cos = np.where(cos_aoi < 0.01, 0.01, cos_aoi)
    exponent = c1 * np.power(np.maximum(0, 1.0 / safe_cos - 1.0), c2)
    iam = np.exp(-exponent)
    iam = np.clip(iam, 0.0, 1.0)
    iam = np.where(aoi >= 85.0, 0.0, iam)
    return iam


def iam_fresnel(aoi_deg: np.ndarray, n_glass: float = _N_GLASS) -> np.ndarray:
    """
    Physical Fresnel IAM for glass/air interface.

    Uses Snell's law and Fresnel equations for unpolarised light.
    Normalised to transmission at normal incidence (AOI = 0°).
    """
    aoi = np.asarray(aoi_deg, dtype=float)
    theta_i = np.radians(np.clip(aoi, 0.0, 89.9))

    sin_t = np.sin(theta_i) / n_glass
    sin_t = np.clip(sin_t, 0.0, 1.0)
    theta_t = np.arcsin(sin_t)

    cos_i = np.cos(theta_i)
    cos_t = np.cos(theta_t)

    r_s = (cos_i - n_glass * cos_t) / (cos_i + n_glass * cos_t + 1e-9)
    r_p = (n_glass * cos_i - cos_t) / (n_glass * cos_i + cos_t + 1e-9)
    R = (r_s ** 2 + r_p ** 2) / 2.0

    # Normal incidence (reference)
    R0 = ((1.0 - n_glass) / (1.0 + n_glass)) ** 2
    T0 = 1.0 - R0

    iam = (1.0 - R) / T0
    iam = np.clip(iam, 0.0, 1.0)
    iam = np.where(aoi >= 89.5, 0.0, iam)
    return iam


def iam_diffuse(tilt_deg: float, model: str = "ashrae") -> float:
    """
    Effective IAM for isotropic diffuse irradiance on a tilted surface.

    Uses the equivalent angle approach (De Soto 2006):

        AOI_eq = 59.68 − 0.1388 × tilt + 0.001497 × tilt²

    The result is the scalar IAM evaluated at this equivalent angle.
    """
    tilt = float(tilt_deg)
    aoi_eq = 59.68 - 0.1388 * tilt + 0.001497 * tilt ** 2
    if model == "martin_ruiz":
        return float(iam_martin_ruiz(np.array([aoi_eq]))[0])
    elif model == "fresnel":
        return float(iam_fresnel(np.array([aoi_eq]))[0])
    else:
        return float(iam_ashrae(np.array([aoi_eq]))[0])
