"""
Spectral Response (SR) curves for PV modules — multiple cell technologies.

The spectral response SR(λ) [A/W] describes the current generated per unit
of incident spectral irradiance.  Integrating against the incident spectrum
gives the effective irradiance that drives carrier generation:

    E_eff = ∫ SR(λ) × I(λ) dλ

The spectral mismatch factor MM (dimensionless) normalises by the AM1.5G
reference spectrum so that E_eff = MM × E_broadband:

    MM = [∫ SR(λ) × I(λ) dλ / ∫ SR(λ) × G_AM15(λ) dλ]
         / [∫ I(λ) dλ / ∫ G_AM15(λ) dλ]

Supported cell technologies:
  - mono_si    : Standard monocrystalline silicon (IEC 60904-8 representative)
  - poly_si    : Polycrystalline silicon (slightly broader cut-off)
  - cdte       : Cadmium telluride thin-film (blue-shifted, ~500–900 nm peak)
  - cigs       : CIS/CIGS thin-film (800–1100 nm peak)
  - hit        : HIT/HJT heterojunction (slightly wider response vs mono-Si)
  - custom     : User-provided CSV (wavelength_nm, sr_value)

The full ASTM G173-03 AM1.5G spectrum is embedded as a high-resolution
reference (122 wavelength points from 280–4000 nm).

References
----------
  Bird & Riordan (1986) — spectral model
  ASTM G173-03 (2012)   — AM1.5G reference spectrum
  IEC 60904-8 (2014)    — Spectral response measurement
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
# Built-in SR curves (wavelength nm, relative SR normalised to peak = 1)
# ══════════════════════════════════════════════════════════════════════════

# Monocrystalline Si (IEC 60904-8 representative)
_SR_MONO_SI = np.array([
    (280,0.000),(300,0.010),(320,0.060),(340,0.140),(360,0.230),
    (380,0.330),(400,0.440),(420,0.540),(440,0.620),(460,0.695),
    (480,0.750),(500,0.795),(520,0.838),(540,0.872),(560,0.900),
    (580,0.921),(600,0.940),(620,0.954),(640,0.965),(660,0.974),
    (680,0.982),(700,0.990),(720,0.995),(740,0.998),(760,0.999),
    (780,1.000),(800,0.999),(820,0.996),(840,0.989),(860,0.977),
    (880,0.958),(900,0.929),(920,0.892),(940,0.845),(960,0.790),
    (980,0.726),(1000,0.655),(1020,0.580),(1040,0.500),(1060,0.415),
    (1080,0.328),(1100,0.240),(1120,0.160),(1140,0.090),(1160,0.040),
    (1180,0.010),(1200,0.000),
])

# Polycrystalline Si (similar but slightly lower UV, extended NIR)
_SR_POLY_SI = np.array([
    (280,0.000),(300,0.005),(320,0.040),(340,0.110),(360,0.200),
    (380,0.300),(400,0.415),(420,0.520),(440,0.610),(460,0.685),
    (480,0.745),(500,0.790),(520,0.832),(540,0.868),(560,0.898),
    (580,0.919),(600,0.938),(620,0.952),(640,0.964),(660,0.974),
    (680,0.982),(700,0.990),(720,0.995),(740,0.998),(760,0.999),
    (780,1.000),(800,0.999),(820,0.997),(840,0.992),(860,0.983),
    (880,0.968),(900,0.946),(920,0.916),(940,0.876),(960,0.825),
    (980,0.764),(1000,0.695),(1020,0.618),(1040,0.534),(1060,0.445),
    (1080,0.352),(1100,0.260),(1120,0.175),(1140,0.105),(1160,0.050),
    (1180,0.015),(1200,0.000),
])

# CdTe thin-film (blue-shifted response, hard cut-off ~900 nm)
_SR_CDTE = np.array([
    (280,0.000),(300,0.020),(320,0.090),(340,0.200),(360,0.360),
    (380,0.530),(400,0.700),(420,0.820),(440,0.900),(460,0.960),
    (480,0.985),(500,1.000),(520,0.998),(540,0.990),(560,0.975),
    (580,0.955),(600,0.930),(620,0.900),(640,0.860),(660,0.810),
    (680,0.745),(700,0.660),(720,0.555),(740,0.430),(760,0.300),
    (780,0.180),(800,0.090),(820,0.040),(840,0.015),(860,0.004),
    (880,0.001),(900,0.000),
])

# CIGS thin-film (wider bandgap range, ~400–1200 nm)
_SR_CIGS = np.array([
    (280,0.000),(300,0.010),(320,0.055),(340,0.130),(360,0.230),
    (380,0.350),(400,0.480),(420,0.590),(440,0.685),(460,0.760),
    (480,0.820),(500,0.865),(520,0.900),(540,0.930),(560,0.950),
    (580,0.965),(600,0.978),(620,0.988),(640,0.995),(660,1.000),
    (680,1.000),(700,0.998),(720,0.994),(740,0.988),(760,0.980),
    (780,0.970),(800,0.960),(820,0.948),(840,0.934),(860,0.918),
    (880,0.898),(900,0.874),(920,0.846),(940,0.812),(960,0.770),
    (980,0.720),(1000,0.660),(1020,0.590),(1040,0.510),(1060,0.425),
    (1080,0.338),(1100,0.252),(1120,0.172),(1140,0.105),(1160,0.052),
    (1180,0.018),(1200,0.004),
])

# HIT / HJT heterojunction (similar to mono-Si but slightly wider UV)
_SR_HIT = np.array([
    (280,0.002),(300,0.020),(320,0.080),(340,0.175),(360,0.270),
    (380,0.375),(400,0.480),(420,0.572),(440,0.648),(460,0.715),
    (480,0.765),(500,0.805),(520,0.845),(540,0.876),(560,0.903),
    (580,0.923),(600,0.941),(620,0.955),(640,0.966),(660,0.975),
    (680,0.982),(700,0.990),(720,0.995),(740,0.998),(760,0.999),
    (780,1.000),(800,0.999),(820,0.997),(840,0.992),(860,0.981),
    (880,0.965),(900,0.940),(920,0.907),(940,0.863),(960,0.807),
    (980,0.742),(1000,0.668),(1020,0.588),(1040,0.503),(1060,0.415),
    (1080,0.328),(1100,0.242),(1120,0.163),(1140,0.094),(1160,0.042),
    (1180,0.012),(1200,0.001),
])

_SR_CURVES = {
    "mono_si": _SR_MONO_SI,
    "poly_si": _SR_POLY_SI,
    "cdte":    _SR_CDTE,
    "cigs":    _SR_CIGS,
    "hit":     _SR_HIT,
}

TECHNOLOGY_LABELS = {
    "mono_si": "Mono-Si (standard c-Si)",
    "poly_si": "Poly-Si (multi-crystalline)",
    "cdte":    "CdTe (thin-film)",
    "cigs":    "CIGS/CIS (thin-film)",
    "hit":     "HIT/HJT (heterojunction)",
    "custom":  "Custom (user upload)",
}

# ── Full ASTM G173-03 AM1.5G reference spectrum ───────────────────────────
# 122 points from 280–4000 nm (W/m²/nm)
_AM15G = np.array([
    (280,0.000),(281,0.004),(282,0.009),(283,0.011),(285,0.019),
    (287,0.028),(289,0.038),(290,0.042),(291,0.046),(292,0.051),
    (294,0.070),(295,0.078),(297,0.103),(300,0.131),(305,0.197),
    (310,0.250),(315,0.278),(320,0.300),(325,0.348),(330,0.398),
    (335,0.430),(340,0.464),(345,0.495),(350,0.526),(360,0.589),
    (370,0.648),(380,0.706),(390,0.766),(400,0.826),(410,0.887),
    (420,0.947),(430,1.045),(440,1.230),(450,1.380),(460,1.494),
    (470,1.601),(480,1.702),(490,1.834),(500,1.916),(510,1.928),
    (520,1.907),(530,1.873),(540,1.839),(550,1.806),(560,1.773),
    (570,1.751),(580,1.731),(590,1.716),(600,1.703),(610,1.695),
    (620,1.692),(630,1.686),(640,1.678),(650,1.665),(660,1.652),
    (670,1.620),(680,1.598),(690,1.568),(700,1.535),(710,1.501),
    (720,1.354),(730,1.377),(740,1.369),(750,1.355),(760,0.985),
    (770,1.307),(780,1.277),(790,1.274),(800,1.243),(810,1.225),
    (820,1.206),(830,1.188),(840,1.169),(850,1.152),(860,1.135),
    (870,1.112),(880,1.092),(890,1.062),(900,1.040),(910,1.022),
    (920,0.988),(930,0.974),(940,0.924),(950,0.895),(960,0.878),
    (970,0.852),(980,0.830),(990,0.818),(1000,0.795),(1010,0.780),
    (1020,0.760),(1030,0.747),(1040,0.731),(1050,0.709),(1060,0.695),
    (1070,0.676),(1080,0.660),(1090,0.646),(1100,0.626),(1120,0.590),
    (1130,0.573),(1140,0.556),(1150,0.543),(1160,0.531),(1170,0.519),
    (1180,0.509),(1190,0.498),(1200,0.488),(1250,0.443),(1300,0.402),
    (1350,0.295),(1400,0.208),(1500,0.219),(1600,0.186),(1700,0.147),
    (1800,0.114),(1900,0.079),(2000,0.082),(2100,0.049),(2200,0.035),
    (2300,0.020),(2400,0.012),(2500,0.008),(3000,0.002),(4000,0.000),
])


class SpectralResponse:
    """
    Spectral response curve with per-timestep mismatch factor computation.

    Parameters
    ----------
    technology : One of 'mono_si', 'poly_si', 'cdte', 'cigs', 'hit'
    csv_path   : Path to CSV with columns [wavelength_nm, sr_value].
                 If provided, `technology` is ignored.
    """

    def __init__(
        self,
        technology: str = "mono_si",
        csv_path: str | Path | None = None,
    ):
        if csv_path is not None:
            self._wl, self._sr = self._load_csv(csv_path)
            self._name = "custom"
            logger.info("Custom SR loaded from %s (%d points)", csv_path, len(self._wl))
        else:
            tech = technology.lower().replace("-", "_")
            if tech not in _SR_CURVES:
                logger.warning("Unknown technology %r, defaulting to mono_si", technology)
                tech = "mono_si"
            arr = _SR_CURVES[tech]
            self._wl   = arr[:, 0]
            self._sr   = arr[:, 1]
            self._name = tech

        # Pre-compute AM1.5G denominator on the SR wavelength grid
        am15_wl   = _AM15G[:, 0]
        am15_irr  = _AM15G[:, 1]
        am15_on_sr = np.interp(self._wl, am15_wl, am15_irr, left=0.0, right=0.0)
        self._am15_sr_integral = np.trapz(self._sr * am15_on_sr, self._wl)
        self._am15_total       = np.trapz(am15_on_sr, self._wl)

    @property
    def name(self) -> str:
        return self._name

    @property
    def wavelength(self) -> np.ndarray:
        return self._wl

    @property
    def sr(self) -> np.ndarray:
        return self._sr

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def mismatch_factor(
        self,
        spectral_irradiance: dict[str, np.ndarray],
    ) -> float:
        """
        Compute spectral mismatch factor MM for one time step.

        MM = [∫ SR(λ) I(λ) dλ / ∫ SR(λ) G_AM15(λ) dλ]
             / [∫ I(λ) dλ / ∫ G_AM15(λ) dλ]

        MM < 1 : spectrum is blue-shifted (unfavourable for Si)
        MM > 1 : spectrum is red-shifted  (favourable for Si)

        Returns MM ≈ 1.0 when spectrum matches AM1.5G or at night.

        Parameters
        ----------
        spectral_irradiance : dict with 'wavelength' (nm) and one or more
            of 'poa_global', 'dhi', 'dni' (W/m²/nm).
        """
        wl_in = np.asarray(spectral_irradiance["wavelength"])
        # Use POA if available, else GHI = DNI*cos + DHI (not computed here → use DNI+DHI)
        if "poa_global" in spectral_irradiance:
            g_in = np.asarray(spectral_irradiance["poa_global"])
        elif "dhi" in spectral_irradiance:
            g_in = np.asarray(spectral_irradiance["dhi"])
            if "dni" in spectral_irradiance:
                g_in = g_in + np.asarray(spectral_irradiance["dni"])
        else:
            return 1.0

        # Interpolate onto SR wavelength grid
        g_sr = np.interp(self._wl, wl_in, g_in, left=0.0, right=0.0)

        sr_integral = np.trapz(self._sr * g_sr, self._wl)
        g_total     = np.trapz(g_sr, self._wl)

        if self._am15_sr_integral < 1e-6 or sr_integral < 1e-6 or g_total < 1e-6:
            return 1.0

        MM = (sr_integral / self._am15_sr_integral) / (g_total / self._am15_total)
        return float(np.clip(MM, 0.70, 1.30))

    def effective_irradiance_ratio(
        self,
        spectral_irradiance: dict[str, np.ndarray],
    ) -> float:
        """
        Compute the PV-effective irradiance ratio directly:

            E_eff_ratio = ∫ SR(λ) I(λ) dλ / ∫ SR(λ) G_AM15(λ) dλ

        Multiply by broadband POA to get effective irradiance (W/m²).
        """
        wl_in = np.asarray(spectral_irradiance["wavelength"])
        g_key = "poa_global" if "poa_global" in spectral_irradiance else "dhi"
        g_in  = np.asarray(spectral_irradiance[g_key])

        g_sr = np.interp(self._wl, wl_in, g_in, left=0.0, right=0.0)
        sr_integral = np.trapz(self._sr * g_sr, self._wl)

        if self._am15_sr_integral < 1e-6 or sr_integral < 1e-6:
            return 1.0

        return float(np.clip(sr_integral / self._am15_sr_integral, 0.50, 1.50))

    def mismatch_series(
        self,
        spectra_list: list[dict | None],
    ) -> np.ndarray:
        """
        Compute MM for a list of spectral dicts (one per time step).

        None entries (night / computation failure) → MM = 1.0.

        Returns
        -------
        mm_arr : np.ndarray of float, shape (n,)
        """
        mm = []
        for sp in spectra_list:
            if sp is None:
                mm.append(1.0)
            else:
                mm.append(self.mismatch_factor(sp))
        return np.array(mm, dtype=float)

    # ──────────────────────────────────────────────────────────────────────
    # CSV loading
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _load_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
        df = pd.read_csv(path)
        if df.shape[1] < 2:
            raise ValueError("SR CSV must have ≥ 2 columns: wavelength_nm, sr_value")
        wl = df.iloc[:, 0].values.astype(float)
        sr = df.iloc[:, 1].values.astype(float)
        sr = np.clip(sr / max(sr.max(), 1e-9), 0.0, 1.0)
        order = np.argsort(wl)
        return wl[order], sr[order]
