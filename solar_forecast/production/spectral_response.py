"""
Spectral Response (SR) curves for PV modules.

The spectral response SR(λ) [A/W] describes the current generated per unit
of incident spectral irradiance.  Integrating it against the incident spectrum
gives the effective irradiance G_eff that actually drives carrier generation:

    G_eff = ∫ SR(λ) × G(λ) dλ / ∫ SR(λ) × G_ref(λ) dλ

where G_ref(λ) is the AM1.5G reference spectrum (ASTM G173-03).

This module provides:
  1. A built-in standard crystalline-silicon (c-Si) SR curve.
  2. A loader for custom SR curves from CSV files.
  3. The spectral mismatch factor MM calculation.

Spectral mismatch matters most under non-standard conditions:
  – Low sun angles (high airmass, shifted spectrum)
  – Overcast skies (blue-shifted diffuse spectrum)
  – Snow / high-altitude sites

For a 10 kW c-Si system the total spectral correction over a year is
typically ±2–3 %, but can reach ±8 % on specific days.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Standard c-Si spectral response: (wavelength nm, relative SR normalised to peak)
# Derived from IEC 60904-8 representative curve for monocrystalline Si
_CSI_SR = np.array([
    (280, 0.000), (300, 0.010), (320, 0.060), (340, 0.140),
    (360, 0.230), (380, 0.330), (400, 0.440), (420, 0.540),
    (440, 0.620), (460, 0.695), (480, 0.750), (500, 0.795),
    (520, 0.838), (540, 0.872), (560, 0.900), (580, 0.921),
    (600, 0.940), (620, 0.954), (640, 0.965), (660, 0.974),
    (680, 0.982), (700, 0.990), (720, 0.995), (740, 0.998),
    (760, 0.999), (780, 1.000), (800, 0.999), (820, 0.996),
    (840, 0.989), (860, 0.977), (880, 0.958), (900, 0.929),
    (920, 0.892), (940, 0.845), (960, 0.790), (980, 0.726),
    (1000,0.655),(1020, 0.580),(1040, 0.500),(1060, 0.415),
    (1080,0.328),(1100, 0.240),(1120, 0.160),(1140, 0.090),
    (1160,0.040),(1180, 0.010),(1200, 0.000),
])

# AM1.5G reference spectrum (compact subset, W/m²/nm) — ASTM G173-03
_AM15G = np.array([
    (280, 0.000), (300, 0.008), (320, 0.068), (340, 0.158),
    (360, 0.210), (380, 0.301), (400, 0.516), (420, 1.329),
    (440, 1.655), (460, 1.803), (480, 1.902), (500, 1.916),
    (520, 1.894), (540, 1.854), (560, 1.823), (580, 1.806),
    (600, 1.814), (620, 1.794), (640, 1.744), (660, 1.718),
    (680, 1.681), (700, 1.649), (720, 1.499), (740, 1.483),
    (760, 0.972), (780, 1.387), (800, 1.370), (820, 1.353),
    (840, 1.328), (860, 1.320), (900, 1.257),(1000, 1.067),
    (1100,0.864),(1200, 0.710),
])


class SpectralResponse:
    """
    Spectral response curve with mismatch factor computation.

    Parameters
    ----------
    csv_path : str or Path, optional
        Path to CSV with columns [wavelength_nm, sr_value].
        If None, the built-in c-Si curve is used.
    """

    def __init__(self, csv_path: str | Path | None = None):
        if csv_path is not None:
            self._wl, self._sr = self._load_csv(csv_path)
            logger.info("Custom SR loaded from %s (%d points)", csv_path, len(self._wl))
        else:
            self._wl = _CSI_SR[:, 0]
            self._sr = _CSI_SR[:, 1]
            logger.debug("Using built-in c-Si SR curve.")

        # Pre-compute AM1.5G denominator on the SR wavelength grid
        am15_wl = _AM15G[:, 0]
        am15_irr = _AM15G[:, 1]
        am15_on_sr_grid = np.interp(self._wl, am15_wl, am15_irr, left=0, right=0)
        self._am15_norm = np.trapz(self._sr * am15_on_sr_grid, self._wl)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def mismatch_factor(
        self,
        spectral_irradiance: dict[str, np.ndarray],
    ) -> float:
        """
        Compute the spectral mismatch factor MM for a given solar spectrum.

        MM = ∫ SR(λ) G(λ) dλ / (∫ SR(λ) G_AM15(λ) dλ)
             / (∫ G(λ) dλ / ∫ G_AM15(λ) dλ)

        Returns MM close to 1.0 for standard conditions; deviates when the
        spectrum shifts (high airmass, overcast, etc.).

        `spectral_irradiance` is a dict with keys 'wavelength' and 'poa_global'
        as returned by pvlib.spectrum.spectrl2.
        """
        wl_in = np.asarray(spectral_irradiance["wavelength"])
        g_in  = np.asarray(spectral_irradiance["poa_global"])

        # Interpolate incident spectrum onto SR wavelength grid
        g_on_sr = np.interp(self._wl, wl_in, g_in, left=0.0, right=0.0)

        numerator   = np.trapz(self._sr * g_on_sr, self._wl)
        denominator = self._am15_norm

        if denominator < 1e-6 or numerator < 1e-6:
            return 1.0  # trivially clear or night → no mismatch correction

        # Broadband normalisation
        g_total     = np.trapz(g_on_sr, self._wl)
        am15_total  = np.trapz(
            np.interp(self._wl, _AM15G[:, 0], _AM15G[:, 1], left=0, right=0),
            self._wl,
        )

        if g_total < 1e-6 or am15_total < 1e-6:
            return 1.0

        MM = (numerator / denominator) / (g_total / am15_total)
        return float(np.clip(MM, 0.80, 1.20))

    def effective_irradiance(self, poa: float, mm: float) -> float:
        """
        G_eff = POA × MM (simplified broadband spectral correction).
        """
        return max(0.0, poa * mm)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _load_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
        df = pd.read_csv(path)
        if df.shape[1] < 2:
            raise ValueError("SR CSV must have at least two columns: wavelength, sr_value")
        wl  = df.iloc[:, 0].values.astype(float)
        sr  = df.iloc[:, 1].values.astype(float)
        sr  = np.clip(sr / sr.max(), 0.0, 1.0)  # normalise to [0, 1]
        order = np.argsort(wl)
        return wl[order], sr[order]
