# Custom Spectral Response (SR) CSV Format

Expert users can upload a panel-specific SR(λ) curve in the dashboard or
pass a server-side path via the API (`sr_csv_path`). This document defines
the file format the engine accepts.

---

## File format

- **Encoding:** UTF-8, comma-separated (no BOM, no quoting required)
- **Header row:** required, first two columns are read
- **Required columns:**
  - column 1 — `wavelength_nm`  (integer or float, 280 – 4000 nm)
  - column 2 — `sr_value`        (relative, will be normalised to peak = 1)
- **Sort order:** any (the loader sorts ascending by wavelength)
- **Negative values:** clipped to 0
- **Out-of-range wavelengths:** silently dropped at integration time

Additional columns are ignored — you can keep `notes`, `device_id`, etc.
in the same file without breaking the loader.

### Minimum example

```csv
wavelength_nm,sr_value
280,0.000
400,0.450
600,0.940
780,1.000
1000,0.660
1200,0.000
```

### Recommended density

20 – 100 samples is usually enough. The loader linearly interpolates
between points, so a coarser curve still produces sensible mismatch
factors. For laboratory-grade modelling, supply 5 nm spacing across
300 – 1200 nm.

---

## What the engine does with it

For each forecast timestep:

1. Build an approximate atmospheric spectrum by Beer-Lambert attenuation
   of AM1.5G with airmass-scaled Rayleigh + H₂O absorption bands.
2. Integrate `SR(λ) × I(λ) dλ` over 280 – 4000 nm.
3. Normalise against the same integral evaluated for AM1.5G to obtain
   the spectral mismatch factor *MM*.
4. Multiply broadband POA by *MM* before the cell-temperature step.

The ratio is clipped to [0.85, 1.15] for numerical stability — uploads
that would force values outside this range are likely measurement noise
or a wrongly-normalised file.

---

## How to validate your upload

1. Drop the CSV in the **Expert** panel of the sidebar.
2. Run a forecast — the **Hourly table** shows a `spectral_factor` column.
   Values should oscillate with airmass; pure-night rows are exactly 1.
3. Switch back to a built-in technology (e.g. `mono_si`) and confirm
   the daily totals shift. If they don't, the upload didn't take.

---

## Source data

Reputable sources for measured SR curves:

- IEC 60904-8 reference modules
- NREL [PVRDB](https://pvdb.nrel.gov/) module datasheets
- Sandia Module Database (legacy, but still widely cited)
- Manufacturer technical specs — typically Figure 4 or 5 of any
  IEC 61853-1 datasheet

Always cross-check that the CSV column units are nm (not µm) and that
SR is dimensionless / normalised — the loader treats column 2 as a
relative response, not as A/W.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Output identical to mono-Si | CSV not loaded | Check the upload toast; re-upload |
| `spectral_factor` always 1.0 | All wavelengths outside 280 – 4000 nm | Verify wavelength column is in nm |
| Output 5 – 15 % off expected | Curve clipped at 0.85 / 1.15 | Re-normalise the curve to peak = 1 |
| CSV parse error | Encoded as UTF-16 / has BOM | Resave as UTF-8 without BOM |
