# EMPOWER Reduction Pipeline

Reduction pipeline for the KMOS IFU data of the EMPOWER programme. It takes raw KMOS frames
to sky-subtracted, illumination-, background- and barycentric-corrected data cubes, one OB at a time,
and corrects the combined cubes of every galaxy for telluric absorption, with a flux scale.

It is based on [pySPARK](https://github.com/jtmendel/pyspark) by Trevor Mendel (MIT licence),
extended for EMPOWER with:

- **row common-mode correction** of detectors 2 and 3 (lowers the noise of IFUs 9-24 by 5-8 %)
- **hybrid sky subtraction** for OBJECT frames whose SKY frame falls into another calibration bin (the DRP
  takes the flat/LCAL of another of the six calibration rotator angles), including a correction of the OH-line wings
- a **mean-free ZAPv2 eigenbasis** for IZ and H (the bundled IZ basis had a mean offset about 1000x too large)
- **telluric correction as a separate Python step** (tried putting the TELLURIC frame into the `kmos_sci_red`
  SOF produced spatial gradients), applied to the combined cubes with a frame-weighted transmission
  (`scripts/telluric_combined.py`)


The Python package keeps its original name, `pyspark`. It is unrelated to Apache PySpark;
don't install both in the same environment!

## Requirements

- Linux or macOS. The pipeline calls `esorex` and is not supported on Windows (WSL works).
- [ESO KMOS pipeline](https://www.eso.org/sci/software/pipelines/kmos/) with `esorex`, tested with KMOS 4.5.11.
  `esorex --recipes` must list the `kmos_*` recipes.
- `dfits` and `fitsort`, the ESO FITS header tools. They are two self-contained C files:
  ```bash
  mkdir -p ~/src/saft ~/.local/bin && cd ~/src/saft
  curl -O https://ftp.eso.org/scisoft/scisoft4/sources/eclipse/eclipse-4.2.0/qfits/saft/dfits.c \
       -O https://ftp.eso.org/scisoft/scisoft4/sources/eclipse/eclipse-4.2.0/qfits/saft/fitsort.c
  cc -O2 -o ~/.local/bin/dfits dfits.c && cc -O2 -o ~/.local/bin/fitsort fitsort.c
  ```
  `~/.local/bin` must be on your `PATH`.
- A C compiler for the Cython extension (`gcc`, or `xcode-select --install` on macOS).
- Python 3 with the packages in `requirements.txt`.

## Installation

```bash
git clone https://github.com/koljareuter/EMPOWER-Reduction-Pipeline.git
cd EMPOWER-Reduction-Pipeline
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install --no-build-isolation -e .
```

`--no-build-isolation` is needed because `setup.py` imports numpy and Cython at build time.

Point `KMOS_CALIB` at the static calibration folder of your KMOS pipeline installation, the folder
that contains `kmos_wave_band.fits`, and add the line to `~/.bashrc` (Linux) or `~/.zshrc` (macOS):

```bash
export KMOS_CALIB=/path/to/esopipes/datastatic/kmos-4.5.11
```

Check the setup:

```bash
python -c "import pyspark, pyspark.utils.interpolation, pyspark.calibration; print('ok')"
```

`pyspark` reads `KMOS_CALIB` when it is imported, so this fails with a `KeyError` if the variable is not set.

## Running

1. Sort the raw data into one folder per OB below a common `rawdir`. Each OB folder holds the science
   and sky frames together with their darks, flats, arcs and telluric standard.
2. Copy `examples/empower.cfg` and set `rawdir`, `procdir`, `subdirs` (the OB folder names), and the absolute
   paths of `svd_path_iz`, `svd_path_h`, `oh_spec_iz` and `recipe_dir`.
3. Run the reduction from an empty working directory (esorex writes `.sof` files and logs there):
   ```bash
   python /path/to/EMPOWER-Reduction-Pipeline/scripts/reduce_kmos.py -p my.cfg
   ```
   The final single frames are `procdir/<OB>/science/SCI_RECONSTRUCTED_*_s_i_bkg2d_rv.fits`.
4. Combine all frames of every galaxy with the ESO EDPS workflow KMOS `cubes_combination` (`kmos_combine`,
   `method=header`, `cmethod=ksigma`). Point EDPS only at the final frames, never at `procdir` itself (it would
   also combine the `sky/` cubes and intermediate stages). Per band, link them into a clean input folder:
   ```bash
   python scripts/make_edps_input.py IZ --procdir /path/to/kmos_proc --out /path/to/edps_in_IZ
   ```
   After the EDPS run, `scripts/edps_pick_jobs.py` checks that every job used only `_s_i_bkg2d_rv` frames and
   copies the combined cubes (`--copy /path/to/res_IZ`).
5. Telluric correction and flux scale of the combined cubes (method `best`, static files in `telluric_static/`):
   ```bash
   python scripts/telluric_combined.py index --frames /path/to/kmos_proc --out telluric_frame_index.json
   python scripts/telluric_combined.py apply --comb /path/to/res_IZ --index telluric_frame_index.json \
          --tell /path/to/kmos_raw --out /path/to/res_IZ_tell
   ```
   The transmission of a combined cube is the on-source-time-weighted mean of the archive `TELLURIC` spectra of
   its frames, each shifted by the frame's barycentric velocity. Fluxes are `DATA × FLUXK`
   (10⁻¹⁷ erg s⁻¹ cm⁻² Å⁻¹, header keyword `FLUXK`); `TELLEFF_*.fits` next to each cube holds the transmission.

`examples/empower.cfg` reduces IZ, YJ and H in one run with the same EMPOWER settings for every band:
row common-mode correction, linear OH wavelength correction, hybrid sky subtraction with the slitlet-position
model (`hybrid_oh_model = pos3`, `hybrid_free_model = ifumed`), OH-wing and slitlet-end terms, and barycentric
correction by resampling. The only per-band settings are the ZAP eigenbasis (`svd_path_<band>`: mean-free bases
for IZ and H, the bundled basis for YJ) and the OH reference spectrum (`oh_spec_<band>`: the new file for IZ, the
standard files for YJ and H).

## Reproducing the EMPOWER cubes

The pipeline runs with a standard ESO installation. To get the same cubes as the EMPOWER reduction,
these four things must match:

**1. KMOS pipeline 4.5.11.** All EMPOWER data were reduced with the KMOS 4.5.11 pipeline kit
(`kmos-kit-4.5.11`: esorex 3.13.10, CPL 7.3.2) and its static calibrations, which `KMOS_CALIB` must point to.
Older versions have not been tested, and their recipes and calibration files can differ. Check with
`esorex --version` and `ls $KMOS_CALIB`.

**2. New IZ OH spectrum.** `kmos_sci_red` corrects the wavelength solution of every frame and IFU with the OH
lines of a reference spectrum. The standard `kmos_oh_spec_iz.fits` of KMOS 4.5.11 has only 10 usable lines, all at
0.98-1.09 µm; the correction is extrapolated down to 0.78 µm, and object and sky frames end up misaligned by
7-29 km/s (median 13 km/s) at 0.79-0.87 µm, which leaves OH residuals in the Hβ range of z ≈ 0.6-0.8 targets.
EMPOWER uses `pyspark/data/kmos_oh_spec_iz_NEW.fits` instead (same format and header, 27 usable lines, the bluest
at 0.835 µm), set in the config with `oh_spec_iz`. YJ and H use the standard files.

**3. Patched `kmos_sci_red` (the only change to ESO code).** The OH wavelength correction in
`kmos/kmo_priv_lcorr.c` fits a hard-coded 2nd-order polynomial. For EMPOWER the KMOS 4.5.11 source was patched
(`drp_patch/kmos_lcorr_degree_patch.py`) so that the degree can be set with the environment variable
`KMOS_LCORR_DEGREE`; nothing else in the DRP was changed. Without the variable the patched recipes give
bit-identical results to the standard ones. The EMPOWER configuration uses degree 1 (linear) for all bands.
In the worst IZ OB this lowered the object-sky misalignment at Hβ from 7.1 to 5.0 km/s and the scatter
between IFUs from 4.3 to 1.8 km/s (with the new OH spectrum in both cases); YJ changes by less than 0.2 km/s.
In H the linear fit was slightly worse than the quadratic in the test OB (IFU scatter in the blue 0.4 → 0.8 km/s).

Build it next to the existing installation (the existing one is not modified):
```bash
PIPE=/path/to/kmos/install TARBALL=/path/to/kmos-4.5.11.tar.gz DEST=/path/to/KMOS_4.5.11_lcorr \
    bash drp_patch/build_kmos_lcorr_drp.sh
```
`kmos-4.5.11.tar.gz` is inside the KMOS 4.5.11 pipeline kit (md5 `f8d1e09b7516834bb7edbbc69e3ec9e7`).
Then set `recipe_dir` in `[cubes]` to the printed recipe directory. pyspark passes it and `lcorr_degree` to
esorex only for `kmos_sci_red`; the recipe log confirms the setting with a line
`OH lambda correction: polynomial degree 1`. Without the patched recipes, comment out both `recipe_dir` and
`lcorr_degree` (pyspark refuses `lcorr_degree` alone); you then get the standard quadratic correction.

**4. Detector-3 flats on seven IZ nights.** On the IZ nights 2026-01-13, 01-14 and 02-10 to 02-14, `kmos_flat`
with the default `-badpix_thresh=10` rejects detector 3 as vignetted, and IFUs 18-24 end up empty.
The EMPOWER reduction used `flat_args = -cmethod=ksigma -badpix_thresh=35` in `[calibration]` for those nights.

Everything else is the standard ESO software: pyspark calls the plain `esorex` on the `PATH` with the recipe
options in its code, and parameters it doesn't set keep the KMOS 4.5.11 recipe defaults. The EMPOWER reduction
ran without any esorex configuration files; esorex would read `~/.esorex/esorex.rc` or `~/.esorex/<recipe>.rc`
if they exist, so make sure there are none. The combination itself runs in the ESO EDPS workflow
`cubes_combination`; this repository provides the scripts that prepare its input and check its jobs.

## Troubleshooting

- **`KeyError: 'KMOS_CALIB'`**: the environment variable is not set in the shell (or Jupyter kernel) running the pipeline.
- **No frames found / empty lists**: `dfits` or `fitsort` are not on the `PATH`.
- **`Empty or corrupt FITS file`**: an earlier step left a truncated file. This lists every file whose size is not
  a whole number of FITS blocks:
  ```bash
  for f in calib/*.fits science/*.fits; do s=$(wc -c < "$f"); [ "$s" -gt 0 ] && [ $((s % 2880)) -eq 0 ] || echo "bad: $f"; done
  ```
  Check `esorex.log` for the failing recipe, delete the bad files and rerun that step.

## Licence

MIT, see `LICENSE`. Original pySPARK code © 2017 Trevor Mendel.
