# EMPOWER Reduction Pipeline

Reduction pipeline for the KMOS IFU data of the EMPOWER programme. It takes raw KMOS frames
to sky-subtracted, illumination-, background-, barycentric- and telluric-corrected data cubes,
one OB at a time.

It is based on [pySPARK](https://github.com/jtmendel/pyspark) by Trevor Mendel (MIT licence),
extended for EMPOWER with:

- **row common-mode correction** of detectors 2 and 3 (lowers the noise of IFUs 9-24 by 5-8 %)
- **hybrid sky subtraction** for OBJECT frames whose SKY frame was taken at a rotator angle more than
  20 deg away, including a correction of the OH-line wings
- a **mean-free ZAPv2 eigenbasis** for IZ and H (the bundled IZ basis had a mean offset about 1000x too large)
- **telluric correction as a separate Python step** (tried putting the TELLURIC frame into the `kmos_sci_red`
  SOF produced spatial gradients)


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
2. Copy `examples/empower_IZ.cfg` and set `rawdir`, `procdir`, `subdirs` (the OB folder names) and `svd_path`.
3. Run the reduction from an empty working directory (esorex writes `.sof` files and logs there):
   ```bash
   python /path/to/EMPOWER-Reduction-Pipeline/scripts/reduce_kmos.py -p my_IZ.cfg
   ```
4. Apply the telluric correction:
   ```bash
   python /path/to/EMPOWER-Reduction-Pipeline/scripts/apply_telluric_only.py -p my_IZ.cfg
   ```

The final cubes are `procdir/<OB>/science/SCI_RECONSTRUCTED_*_s_i_bkg2d_rv_tell.fits`.
`scripts/rerun_rv_tell.py` re-applies the barycentric and telluric corrections to OBs that are already reduced.

For YJ or H, set `obsband` accordingly. For H, use `ZAPv2_SVD_H_n500004_nomean.fits` as `svd_path`;
for YJ, leave `svd_path` out to use the bundled basis.

### Optional: patched KMOS DRP

The EMPOWER production runs used a KMOS DRP patched to use a linear (instead of quadratic)
OH-line wavelength correction, selected with `recipe_dir` and `lcorr_degree = 1` in `[cubes]`, together with
an updated IZ OH spectrum (`oh_spec_iz`). These options are not needed for a standard reduction;
without them the stock esorex recipes are used.

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
