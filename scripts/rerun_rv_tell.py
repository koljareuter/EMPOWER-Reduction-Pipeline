#!/usr/bin/env python
"""Re-run RV correction AND telluric correction for selected OBs.

Re-applies the barycentric RV shift (overwriting *_s_i_bkg2d_rv.fits) and
then the telluric correction (overwriting *_s_i_bkg2d_rv_tell.fits).
Existing output files are silently replaced (duplicate() removes them first).

Use --skip-pattern to exclude OBs whose folder name contains a substring,
e.g. 'KMOS.2026-01' to skip January OBs.

Usage (from the repository root):
    python scripts/rerun_rv_tell.py -p examples/empower_IZ.cfg \
        --skip-pattern KMOS.2026-01
"""
import os, glob, argparse
from pyspark.utils import setuputils
from pyspark.cubes import velocity_correct, telluric_correct

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--parameter_file', type=str, required=True,
                        help='Config file (same as reduce_kmos.py)')
    parser.add_argument('--skip-pattern', type=str, default=None,
                        help='Skip OB folders whose name contains this string '
                             '(e.g. "KMOS.2026-01" to skip January OBs)')
    parser.add_argument('--all-subdirs', action='store_true',
                        help='Ignore subdirs from config; scan procdir for all KMOS.* folders')
    args = parser.parse_args()

    master_logger = setuputils.set_logging(logbase='master_rv_tell', log_level=3)
    run_params = setuputils.set_config(args.parameter_file, logger=master_logger)

    procdir = run_params['procdir']
    bands   = run_params['obsband']

    if args.all_subdirs:
        folders = sorted(f for f in os.listdir(procdir)
                         if f.startswith('KMOS.') and os.path.isdir(os.path.join(procdir, f)))
        master_logger.info('--all-subdirs: found {0} folders in {1}'.format(len(folders), procdir))
    else:
        folders = [f for f in run_params['subdirs'] if f.strip()]

    for band in bands:
        master_logger.info('### Band: {0} ###'.format(band))
        for folder in folders:
            if args.skip_pattern and args.skip_pattern in folder:
                master_logger.info('Skipping (pattern): {0}'.format(folder))
                continue

            science_dir = os.path.join(procdir, folder, 'science')
            if not os.path.isdir(science_dir):
                master_logger.info('No science dir: {0}'.format(folder))
                continue

            bkg_files = glob.glob(os.path.join(science_dir, '*_s_i_bkg2d.fits'))
            if not bkg_files:
                master_logger.info('No _bkg2d.fits in {0}, skipping'.format(folder))
                continue

            # Step 1: RV correction (overwrites existing _rv.fits)
            master_logger.info('[RV]      {0} {1}'.format(folder, band))
            ecode = velocity_correct(logger=master_logger, subdir=folder, obsband=band,
                                     search_suffix='s_i_bkg2d', **run_params['cubes'])
            if ecode:
                master_logger.error('!! RV failed for {0}/{1}, skipping telluric'.format(
                    folder, band))
                continue

            # Step 2: Telluric correction (overwrites existing _rv_tell.fits)
            master_logger.info('[TELLURIC] {0} {1}'.format(folder, band))
            ecode = telluric_correct(logger=master_logger, subdir=folder, obsband=band,
                                     **run_params['cubes'])
            if ecode:
                master_logger.error('!! Telluric failed for {0}/{1}'.format(folder, band))
            else:
                master_logger.info('-- Done: {0}/{1}'.format(folder, band))
