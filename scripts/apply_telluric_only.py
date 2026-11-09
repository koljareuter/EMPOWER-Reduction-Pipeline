#!/usr/bin/env python
"""Apply telluric correction only to already-reduced OBs.

Reads the same config as reduce_kmos.py.  For each OB / band it calls
telluric_correct() on existing _rv.fits files, producing _rv_tell.fits.
OBs where _rv_tell.fits already exist are skipped unless --overwrite is set.

Usage (from the repository root):
    python scripts/apply_telluric_only.py -p examples/empower_IZ.cfg
    # overwrite existing, skip January OBs:
    python scripts/apply_telluric_only.py -p examples/empower_IZ.cfg \
        --overwrite --skip-pattern KMOS.2026-01
"""
import os, glob, argparse
from pyspark.utils import setuputils, fileutils
from pyspark.cubes import telluric_correct

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--parameter_file', type=str,
                        help='Config file (same as reduce_kmos.py)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Remove existing _rv_tell.fits and recompute')
    parser.add_argument('--skip-pattern', type=str, default=None,
                        help='Skip OB folders whose name contains this string '
                             '(e.g. "KMOS.2026-01" to skip January OBs)')
    args = parser.parse_args()

    master_logger = setuputils.set_logging(logbase='master_telluric', log_level=3)
    run_params = setuputils.set_config(args.parameter_file, logger=master_logger)

    procdir = run_params['procdir']
    folders = run_params['subdirs']
    bands   = run_params['obsband']

    for band in bands:
        master_logger.info('### Telluric pass: {0}-band ###'.format(band))
        for folder in folders:
            if args.skip_pattern and args.skip_pattern in folder:
                master_logger.info('Skipping (pattern match): {0}'.format(folder))
                continue

            science_dir = os.path.join(procdir, folder, 'science')
            if not os.path.isdir(science_dir):
                master_logger.info('No science dir for {0}, skipping'.format(folder))
                continue

            rv_files = glob.glob(os.path.join(science_dir, '*_s_i_bkg2d_rv.fits'))
            if not rv_files:
                master_logger.info('No _rv.fits in {0}/{1}, skipping'.format(folder, band))
                continue

            tell_files = glob.glob(os.path.join(science_dir, '*_s_i_bkg2d_rv_tell.fits'))
            if tell_files and not args.overwrite:
                master_logger.info('Already done: {0}/{1} ({2} files), skipping'.format(
                    folder, band, len(tell_files)))
                continue

            if tell_files and args.overwrite:
                for f in tell_files:
                    os.remove(f)
                master_logger.info('Removed {0} existing _rv_tell.fits in {1}'.format(
                    len(tell_files), folder))

            master_logger.info('Applying telluric: {0} {1}'.format(folder, band))
            ecode = telluric_correct(logger=master_logger, subdir=folder, obsband=band,
                                     **run_params['cubes'])
            if ecode:
                master_logger.error('!! Telluric failed for {0}/{1}'.format(folder, band))
            else:
                master_logger.info('-- Done: {0}/{1}'.format(folder, band))
