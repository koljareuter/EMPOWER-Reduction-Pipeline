from __future__ import print_function
import numpy as np
import os
import glob
from astropy.io import fits

from ..utils.fileutils import get_obs_exposures, duplicate

__all__ = ["telluric_correct"]


def telluric_correct(logger=None, subdir=None, obsband=None,
                     telluric_in_sof=False, **kwargs):
    """Divide *_s_i_bkg2d_rv.fits cubes by the per-IFU telluric transmission.

    Reads TELLURIC_KMOS.*_<band>.fits from rawdir/<subdir>/ and writes
    *_s_i_bkg2d_rv_tell.fits. Channels with transmission <= 0.1 are set to
    NaN. The original _rv.fits files are left untouched.
    """
    if not telluric_in_sof:
        return 0

    rawdir = kwargs.get('rawdir', kwargs.get('inputdir'))
    tel_matches = glob.glob(
        os.path.join(rawdir, subdir, '*TELLURIC*{0}*'.format(obsband.upper())))
    if not tel_matches:
        if logger is not None:
            logger.warning('-- telluric_in_sof=True but no TELLURIC file in {0}/{1}'.format(
                rawdir, subdir))
        return 0

    tel_path = tel_matches[0]
    if logger is not None:
        logger.info('-- Telluric file: {0}'.format(os.path.basename(tel_path)))

    exposures = get_obs_exposures(
        subdir=os.path.join(subdir, 'science'),
        obsband=obsband, suffix='s_i_bkg2d_rv', **kwargs)

    if not len(exposures):
        if logger is not None:
            logger.warning('-- No _rv.fits frames found, skipping telluric step')
        return 0

    with fits.open(tel_path) as tel_hdul:
        for exposure, _, _, _ in exposures:
            fpath = os.path.split(exposure)[0]
            out_path = duplicate(exposure, outputdir=fpath, suffix='tell')

            with fits.open(out_path, mode='update') as sci_hdul:
                for ifu in range(1, 25):
                    try:
                        sci_data = sci_hdul[ifu].data
                    except IndexError:
                        continue
                    if sci_data is None or sci_data.ndim != 3:
                        continue

                    # Try named extension first, fall back to index
                    tel_data = None
                    for key in ('IFU.{0}.DATA'.format(ifu), ifu):
                        try:
                            tel_data = tel_hdul[key].data
                            break
                        except (KeyError, IndexError):
                            continue
                    if tel_data is None:
                        continue

                    if len(tel_data) != sci_data.shape[0]:
                        if logger is not None:
                            logger.warning(
                                'IFU {0}: nwave mismatch sci={1} tel={2}, skip'.format(
                                    ifu, sci_data.shape[0], len(tel_data)))
                        continue

                    tel_corr = np.array(tel_data, dtype=np.float64)
                    tel_corr[tel_corr <= 0.1] = np.nan

                    sci_hdul[ifu].data = np.asarray(
                        sci_data / tel_corr[:, np.newaxis, np.newaxis],
                        dtype=np.float32)

                    if logger is not None:
                        logger.debug(
                            'IFU {0}: tel min={1:.3f} max={2:.3f} median={3:.3f}'.format(
                                ifu, float(np.nanmin(tel_corr)),
                                float(np.nanmax(tel_corr)),
                                float(np.nanmedian(tel_corr))))

                sci_hdul[0].header['HISTORY'] = (
                    'Applied telluric correction from {0}'.format(
                        os.path.basename(tel_path)))
                sci_hdul.flush()

            if logger is not None:
                logger.info('-- Telluric corrected -> {0}'.format(
                    os.path.basename(out_path)))

    return 0
