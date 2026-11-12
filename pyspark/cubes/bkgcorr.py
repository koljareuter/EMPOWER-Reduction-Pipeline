from __future__ import print_function
import os
import numpy as np
import bottleneck as bn

from astropy.io import fits
from scipy.ndimage import median_filter, uniform_filter1d

from ..utils.fileutils import (get_obs_exposures, duplicate)
from ..utils.obsutils import (hsm, find_lines, reconstruct_wave, band_limits,
                              make_image, make_var_cube, sky_path)

from ..utils.psfutils import fit_psf

import matplotlib.pyplot as pl


__all__ = ["bkg_correct_2d", "bkg_correct_spec", "plane_offsets"]

PLANE_MIN_SPX = 30          # planes with fewer valid empty spaxels are not corrected


def plane_offsets(cube, rmask=4., hp=31, peak_sigma=5., shrink=101):
    """Per-plane residual-sky offset of one IFU (the 'bkg_plane' step of bkg_correct_2d).

    Per wavelength plane the median of the empty spaxels: not on the outer spaxel ring (biased edge rows),
    not within rmask spaxels of the IFU centre (where the arm puts the target), and not within rmask spaxels
    of the continuum peak if that is detected at > peak_sigma. Only the part that varies from plane to plane
    is returned (the offset minus its running median over hp planes), so a smooth continuum, e.g. the outskirts
    of an extended galaxy, is left alone; the smooth level is bkg_correct_2d's single value.
    shrink > 1: every offset is scaled by the Wiener factor max(0, 1 - n2 / m2), with n2 the variance of the median
    from the spaxel scatter of that plane (1.2533**2 sigma**2 / N) and m2 the mean square offset over shrink planes
    around it. Where the IFU has no common residual (m2 ~ n2) nothing is subtracted, so the step does not add the
    noise of its own estimate to every spaxel; where the residual is large (m2 >> n2) it is removed fully.
    Returns (offsets [nz], number of empty spaxels).
    """
    nz, ny, nx = cube.shape
    yy, xx = np.mgrid[:ny, :nx]
    est = (yy > 0) & (yy < ny - 1) & (xx > 0) & (xx < nx - 1)
    est &= np.hypot(yy - (ny - 1) / 2., xx - (nx - 1) / 2.) > rmask
    img = bn.nanmedian(cube, axis=0)
    good = np.isfinite(img) & (yy > 1) & (yy < ny - 2) & (xx > 1) & (xx < nx - 2)
    if good.sum() > 10:
        vals = img[good]
        med = np.median(vals)
        sig = 1.4826 * np.median(np.abs(vals - med))
        k = np.argmax(np.where(good, img, -np.inf))
        py, px = np.unravel_index(k, img.shape)
        if sig > 0 and img[py, px] - med > peak_sigma * sig:
            est &= np.hypot(yy - py, xx - px) > rmask
    flat = cube[:, est]
    nval = np.isfinite(flat).sum(axis=1)
    off = bn.nanmedian(flat, axis=1)
    off[(nval < PLANE_MIN_SPX) | ~np.isfinite(off)] = np.nan
    if hp and int(hp) > 1:
        fill = np.where(np.isfinite(off), off, bn.nanmedian(off) if np.isfinite(off).any() else 0.)
        off = off - median_filter(fill, size=int(hp), mode='nearest')
    off = np.nan_to_num(off)
    if shrink and int(shrink) > 1:
        sig = 1.4826 * bn.nanmedian(np.abs(flat - bn.nanmedian(flat, axis=1)[:, np.newaxis]), axis=1)
        n2 = np.where(nval >= PLANE_MIN_SPX, (1.2533 * np.nan_to_num(sig)) ** 2 / np.maximum(nval, 1), 0.)
        m2 = uniform_filter1d(off ** 2, size=int(shrink), mode='nearest')
        n2s = uniform_filter1d(n2, size=int(shrink), mode='nearest')
        w = np.clip(1. - n2s / np.where(m2 > 0, m2, np.inf), 0., 1.)
        off = off * w
    return off, int(est.sum())


def bkg_correct_2d(logger=None, subdir=None, obsband=None, 
               search_suffix=None, bkg_corr=True, bkg_plane=False, bkg_plane_rmask=4.,
               bkg_plane_hp=31, bkg_plane_shrink=101, **kwargs):
    """Background correction of the sky-subtracted, illumination-corrected frames (suffix bkg2d): one modal
    level per IFU from the collapsed image (DRP-like), and with [cubes] bkg_plane = True in addition the
    per-plane residual-sky offset of the empty spaxels (plane_offsets; bkg_plane_rmask: object mask radius in
    spaxels around the IFU centre and a detected continuum peak, default 4; bkg_plane_hp: planes of the running
    median whose part is kept, default 31; bkg_plane_shrink: planes of the Wiener factor, default 101, 0 = off).
    The per-plane offset is the sky residual common to the whole IFU
    that the 3x3-aperture noise of the hybrid frames is dominated by (docs/hybrid-sky.md); it does not depend on
    the object, which is masked.
    """

    if not bkg_corr:
        return 0
    bkg_plane = str(bkg_plane).strip().lower() in ('true', '1', 'yes')
    
    if logger is not None:
        logger.debug('- Running background correction')

    # get list of frames to process for background correction.  Assumes 
    # the frames are sky subtracted and illumination corrected (is this true?)
    exposures = get_obs_exposures(subdir=os.path.join(subdir, 'science'), 
                                  obsband=obsband, suffix=search_suffix, **kwargs)

    # directory with un-sky-subtracted data for line identification
    sky_dir = sky_path(subdir, **kwargs)
    
    if not len(exposures):
        if logger is not None:
            logger.warning('-- No exposures found for background correction')
        return 1

    # iterate over each exposure, removing background term.
    for exposure, _, _, _ in exposures:
        fpath, fname = os.path.split(exposure)
    
        #copy object to output
        exp_corr = duplicate(exposure, outputdir=fpath, suffix='bkg2d')

        # un-sky-subtracted cube for sky-line masking (exists with zapv2/zapv1);
        # with spark sky subtraction kmos_sci_red does not save that intermediate,
        # so fall back to the science frame itself — residuals are still usable
        exp_sky = os.path.join(sky_dir, "{0}.fits".format(fname[:-9]))
        if not os.path.isfile(exp_sky):
            exp_sky = exposure
        with fits.open(exp_corr, mode='update') as ofile,\
                fits.open(exp_sky) as sfile:

            for ifu in range(1,25):
                cube = np.array(ofile[ifu].data)
                scube = np.array(sfile[ifu].data)

                try:
                    oz, ox, oy = cube.shape
                except ValueError:
                    continue
                else:
                    #generate median sky spectrum 
                    med_sky = bn.nanmedian(scube.reshape(2048,-1), axis=1)
                    msky = find_lines(med_sky)

                    #estimate of variance cube
                    vcube = make_var_cube(scube)

                    #generate wavelength array
                    wave_array = reconstruct_wave(ofile[ifu].header)

                    #generate collapsed image
                    band_low, band_high = band_limits(obsband)
                    bkg_image = make_image(wave_array, cube, vcube, msky, llim=band_low+0.05, hlim=band_high-0.05)

                    #estimate modal background value, excluding edges
                    mode_bkg = hsm(bkg_image[1:-1,1:-1])

                    #re-apply to 
                    corr = np.copy(cube) - mode_bkg

                    #per-plane residual-sky offset of the empty spaxels (optional)
                    if bkg_plane:
                        off, nest = plane_offsets(corr, rmask=float(bkg_plane_rmask), hp=int(bkg_plane_hp),
                                                  shrink=int(bkg_plane_shrink))
                        corr = corr - off[:, np.newaxis, np.newaxis]
                        ofile[ifu].header['HIERARCH ESO PYSPARK BKG PLANE NSPX'] = (nest, 'empty spaxels per plane')
                        ofile[ifu].header['HIERARCH ESO PYSPARK BKG PLANE RMS'] = (
                            float(np.round(np.std(off), 5)), 'rms of the per-plane offsets')
                        if logger is not None:
                            logger.debug('-- IFU {0}: per-plane offsets from {1} spaxels, rms {2:.4f}'.format(
                                ifu, nest, np.std(off)))

                    ofile[ifu].data = corr

                    if logger is not None:
                        logger.debug('-- Applied background correction to IFU {0}: {1}'.format(ifu, mode_bkg))
            
                ofile[ifu].header['HISTORY'] = "Applied background correction: {0:.3f}".format(mode_bkg)
            if bkg_plane:
                ofile[0].header['HIERARCH ESO PYSPARK BKG PLANE'] = (True, 'per-plane residual-sky offset removed')
                ofile[0].header['HIERARCH ESO PYSPARK BKG PLANE RMASK'] = (float(bkg_plane_rmask),
                                                                           '[spx] object mask radius')
                ofile[0].header['HIERARCH ESO PYSPARK BKG PLANE HP'] = (int(bkg_plane_hp), 'running-median planes')
                ofile[0].header['HIERARCH ESO PYSPARK BKG PLANE SHRINK'] = (int(bkg_plane_shrink), 'Wiener-factor planes')
           

    return 0



def bkg_correct_spec(logger=None, subdir=None, obsband=None, 
               search_suffix=None, bkg_corr=True, **kwargs):

    if not bkg_corr:
        return 0
    
    if logger is not None:
        logger.debug('- Running background correction')

    # get list of frames to process for background correction.  Assumes 
    # the frames are sky subtracted and illumination corrected (is this true?)
    exposures = get_obs_exposures(subdir=os.path.join(subdir, 'science'), 
                                  obsband=obsband, suffix=search_suffix, **kwargs)

    # directory with un-sky-subtracted data for line identification
    sky_dir = sky_path(subdir, **kwargs)
    
    if not len(exposures):
        if logger is not None:
            logger.warning('-- No exposures found for background correction')
        return 1

    # iterate over each exposure, removing background term.
    for exposure, _, _, _ in exposures:
        fpath, fname = os.path.split(exposure)
    
        #copy object to output
        exp_corr = duplicate(exposure, outputdir=fpath, suffix='bkgspec')

	#define un-sky-subtracted exposure for line masking
        exp_sky = os.path.join(sky_dir, "{0}.fits".format(fname[:42]))

        with fits.open(exp_corr, mode='update') as ofile,\
                fits.open(exp_sky) as sfile:

            for ifu in range(1,25):
                cube = np.array(ofile[ifu].data)
                scube = np.array(sfile[ifu].data)

                try:
                    oz, ox, oy = cube.shape
                except ValueError:
                    continue
                else:
                    #generate median sky spectrum 
                    med_sky = bn.nanmedian(scube.reshape(2048,-1), axis=1)
                    msky = find_lines(med_sky)

                    #estimate of variance cube
                    vcube = make_var_cube(scube)

                    #generate wavelength array
                    wave_array = reconstruct_wave(ofile[1].header)

                    #generate collapsed image
                    band_low, band_high = band_limits(obsband)
                    bkg_image = make_image(wave_array, cube, vcube, msky, llim=band_low+0.05, hlim=band_high-0.05)
                   
                    #clip edges off the background image
                    clipped_bkg = bkg_image[1:-1,1:-1]

                    #remove upper and lower 5% of data, and compute the mode of remaining cube
                    low, high = np.percentile(clipped_bkg, (5,95))
                    mode_bkg = hsm(clipped_bkg[(clipped_bkg >= low) & (clipped_bkg <= high)])

                    #compute second quartile below the mode, use this to derive 
                    #bounds for "good" pixels in the background
                    low = np.percentile(clipped_bkg[clipped_bkg <= mode_bkg], 50)
                    dpct = mode_bkg-low

                    #make spectral mask
                    mask = np.ones_like(bkg_image, dtype=int)
                    mask[bkg_image < mode_bkg-dpct] = 0
                    mask[bkg_image > mode_bkg+dpct] = 0
                    mask[0,:], mask[-1,:], mask[:,0], mask[:,-1] = 0, 0, 0, 0

                    ### ABORT!! THIS SCRIPT ISN'T DONE YET ###
#                    #re-apply to 
#                    ofile[ifu].data = np.copy(cube)-mode_bkg
#
#                    if logger is not None:
#                        logger.debug('-- Applied background correction to IFU {0}: {1}'.format(ifu, mode_bkg))
#            
#                ofile[ifu].header['HISTORY'] = "Applied background correction: {0:.3f}".format(mode_bkg)
           

    return 0

