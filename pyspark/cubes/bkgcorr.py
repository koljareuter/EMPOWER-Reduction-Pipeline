from __future__ import print_function
import os
import numpy as np
import bottleneck as bn

from astropy.io import fits

from ..utils.fileutils import (get_obs_exposures, duplicate)
from ..utils.obsutils import (hsm, find_lines, reconstruct_wave, band_limits,
                              make_image, make_var_cube, sky_path)

from ..utils.psfutils import fit_psf

import matplotlib.pyplot as pl


__all__ = ["bkg_correct_2d", "bkg_correct_spec"]


def bkg_correct_2d(logger=None, subdir=None, obsband=None, 
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
                    ofile[ifu].data = np.copy(cube)-mode_bkg

                    if logger is not None:
                        logger.debug('-- Applied background correction to IFU {0}: {1}'.format(ifu, mode_bkg))
            
                ofile[ifu].header['HISTORY'] = "Applied background correction: {0:.3f}".format(mode_bkg)
           

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

