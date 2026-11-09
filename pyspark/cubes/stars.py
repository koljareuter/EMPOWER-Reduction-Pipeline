from __future__ import print_function
import numpy as np
import os
import bottleneck as bn
import scipy.optimize as opt

from astropy.io import fits

from ..utils.fileutils import (get_obs_exposures, duplicate, update_header)
from ..utils.obsutils import (find_lines, reconstruct_wave, band_limits,
                              make_image, make_var_cube, sky_path, read_cube,
                              dither_tel2ifu)
from ..utils.psfutils import fit_psf


__all__ = ["get_star_params"]


def get_star_ifus(exposure, star_prefix=None):
    """ given an input frame, identify IFUs that contain stars based
    on the prefix string/list given by star_prefix """

    #import the first frame header and iterate over object 
    #names to identify stars.

    hdr = fits.getheader(exposure, ext=0)
    print(star_prefix)
    star_arms = []
    for ifu in range(1,25):
        arm_name = hdr['HIERARCH ESO OCS ARM{0} NAME'.format(ifu)]
        print(arm_name)
        if len([arm_name.find(star_string) for star_string in star_prefix if arm_name.find(star_string) >= 0]):
            star_arms.append(ifu)
    return star_arms


def get_star_params(logger=None, subdir=None, obsband=None,
                    search_suffix=None, star_prefix=None, **kwargs):
    """ Measure star parameters (centroid mostly) from a list of KMOS cubes
    """

    if not get_star_params or star_prefix is None:
        return 0


    if logger is not None:
        logger.debug('- Measuring parameters of monitoring stars')

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

    # iterate over each exposure measuring star parameters
    for exposure, _, _, _ in exposures:
        fpath, fname = os.path.split(exposure)

        if logger is not None:
            logger.debug('-- Processing stars in {0}'.format(fname))
	
        #define un-sky-subtracted exposure for line masking
        exp_sky = os.path.join(sky_dir, "{0}.fits".format(fname[:42]))

        #fetch IFUs hosting stars
        star_ifus = get_star_ifus(exposure, star_prefix=star_prefix)
        print(star_ifus)
        #get the correction for this pointing
        header = fits.getheader(exposure, ext=0)
        xoff = float(header['HIERARCH ESO OCS TARG DITHA'])/0.2
        yoff = float(header['HIERARCH ESO OCS TARG DITHD'])/0.2

        #estimate of seeing in sigma(pixels)
        try:
            seeing = float(header['HIERARCH ESO TEL IA FWHMLIN'])*0.7/0.2
        except:
            seeing = float(header['HIERARCH ESO TEL IA FWHM'])*0.7/0.2
        seeing = max(0.6*2.355,seeing)

        header_cx, header_cy = 0., 0.
        header_dx, header_dy = 0., 0.
        obs_cx, obs_cy = 0., 0.
        obs_nstars = 0

        #iterate over each IFU which contains a star:
        for sifu in star_ifus:
            wavelength, idata, _, _, _, shdr, _ = read_cube(exposure, ext=sifu)
            _, isky, _, _, _, _, _ = read_cube(exp_sky, ext=sifu) #corresponding sky file

            #generate median sky spectrum 
            med_sky = bn.nanmedian(isky.reshape(2048,-1), axis=1)
            msky = find_lines(med_sky)

            #estimate of variance cube
            vcube = make_var_cube(isky)

            #generate collapsed image
            band_low, band_high = band_limits(obsband)
            star_image = make_image(wavelength, idata, vcube, msky, llim=band_low+0.05, hlim=band_high-0.05)

            #measure star centroid and Gaussian FWHM
            star_params, _ = fit_psf(star_image)

            #get detector for this IFU
            detector = (sifu-1)//8 + 1

            ##compute appropriate header offset based on dither and inst. rotation
            refx, refy = shdr['CRPIX1'], shdr['CRPIX2']
            rot_x, rot_y = dither_tel2ifu(xoff, yoff, shdr)
            header_cx += rot_x + refx
            header_cy += refy - rot_y

            if (0 < star_params[0] < 14) and (0 < star_params[1] < 14) \
               and (0.2 < star_params[2]*0.2 < 1.8) and (0.2 < star_params[3]*0.2 < 1.8):
                    #the star is fine, use it
                    obs_nstars += 1
                    obs_cx += star_params[0]
                    obs_cy += star_params[1]

        #only use star shifts if a reasonable fraction have been
        #fitted successfully
        if obs_nstars > np.floor(len(star_ifus)//2):
            frame_x = obs_cx/obs_nstars
            frame_y = obs_cy/obs_nstars
        else:
            frame_x = header_cx/len(star_ifus)
            frame_y = header_cy/len(star_ifus)
            logger.warning('--- Problem with star centroids, adopting header positions')
        logger.debug('--- Derived average star centroids: {0:.2f}, {1:.2f}'.format(frame_x, frame_y))

        #insert values into the header
        update_header(exposure, {'HIERARCH ESO PYSPARK PSF XPOS':frame_x,
                                 'HIERARCH ESO PYSPARK PSF YPOS':frame_y,
                                 'HISTORY':'Derived star centroids'
                                 })

        


