from __future__ import print_function
import numpy as np

from astropy.io import fits

from ..utils.fileutils import (get_obs_exposures, duplicate)
from ..utils.obsutils import (hsm, find_lines, reconstruct_wave, band_limits,
                              make_image, make_var_cube, sky_path)


from matplotlib import pyplot as pl

__all__ = ["bkg_correct"]


def get_star_params(logger=None, subdir=None, obsband=None,
                    search_suffix=None, **kwargs):
    """ Measure star parameters (centroid mostly) from a list of KMOS cubes
    """

    if not get_star_shifts:
        return 0

    if logger is not None:
        logger.debug('- Getting star parameters')

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
    for exposure, _, _ in exposures:
        fpath, fname = os.path.split(exposure)

    

	#define un-sky-subtracted exposure for line masking
        exp_sky = os.path.join(sky_dir, "{0}.fits".format(fname[:42]))


