from __future__ import print_function
import bottleneck as bn
import numpy as np
import shutil
import subprocess
import os
import scipy.ndimage as nd

from astropy.io import fits
from multiprocessing import freeze_support#, set_start_method
freeze_support()

__all__ = ["nanMAD", "hl_diff", "hsm", "read_cube", "match_obs_rotator",
           "get_cal_rotator", "get_obs_extensions", "run_spark", "find_lines",
           "reconstruct_wave", "split_sex", "sex_to_deg", "band_limits",
           "make_image", "make_var_cube", "sky_path", "dither_tel2ifu",
           "dither_ifu2tel"]


#static calibration directory based on KMOS_CALIB environment variable
static_cal = os.environ['KMOS_CALIB']


def nanMAD(a, axis=None):
    """Compute the Median Absolute Deviation accounting
    for Inf and NaN values.
    """
    mu = bn.nanmedian(a, axis=axis)
    if axis is not None:
        mu = np.expand_dims(mu, axis=axis)    
    return bn.nanmedian(np.fabs(a-mu), axis=axis)


def hl_diff(a,b):
    """Hodges-Lehmann estimate of the difference between
    two input arrays.
    """
    pairs = a[:,None]-b[None,:]
    return bn.nanmedian(pairs.flatten())


def hsm(a, min_samp=5):
    """Estimate the mode of an input array using the
    the half-sample algorithm.
    """
    a = np.atleast_1d(a)
    a = a[np.isfinite(a)]

    srt = np.argsort(a.ravel())
    cut_array = a.ravel()[srt]
    nsamp_full = cut_array.size

    power = 1
    while nsamp_full/float((2**(power+1))) > min_samp:
        interval = nsamp_full//(2**power)
        
        #find minimum interval
        cut_low, cut_high = cut_array[:interval], cut_array[interval:]
        if cut_low.size > cut_high.size:
            interval_size = cut_high.size
            cut_low = cut_low[:interval_size]
        elif cut_high.size > cut_low.size:
            interval_size = cut_low.size
            cut_high = cut_high[:interval_size]
        else:
            interval_size = interval

        min_interval = np.argmin(cut_high-cut_low)

        #cut down the sample
        cut_array = cut_array[min_interval:min_interval+interval_size]
        power += 1

    return np.mean(cut_array)


def read_cube(data_file, ext=None, fixzero=False):
    
    """ Read in a reconstructed data cube.  Return data, reconstructed wavelength
    array, size information, and header.
    """
    
    file = fits.open(data_file)
    data = file[ext].data
    data = np.asarray(data, dtype=np.float64)
    
    #pull header information for the wavelength array
    hdr = file[ext].header
    
    file.close()
    try:
        #pull cube sizes
        xsize = hdr['NAXIS1']
        ysize = hdr['NAXIS2']
        zsize = hdr['NAXIS3']

        #wavelength array in microns
        wl_array = reconstruct_wave(hdr)

        #set zero values to nan for proper handling
        findzero = np.zeros(data.shape, dtype=bool)
        if fixzero:
            findzero = (data == 0.0)
            if np.count_nonzero(findzero):
                data[findzero] = np.nan
     
        return wl_array, data, xsize, ysize, zsize, hdr, findzero
    except:
        return np.array([]), np.array([]), 0, 0, 0, hdr, None


def get_cal_rotator(cal_file):

    """Fetch the rotation angles from a KMOS calibration file
    returns an array with the observed angles.
    """
    
    pro_naangles = []
    for iter in range(1,100):
        try:
            head = cal_file[iter].header
            pro_naangles.append(head['HIERARCH ESO PRO ROT NAANGLE'])
        except:
            break
    pro_naangles = np.unique(np.array(pro_naangles))
    return pro_naangles


def match_obs_rotator(head, cal_rot, get_second=False, input_rot=None):
    
    """find closest match in a list of calibration rotation angles
    returns appropriate rotator index and closest match.
    """

    if input_rot is not None:
        obs_rot = input_rot
    else:
        obs_rot = head['HIERARCH ESO OCS ROT NAANGLE']

    #find closest from calibration rotator angles
    offset = np.abs(((cal_rot-obs_rot + 540.)%360-180))
    srt = np.argsort(offset)
    rotator = srt[0]
    this_frame_match = int(cal_rot[rotator])

    if obs_rot < 0: obs_rot += 360.

    if get_second: #also get extension for the second closest frame
        return rotator, this_frame_match, (obs_rot), srt[1], int(cal_rot[srt[1]])
    else:
        return rotator, this_frame_match, (obs_rot)


def get_obs_extensions(obs_file):

    """find which extensions (arms for KMOS) contain objects
    returns list of filled IFUs
    """
    
    filled_ifus = []
    for ii in range(1,25):
        try:
            used = obs_file[0].header['HIERARCH ESO OCS ARM{0} NOTUSED'.format(ii)]
        except:
            filled_ifus.append(ii)

    return filled_ifus


def run_spark(routine, args='', sof=None, logger=None, esorex_opts='', env=None):

    """ Execute SPARK routine given by cmd.  esorex_opts go before the recipe
    name (e.g. --recipe-dir=...), env adds environment variables for this call only.
    """

    cmd = 'esorex {0} {1} {2}'.format(routine, args, sof)
    if esorex_opts:
        cmd = 'esorex {0} {1} {2} {3}'.format(esorex_opts, routine, args, sof)
    if logger is not None:
        logger.debug('- Executing command: {0}'.format(cmd))
        if env:
            logger.debug('- with environment {0}'.format(env))

#    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=1, shell=True)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=-1, shell=True,
                         env=dict(os.environ, **env) if env else None)
    for line in iter(p.stdout.readline, b''):
        if logger is not None:
            logger.debug(line.decode('utf-8').strip())
        else:
            continue
    p.wait()
    
    return p.returncode


def find_lines(spec_in):
    """returns an boolean array which selects possible sky line pixels
    """

    #compute derivative spectrum
    derivatives = np.r_[np.array([0.]),np.diff(spec_in)]
    derivatives[~np.isfinite(spec_in) | ~np.isfinite(derivatives)] = 0.
    
    #array of line type flags
    dflags = np.zeros(len(derivatives), dtype=int)

    #identify pixels in the line wings by their large first derivatives
    dnon_zero = np.abs(derivatives[derivatives != 0])

    med = np.median(dnon_zero)
    sig = 1.4826*1.253*np.median(np.abs(dnon_zero-med))

    #work out relative amplitude to detect line given S/N and width
    if sig/med > 4.:
        der_lim = 3.*med
    else:
        der_lim = sig

    dlow, dhigh = -1.5*der_lim, 1.5*der_lim
    dflags[derivatives < dlow] = -1
    dflags[derivatives > dhigh] = 1

    #perform two searches of the flagged pixels, one from the left and one from the right.
    search_right = np.copy(dflags)
    search_left = np.copy(dflags)

    range_flag = 0
    for ii in range(0,len(derivatives)-1):
        if dflags[ii] == 1:
            range_flag = 1
        if (dflags[ii]==-1 and dflags[ii+1]==0) or (dflags[ii]==-1 and dflags[ii+1]==1): 
            range_flag = 0
        search_right[ii] = range_flag

    for ii in range(len(derivatives)-2,0,-1):
        if dflags[ii] == -1:
            range_flag = 1
        if (dflags[ii+1]==1 and dflags[ii]==0) or (dflags[ii+1]==1 and dflags[ii]==-1): 
            range_flag = 0
        search_left[ii] = range_flag
    
    lrange = (search_right == 1) & (search_left == 1)

    return lrange


def reconstruct_wave(header):
    """ Reconstruct wavelength array based on input header.
    """
    zsize = header['NAXIS3']
    crval = header['CRVAL3']
    crpix = header['CRPIX3']
    try:
        cdelt = header['CDELT3']
    except:
        cdelt = header['CD3_3']

    return (np.arange(zsize, dtype=float)+1 - crpix)*cdelt + crval


def split_sex(coord):
    ss, llmm = np.modf(coord/100.)
    mm, ll = np.modf(llmm/100.)
    return (ll, mm*100., ss*100.)


def sex_to_deg(alpha, delta):
    """Convert between hms/dms as in the KMOS headers to decimal degrees
    """
    hh_ra, mm_ra, ss_ra = split_sex(alpha)
    dd_dec, mm_dec, ss_dec = split_sex(delta)

    ra_out = (hh_ra + mm_ra/60. + ss_ra/3600.)*15.
    dec_out = dd_dec + mm_dec/60. + ss_dec/3600.

    return ra_out, dec_out

def band_limits(band):
    #open the calibration file and fetch wavelengths
    rfile = fits.open('{0}/kmos_wave_band.fits'.format(static_cal))
    rtab = rfile[1].data
    rfile.close()
    return rtab[band.upper()]


def make_image(wl, data, noise, mask, llim=1.05, hlim=1.35):
    nwl, nx, ny = data.shape

    left_idx, right_idx = np.searchsorted(wl, llim, side='left'), np.searchsorted(wl, hlim, side='right')
    det_image = np.zeros((nx, ny), dtype=float)

    for jj in range(nx):
        for kk in range(ny):
            this_data = data[:,jj,kk]
            this_var = noise[:,jj,kk]**2

            #generate masking for this object
            good = np.ones(len(this_data), dtype=bool)
            good[(wl > 1.25) & (wl < 1.29)] = False
            good[mask] = False
            good[:left_idx], good[right_idx:] = False, False
            good[this_var <= 0] = False
            good[np.isnan(this_data) | ~np.isfinite(this_data)] = False

            this_data = this_data[good]
            this_var = this_var[good]

            srt = np.argsort(this_data)
            srt_data = this_data[srt]
            srt_ivar = 1./this_var[srt]

            #imean = bn.nanmedian(srt_data)
            try:
                imean = bn.nansum(srt_data*srt_ivar)/bn.nansum(srt_ivar)
            except:
                imean = 0.

            for iter in range(2):
                dev = (srt_data-imean)**2
                srt_iwt = np.copy(srt_ivar)
                outlier = (np.sqrt(dev*srt_ivar) > 3.)
                srt_iwt[outlier] = srt_ivar[outlier]/(1.+(dev*srt_ivar)[outlier])
                try:
                    imean = bn.nansum(srt_data*srt_iwt)/bn.nansum(srt_iwt)
                except:
                    imean = 0.
            det_image[jj,kk] = np.copy(imean)

    det_image[~np.isfinite(det_image)] = 0.
    return det_image


def make_var_cube(input_cube): 
    """ stupid estimate of the noise in the cube using a volumetric 
    maximum filter.  Input should *not* be sky subtracted for this to 
    be of any use what so ever (and even then...)"""
    filt_cube = nd.filters.maximum_filter(input_cube, size=(3,3,3))
    filt_cube[np.isnan(filt_cube) | ~np.isfinite(filt_cube)] = 0.
    return filt_cube


def sky_path(subdir, inputdir=None, **kwargs):
    """ returns proper path for sky files """
    mpath = os.path.join(subdir, 'sky')
    return os.path.join(inputdir, mpath)


def dither_tel2ifu(xoff, yoff, header):
    """
    convert telescope dither to ifu coords.  Takes offsets in pixels
    """
    dxx = float(header['CD1_1'])
    dxy = float(header['CD1_2'])

    dyx = float(header['CD2_1'])
    dyy = float(header['CD2_2'])

    delx, dely = float(header['CDELT1']), float(header['CDELT2'])
    return (dxx*xoff + dxy*yoff)/delx, (dyx*xoff + dyy*yoff)/dely

def dither_ifu2tel(ifu_shiftx, ifu_shifty, header):
    """
    convert dither in ifu coords to sky plane.  Takes shifts in pixels
    """
    dxx = float(header['CD1_1'])
    dxy = float(header['CD1_2'])

    dyx = float(header['CD2_1'])
    dyy = float(header['CD2_2'])

    delx, dely = float(header['CDELT1']), float(header['CDELT2'])

    tx = (ifu_shiftx*delx - ifu_shifty*dely*dxy/dyy)/(dxx - dyx*dxy/dyy)
    ty = (ifu_shifty*dely - ifu_shiftx*delx*dyx/dxx)/(dyy - dxy*dyx/dxx)
    return round(tx, 3), round(ty, 3)

