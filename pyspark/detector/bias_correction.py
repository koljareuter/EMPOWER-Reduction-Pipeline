from __future__ import print_function
import numpy as np
import bottleneck as bn
import os
import astropy.io.fits as fits

from astropy.stats.biweight import (biweight_location, biweight_scale)

from ..utils.obsutils import (hl_diff, hsm)
from ..utils.fileutils import (fetch_resource_path, clean_copy, duplicate)
from ..utils.setuputils import get_date

from .._version import __version__

from astropy.convolution import (convolve, Box1DKernel)


__all__ = ["get_ref_channel_correct", "get_picture_correct", "fix_detector_levels",
           "get_interchannel_correct"]


#static calibration directory based on KMOS_CALIB environment variable
static_cal = os.environ['KMOS_CALIB']

def get_ref_channel_correct(input_frame):
    """Given an input exposure, estimate and remove individual H2RG channel 
    drifts using reference pixels.  Applies both a row and column correction.
    
    Parameters
    ----------
    input_frame : ndarray
        Input (raw) detector image.

    Returns
    -------
    ref_correct : ndarray
        Correction to be applied for row and column shifts.
   """

    #first pixel is always a bit funny
    input_frame[2047,0] = np.nan

    #rip out the reference columns (skip the first and last)
    cols = np.zeros((2048,8), dtype=np.float32)
    cols[:,:4] = input_frame[:,:4]
    cols[:,4:] = input_frame[:,-4:]

    #compute the mean
    col = bn.nanmean(np.vstack((bn.nanmedian(cols[:,:4],axis=1),\
            bn.nanmedian(cols[:,4:],axis=1))), axis=0)

    #estimate mean of column pixels
    col -= bn.nanmean(col[4:-4])
    col[~np.isfinite(col)] = 0.

    #reconstruct the 2d column correction
    col[:4], col[-4:] = 0.0, 0.0
    ccorr = np.atleast_2d(col).T
    ccorr = np.repeat(ccorr, 2048, axis=1)
    ccorr[:,:4], ccorr[:,-4:] = 0., 0.
    ccorr[:4,:], ccorr[-4:,:] = 0., 0.

    ##produce the column-corrected cube
    rows = input_frame[-4:,:]

    oe_oned = np.zeros(2048, dtype=np.float32)
    for ii in range(1,32):
        this_channel = rows[:,ii*64:ii*64+64]
        prev_channel = rows[:,(ii-1)*64:(ii-1)*64+64]
        e_channel = this_channel[:,::2].flatten()
        o_channel = this_channel[:,1::2].flatten()

        oe_diff = hl_diff(e_channel, o_channel)
        oe_corr = np.r_[e_channel, o_channel+oe_diff]
        diff = hl_diff(prev_channel.flatten(), oe_corr)

        oe_oned[ii*64:ii*64+64:2] = -diff
        oe_oned[ii*64+1:ii*64+64:2] = -(diff+oe_diff)
        rows[:,ii*64:ii*64+64:2] += diff
        rows[:,ii*64+1:ii*64+64:2] += (diff+oe_diff)
        
    oe_oned -= bn.nanmean(oe_oned)
    oe_oned[:4], oe_oned[-4:] = 0.0, 0.0

    rcorr = np.atleast_2d(oe_oned)
    rcorr = np.repeat(rcorr, 2048, axis=0)

    rcorr[:,:4], rcorr[:,-4:] = 0., 0.
    rcorr[:4,:], rcorr[-4:,:] = 0., 0.

    #corrected = np.copy(input_frame) - rcorr + ccorr
    return rcorr-ccorr


def get_picture_correct(input_frame, detector=None):

    """ Apply correction for the H2RG "picture frame" effect based
    on the correlation between reference pixel values and residual
    bias signal.  Correction is based on a stack of a few thousand
    60s dark frames.
    """

    #fetch the superdark reference frame for this detector
    fpath = fetch_resource_path('data/superdark_coeffs_mean.fits.fz') 

    #pull the information for the picture frame correction
    sdfile = fits.open(fpath)
    picture_coeff = np.array(sdfile[detector].data, dtype=np.float32)
    sdfile.close() 

    #lookup values for reference classification
    idx, idy = np.meshgrid(range(2048), range(2048))
    ref_select = ((idx < 4) | (idx > 2043) | (idy > 2043)) & (idy > 3)
    ref_vals = np.array([-100, -4.5, -1.5, 1.5, 4.5, 100])

    #picture frame scaling based on median of input reference pixels
    reference_coeff = np.median(input_frame[ref_select & (input_frame != 0.) & np.isfinite(input_frame)])

    return picture_coeff*reference_coeff



def get_interchannel_correct(input_frame, detector=None, caldir=None, obsband=None):

    #get the mask from the interslice calibration file
    slice_mask = fits.getdata(os.path.join(caldir, 'master_channel_{0}{0}{0}.fits'.format(obsband)),ext=detector)

    #iterate over each slice in on this detector
    cvals = np.zeros(32, dtype=np.float)
    for chan in range(32):
        this_channel = input_frame[:,chan*64:chan*64+64]
        this_mask = slice_mask[:,chan*64:chan*64+64]

        trimmed_channel = this_channel[this_mask == 1]
        
        #remove upper 50 percentile
        val = hsm(trimmed_channel[trimmed_channel <= np.median(trimmed_channel)])
        cvals[chan] = val

    filt = np.copy(cvals)
    norm = convolve(np.ones(len(cvals)-2), Box1DKernel(4))
    filt[1:-1] = convolve(cvals[1:-1], Box1DKernel(4))/norm


    #normalize to zero mean, excluding crazy outliers
    cvals -= filt
    med = biweight_location(cvals)
    sig = biweight_scale(cvals)

    cvals -= med
    cvals[(np.fabs(cvals)/sig > 3.)] = 0.

    #reconstruct the correction array
    tcorr = np.zeros((2048,2048), dtype=np.float)
    for chan in range(32):
        tcorr[:,chan*64:chan*64+64] = cvals[chan]

    return tcorr


def fix_detector_levels(frame_list, subdir=None, inputdir=None, outputdir=None, prefix=None, 
        channel_correct=True, picture_correct=True, interchannel_correct=False, logger=None, 
        obsband=None, **kwargs):

    """ Given an input set of frames, apply detector-level 
    corrections and write out the new set of frames.
    """

    nframes = len(frame_list)


    #path for the output calibration directory
    caldir = os.path.join(outputdir, subdir, 'calib')

    #iterate over each frame and apply requested corrections
    for frame in frame_list:
        fname = os.path.split(frame)[1]
        #make a copy of the input frame to outputdir/subdir
        if logger is not None:
            logger.debug('- Copying {0} to {1}'.format(frame.split('/')[-1], '_'.join([prefix, fname])))
        outname = duplicate(frame, outputdir=os.path.join(outputdir, subdir), prefix=prefix)
        
        #perform any detector corrections (if requested)
        if channel_correct or picture_correct or interchannel_correct:
            fdata = fits.open(outname, mode='update')

            fdata[0].header['history'] = 'Processed by PySPARK {0} {1}'.format(__version__, get_date())

            #iterate over each detector (assumes KMOS)
            for det in range(1,4):
                channel_array = 0.
                picture_array = 0.
                interslice_array = 0.

                orig_data = np.array(fdata[det].data)
                orig_header = fdata[det].header

                if channel_correct:
                    logger.debug('-- Applying reference correction to detector {0}'.format(det))
                    channel_array = get_ref_channel_correct(orig_data)
                    orig_header['history'] = 'Applied channel correction {0}'.format(get_date()) 

                if interchannel_correct:
                    #optional correction using interslice pixels
                    logger.debug('-- Applying inter-channel correction to detector {0}'.format(det))
                    interslice_array = get_interchannel_correct(orig_data-channel_array, detector=det, caldir=caldir,
                                                                obsband=obsband)
                    orig_header['history'] = 'Applied interchannel correction {0}'.format(get_date()) 

                if picture_correct:
                    logger.debug('-- Applying picture frame correction to detector {0}'.format(det))
                    picture_array = get_picture_correct(orig_data, detector=det)
                    orig_header['history'] = 'Applied picture frame correction {0}'.format(get_date()) 

                #update the data in place
                total_correction = channel_array + picture_array + interslice_array
                #print('total correction min/max', np.nanmin(total_correction), np.nanmax(total_correction))

                fdata[det].data = orig_data - channel_array - picture_array - interslice_array
            
            fdata.flush()
    return



