from __future__ import print_function
import numpy as np
from astropy.io import fits
import os

from ..utils.fileutils import (get_sof_list, clean_move)
from ..utils.obsutils import (get_cal_rotator, run_spark)
from ..utils.setuputils import id_gen

import matplotlib.pyplot as pl

__all__ = ["make_interchannel_ref", "generate_channel_cube", "reconstruct_channel_cube"]


#static calibration directory based on KMOS_CALIB environment variable
static_cal = os.environ['KMOS_CALIB']


def make_interchannel_ref(logger=None, subdir=None, obsband=None, **kwargs):
    """ Make a set of reference frames that can be used to identify detector pixels
    that fall outside of the IFU slices and can be used for a second order reference pixel
    correction.
    """

    #path for the output calibration directory
    caldir = os.path.join(kwargs['outputdir'], subdir, 'calib')
   
    #open up the badpixel dark frames and read them in
    badpixel_mask = np.zeros((3,2048,2048), dtype=bool)
    with fits.open(os.path.join(caldir, 'BADPIXEL_DARK.fits')) as ffile:
        for detector in range(3):
            tframe = np.array(ffile[detector+1].data, dtype=int)
            badpixel_mask[detector,:,:] = (tframe == 1)

    #fetch rotator angles and filled extensions from the master flat file
    flat_file = fits.open(os.path.join(caldir, 'MASTER_FLAT_{0}{0}{0}.fits'.format(obsband.upper())))
    rotators = get_cal_rotator(flat_file)#[0]

    islice_mask = np.zeros((3,2048,2048), dtype=bool)
    for detector in range(3): #iterate over each detector
        track_det = np.zeros((2048,2048), dtype=int)
        for rotator in range(len(rotators)): #iterate over each rotator angle in the master flat
            findex = (2*detector)+1 + 6*rotator
            dimage = np.array(flat_file[findex].data, dtype=float)
            track_det += np.isnan(dimage)*1

        slice_mask = (track_det == len(rotators))
        islice_mask[detector] = (slice_mask & badpixel_mask[detector,:,:])

    #write output frame
    outfile = os.path.join(caldir, 'master_channel_{0}{0}{0}.fits'.format(obsband))
    hdulist = fits.HDUList(fits.PrimaryHDU())
    for detector in range(3):
        hdulist.append(fits.ImageHDU(data=np.array(islice_mask[detector], dtype=np.int32)))
    hdulist.writeto(outfile, overwrite=True)

    return 0



def reconstruct_channel_cube(channel_frame, logger=None, obsband=None, caldir=None):

    #write temporary .sof file to feed to kmo_sci_red
    outfile = '{0}.sof'.format(id_gen())
    
    if logger is not None:
        logger.debug('- Writing channel frame temporary file {0}'.format(outfile))

    with open(outfile, 'wt') as ofile:
        ofile.write('{0} SCIENCE\n'.format(channel_frame))
        logger.debug('-- Added {0} SCIENCE'.format(channel_frame))

        #include additional files required for SPARK
        ofile.write('{0} XCAL\n'.format(os.path.join(caldir, 'xcal_{0}{0}{0}.fits'.format(obsband.upper()))))
        ofile.write('{0} YCAL\n'.format(os.path.join(caldir, 'ycal_{0}{0}{0}.fits'.format(obsband.upper()))))
        ofile.write('{0} LCAL\n'.format(os.path.join(caldir, 'lcal_{0}{0}{0}.fits'.format(obsband.upper()))))
        ofile.write('{0} MASTER_FLAT\n'.format(os.path.join(caldir, 'master_flat_{0}{0}{0}.fits'.format(obsband.upper()))))
        ofile.write('{0} WAVE_BAND\n'.format(os.path.join(static_cal, 'kmos_wave_band.fits')))
        
        if logger is not None:
            logger.debug('-- Added {0} XCAL'.format(os.path.join(caldir, 'xcal_{0}{0}{0}.fits'.format(obsband.upper()))))
            logger.debug('-- Added {0} YCAL'.format(os.path.join(caldir, 'ycal_{0}{0}{0}.fits'.format(obsband.upper()))))
            logger.debug('-- Added {0} LCAL'.format(os.path.join(caldir, 'lcal_{0}{0}{0}.fits'.format(obsband.upper()))))
            logger.debug('-- Added {0} MASTER_FLAT'.format(os.path.join(caldir, 'master_flat_{0}{0}{0}.fits'.format(obsband.upper()))))
            logger.debug('-- Added {0} WAVE_BAND'.format(os.path.join(static_cal, 'kmos_wave_band.fits')))

    #control args
    routine_args = '-background=FALSE -tbsub=FALSE -imethod="CS" '
    
    #sky method
    routine_args += '-no_subtract=FALSE -sky_tweak=FALSE '
    routine_args += '-no_combine=FALSE '
    
    #execute spark script
    if logger is not None:
        logger.debug('- Executing reconstruction')
    ecode = run_spark('kmo_sci_red', args=routine_args, sof=outfile, logger=logger)
    if logger is not None:
        logger.debug('-- Complete')

    #check the exit status of the SPARK call
    if ecode != 0:
        return 1


def generate_channel_cube(logger=None, subdir=None, obsband=None, **kwargs):

    """ Generate calibration channel cubes for each angle present in the 
    master flat.  
    """

    #path for the output calibration directory
    caldir = os.path.join(kwargs['outputdir'], subdir, 'calib')

    #fetch rotator angles and filled extensions from the master flat file
    flat_file = fits.open(os.path.join(caldir, 'MASTER_FLAT_{0}{0}{0}.fits'.format(obsband.upper())))
    pro_naangles = get_cal_rotator(flat_file)
    nrot = len(pro_naangles)
    flat_file.close()

    #pull header from an example frame for reconstruction
    #manually change the rotator angle later to match calibration position(s)
    frames = get_sof_list(filetype='science', subdir=subdir, obsband=obsband, **kwargs)
    store_hdr = []
    for ihdr in range(4):
        store_hdr.append(fits.getheader(frames[0][0], ext=ihdr))

    #iterate over each rotation angle
    for irot, rot in enumerate(pro_naangles):
        logger.debug('-- Creating channel cubes at {0} deg rotation'.format(rot))

        store_hdr[0]['HIERARCH ESO OCS ROT NAANGLE'] = rot
        
        for ichan in range(1):

            #make mock channel array
            chan_array = np.ones((2048,2048), dtype=float) * 1e-5
            chan_array[:,ichan*64:ichan*64+64] = 1.
            
            #write temporary .sof file to feed to
            chanout = '{0}.fits'.format(id_gen())

            #blank primary header
            hdulist = fits.HDUList()
            hdulist.append(fits.PrimaryHDU(header=store_hdr[0]))
            for detector in range(3):
                hdulist.append(fits.ImageHDU(data=np.array(chan_array, dtype=np.float32), header=store_hdr[detector+1]))
            hdulist.writeto(chanout)

            #reconstruct this channel cube
            reconstruct_channel_cube(chanout, logger=logger, obsband=obsband, caldir=caldir)

#
#        frames = get_sof_list(filetype='flat', subdir=subdir, obsband=obsband, rotangle=rot, **kwargs)
#
#        #write temporary .sof file to feed to
#        outfile = '{0}.sof'.format(id_gen())
#        logger.debug('-- Writing illumination frames to temporary file {0}'.format(outfile))
#
#
#
#
#
#d = np.zeros((2048,2048), dtype=np.float32)
#for detector in range(1,4):
#    for ii in range(32):
#        d[:,ii*64:ii*64+64] = (ii+1) + 32.*(detector-1)
#    fits.update('darktrace.fits', data=np.asarray(d, dtype=np.float32), \
#            header=fits.getheader('{0}/dc_{1}'.format(kmos_data,this_file_name), ext=detector), ext=detector)
#    
#
