#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import os, sys
import glob
import argparse
#######!/usr/local/bin/python

from pyspark.utils import (fileutils, setuputils, obsutils)
from pyspark.detector import fix_detector_levels
from pyspark.detector.row_commonmode import row_commonmode_ob
from pyspark.calibration import make_spark_calibs, make_interchannel_ref
from pyspark.cubes import (reconstruct, velocity_correct, illuminate, bkg_correct_2d,
                           telluric_correct)
from pyspark.sky import subtract_sky
from pyspark.cubes.wcssnap import snap_wcs

from multiprocessing import freeze_support#, set_start_method
freeze_support()
#set_start_method('spawn')

def break_check(routine, logger):
    logger.error('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ')
    logger.error('!!!! An error occured during {0}, check the log'.format(routine))
    logger.error('!!!! Moving on to next night')
    logger.error('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ')

    #clean up any dangling sof files
    [os.remove(x) for x in glob.glob('*.sof')]
    

if __name__ == "__main__":
 #handle parsing of command line arguments
 freeze_support()
 parser = argparse.ArgumentParser()
 parser.add_argument('-p', '--parameter_file', type=str, help='paramter file controlling reduction')
 parser.add_argument('--verbose', type=int, default=3, help='Logging level for stdout')
 args = parser.parse_args()

 param_file = args.parameter_file
 verbosity = args.verbose

 # --- initialize log for this reduction session ---
 #master_logger = setuputils.set_logging(logbase='master', log_level=verbosity)
 master_logger = setuputils.set_logging(logbase='master', log_level=3)


 # --- read input configuration file ------------------
 run_params = setuputils.set_config(param_file, logger=master_logger)


 #pull out information of interest for iteration
 rawdir = run_params['rawdir']
 procdir = run_params['procdir']
 folders = run_params['subdirs']
 bands = run_params['obsband']


 # --- start iterating over bands ------------------
 for band in bands:
    master_logger.info('######## Processing {0}-band observations ########'.format(band))

    #iterate over folders
    for folder in folders:

        # before anything, check if there are science files in this
        # folder observed with the given band
        iframes = fileutils.get_sof_list(filetype='science', subdir=folder,
                inputdir=rawdir, obsband=band)

        if not len(iframes):
            master_logger.info("No {0}-band science exposures for {1}, moving on".format(band, folder))
            continue

        master_logger.info('Reducing {0} '.format(folder))

        #make directory structure in case it doesn't exist
        fileutils.verify_structure(outputdir=procdir, subdir=folder)

        #initialize night/band specific log
        dir_logger = setuputils.set_logging(logbase='{0}'.format(band), 
                outputdir=os.path.join(procdir, folder), log_level=verbosity, 
                master_log=master_logger)
       

        # -- calibrations -----------------------
        master_logger.info('- Running calibrations')
        ecode, comp = make_spark_calibs(logger=dir_logger, subdir=folder, obsband=band, 
                          **run_params['calibration'])
        if ecode:
            break_check(comp, master_logger)
            continue
        master_logger.info('-- Finished calibrations')
        master_logger.info(' ')

        # -- interchannel reference calibration --
        if run_params['detector'].get('interchannel_correct', False):
            master_logger.info('- Building interchannel reference')
            try:
                make_interchannel_ref(logger=dir_logger, subdir=folder, obsband=band,
                                      outputdir=procdir)
            except Exception:
                break_check('interchannel reference', master_logger)
                continue
            master_logger.info('-- Finished interchannel reference')
            master_logger.info(' ')


        # -- detector corrections ----------------
        master_logger.info('- Running detector corrections')
        
        #get list of obs frames
        frames = fileutils.get_sof_list(filetype='science', subdir=folder, 
                inputdir=rawdir, obsband=band)
   
        #apply detector fix(es); with the row common-mode correction on, they write the
        #row_commonmode_input prefix and the row step writes the [detector] prefix
        row_cm = run_params['detector'].get('row_commonmode_correct', False)
        det_kw = dict(run_params['detector'])
        if row_cm:
            det_kw['prefix'] = det_kw.get('row_commonmode_input', 'dc')
        if run_params['detector']['channel_correct'] or \
           run_params['detector']['picture_correct'] or \
           run_params['detector']['interchannel_correct']:
            try:
                fix_detector_levels([frame for frame, type in frames], subdir=folder,
                        obsband=band, logger=dir_logger, **det_kw)
            except:
                print('----detector correction failed')
                break_check('Det. correction', master_logger)
                continue
            else:
                master_logger.info('-- Finished detector corrections')
                master_logger.info(' ')
        else:
            master_logger.info('-- Detector corrections skipped')
            master_logger.info(' ')

        # -- row common-mode correction (optional, docs/noise-improvements-2026-10.md) --
        if row_cm:
            master_logger.info('- Running row common-mode correction')
            try:
                row_commonmode_ob(subdir=folder, obsband=band, logger=dir_logger,
                                  **run_params['detector'])     # outputdir = procdir is in there
            except Exception:
                dir_logger.exception('row common-mode correction failed')
                break_check('row common-mode correction', master_logger)
                continue
            master_logger.info('-- Finished row common-mode correction')
            master_logger.info(' ')

   
   
        # -- reconstruct -----------------------
        master_logger.info('- Running reconstruction')
        ecode = reconstruct(logger=dir_logger, subdir=folder, obsband=band, 
                            **run_params['cubes'])
        if ecode:
            break_check('reconstruction', master_logger)
            continue
        master_logger.info('-- Finished reconstruction')
        master_logger.info(' ')
  
  
        # -- sky subtraction -------------------
        master_logger.info('- Running sky subtraction')
        try:
            ecode = subtract_sky(logger=dir_logger, subdir=folder, obsband=band,
                                 **run_params['cubes'])
        except Exception:
            master_logger.exception('sky subtraction raised an exception (likely no matching sky exposures for {0})'.format(folder))
            break_check('sky_subtraction', master_logger)
            continue
        if ecode:
            break_check('sky_subtraction', master_logger)
            continue
        master_logger.info('-- Finished sky subtraction')
        master_logger.info(' ')
  
   
        # -- illumination correction -----------
        master_logger.info('- Applying illumination correction')
        ecode = illuminate(logger=dir_logger, subdir=folder, obsband=band, 
                           search_suffix='s', **run_params['cubes'])
        if ecode:
            break_check('illumination correction', master_logger)
            continue
        master_logger.info('-- Finished illumination correction')
        master_logger.info(' ')
  
  
        # -- global background correction
        master_logger.info('- Applying global background correction')
        ecode = bkg_correct_2d(logger=dir_logger, subdir=folder, obsband=band,
                            search_suffix='s_i', **run_params['cubes'])
        if ecode:
            break_check('background correction', master_logger)
            continue
        master_logger.info('-- Finished background correction')
        master_logger.info(' ')
    
  
        # -- barycentric velocity correction -----
        master_logger.info('- Applying heliocentric correction')
        ecode = velocity_correct(logger=dir_logger, subdir=folder, obsband=band,
                                 search_suffix='s_i_bkg2d', **run_params['cubes'])
        if ecode:
            break_check('velocity correction', master_logger)
            continue
        master_logger.info('-- Finished heliocentric correction')
        master_logger.info(' ')

        # -- integer-pixel registration for the combination (optional, docs/noise-improvements-2026-10.md) --
        if run_params['cubes'].get('snap_wcs', False):
            master_logger.info('- Snapping IFU WCS for integer-shift combination')
            try:
                ecode = snap_wcs(logger=dir_logger, subdir=folder, obsband=band, **run_params['cubes'])
            except Exception:
                dir_logger.exception('snap_wcs failed')
                ecode = 1
            if ecode:
                break_check('snap_wcs', master_logger)
                continue
            master_logger.info('-- Finished snap_wcs')
            master_logger.info(' ')


        # -- telluric correction (Python step; _rv.fits → _rv_tell.fits) -----
        if run_params['cubes'].get('telluric_in_sof', False):
            master_logger.info('- Applying telluric correction')
            ecode = telluric_correct(logger=dir_logger, subdir=folder, obsband=band,
                                     **run_params['cubes'])
            if ecode:
                break_check('telluric correction', master_logger)
                continue
            master_logger.info('-- Finished telluric correction')
            master_logger.info(' ')


        master_logger.info('- Finished reducing {0} '.format(folder))
  
  