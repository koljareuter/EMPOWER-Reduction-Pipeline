from __future__ import print_function
import numpy as np
from astropy.io import fits
import os

from .telluric_standard import spark_standard

from ..utils.fileutils import (get_sof_list, clean_move)
from ..utils.obsutils import (get_cal_rotator, run_spark)
from ..utils.setuputils import id_gen


__all__ = ["make_spark_calibs", "spark_dark", "spark_flat", "spark_arc", 
           "spark_illum"]

#static calibration directory based on KMOS_CALIB environment variable
static_cal = os.environ['KMOS_CALIB']


def spark_dark(logger=None, subdir=None, obsband=None, **kwargs):

    """ Generate .sof file and Execute esorex dark routine.
    """
    
    #get list of dark frames for processing
    frames = get_sof_list(filetype='dark', subdir=subdir, logger=logger, **kwargs)

    #write temporary .sof file to feed to
    outfile = '{0}.sof'.format(id_gen())
    logger.debug('- Writing frames to temporary file {0}'.format(outfile))

    with open(outfile, 'wt') as ofile:
        for frame, ftype in frames:
            ofile.write('{0} DARK\n'.format(frame))
            logger.debug('-- Added {0} DARK'.format(frame))
         
    #execute spark script
    logger.debug('- Executing kmos_dark')
    ecode = run_spark('kmos_dark', sof=outfile, logger=logger)
    logger.debug('-- Complete')
   
    #check the exit status of the SPARK call
    if ecode != 0:
        return 1

    #move files to their final location
    #products = ['badpixel_dark.fits', 'master_dark.fits']
    products = ['BADPIXEL_DARK.fits', 'MASTER_DARK.fits']
    for product in products:
        clean_move(product, os.path.join(kwargs['outputdir'], subdir, 'calib'))

    #move the .sof file for logging
    clean_move(outfile, os.path.join(kwargs['outputdir'], subdir, 'sof', '{0}_dark_{1}.sof'\
            .format(subdir, obsband.lower())))

    return 0

def spark_flat(logger=None, subdir=None, obsband=None, **kwargs):

    """ Execute esorex flat routine.
    """

    #path for the output calibration directory
    caldir = os.path.join(kwargs['outputdir'], subdir, 'calib')
    
    #get list of flat frames for processing
    frames = get_sof_list(filetype='flat', subdir=subdir, obsband=obsband, **kwargs)

    #write temporary .sof file to feed to
    outfile = '{0}.sof'.format(id_gen())
    logger.debug('- Writing flat frames to temporary file {0}'.format(outfile))

    with open(outfile, 'wt') as ofile:
        for frame, ftype in frames:
            if ftype == 'FLAT,LAMP':
                line_out = '{0} FLAT_ON'.format(frame)
            if ftype == 'FLAT,OFF':
                line_out = '{0} FLAT_OFF'.format(frame)
            ofile.write('{0}\n'.format(line_out))
            logger.debug('-- Added {0}'.format(line_out))

        #include additional files required for SPARK
        badpixel_file = os.path.join(caldir, 'BADPIXEL_DARK.fits')
        ofile.write('{0} BADPIXEL_DARK\n'.format(badpixel_file))
        logger.debug('-- Added {0} BADPIXEL_DARK'.format(badpixel_file))

    #control args, overridable via flat_args in the [calibration] cfg section
    #(see docs/det3-iz-flat-bugfix.md)
    #routine_args = '-cmethod=median'
    routine_args = kwargs.get('flat_args', '-cmethod=ksigma -badpix_thresh=10')

    #execute spark script
    logger.debug('- Executing kmos_flat')
    ecode = run_spark('kmos_flat', args=routine_args, sof=outfile, logger=logger)
    logger.debug('-- Complete')

    #check the exit status of the SPARK call
    if ecode != 0:
        return 1

    #move files to their final location
    products = ['MASTER_FLAT_{0}{0}{0}.fits', 'XCAL_{0}{0}{0}.fits', 'YCAL_{0}{0}{0}.fits',
                'FLAT_EDGE_{0}{0}{0}.fits', 'BADPIXEL_FLAT_{0}{0}{0}.fits']
    for product in products:
        clean_move(product.format(obsband.upper()), caldir)

    #move the .sof file for logging
    clean_move(outfile, os.path.join(kwargs['outputdir'], subdir, 'sof', '{0}_flat_{1}.sof'\
            .format(subdir, obsband.lower())))
    
    return 0


def spark_arc(logger=None, subdir=None, obsband=None, **kwargs):

    """ Execute esorex flat routine.
    """

    #path for the output calibration directory
    caldir = os.path.join(kwargs['outputdir'], subdir, 'calib')

    #get list of flat frames for processing
    frames = get_sof_list(filetype='arc', subdir=subdir, obsband=obsband, **kwargs)

    #write temporary .sof file to feed to
    outfile = '{0}.sof'.format(id_gen())
    logger.debug('- Writing arc frames to temporary file {0}'.format(outfile))

    with open(outfile, 'wt') as ofile:
        for frame, ftype in frames:
            if ftype == 'WAVE,LAMP':
                line_out = '{0} ARC_ON'.format(frame)
            if ftype == 'WAVE,OFF':
                line_out = '{0} ARC_OFF'.format(frame)
            ofile.write('{0}\n'.format(line_out))
            logger.debug('-- Added {0}'.format(line_out))

        #include additional files required for SPARK
        ofile.write('{0} BADPIXEL_FLAT\n'.format(os.path.join(caldir, 'BADPIXEL_FLAT_{0}{0}{0}.fits'.format(obsband.upper()))))
        logger.debug('-- Added {0} BADPIXEL_FLAT'.format(os.path.join(caldir, 'BADPIXEL_FLAT_{0}{0}{0}.fits'.format(obsband.upper()))))
        ofile.write('{0} FLAT_EDGE\n'.format(os.path.join(caldir, 'FLAT_EDGE_{0}{0}{0}.fits'.format(obsband.upper()))))
        logger.debug('-- Added {0} FLAT_EDGE'.format(os.path.join(caldir, 'FLAT_EDGE_{0}{0}{0}.fits'.format(obsband.upper()))))
        ofile.write('{0} MASTER_FLAT\n'.format(os.path.join(caldir, 'MASTER_FLAT_{0}{0}{0}.fits'.format(obsband.upper()))))
        logger.debug('-- Added {0} MASTER_FLAT'.format(os.path.join(caldir, 'MASTER_FLAT_{0}{0}{0}.fits'.format(obsband.upper()))))
        ofile.write('{0} XCAL\n'.format(os.path.join(caldir, 'XCAL_{0}{0}{0}.fits'.format(obsband.upper()))))
        logger.debug('-- Added {0} XCAL'.format(os.path.join(caldir, 'XCAL_{0}{0}{0}.fits'.format(obsband.upper()))))
        ofile.write('{0} YCAL\n'.format(os.path.join(caldir, 'YCAL_{0}{0}{0}.fits'.format(obsband.upper()))))
        logger.debug('-- Added {0} YCAL'.format(os.path.join(caldir, 'YCAL_{0}{0}{0}.fits'.format(obsband.upper()))))
        ofile.write('{0} ARC_LIST\n'.format(os.path.join(static_cal, 'kmos_ar_ne_list_{0}.fits'.format(obsband.lower()))))
        logger.debug('-- Added {0} ARC_LIST'.format(os.path.join(static_cal, 'kmos_ar_ne_list_{0}.fits'.format(obsband.lower()))))
        ofile.write('{0} WAVE_BAND\n'.format(os.path.join(static_cal, 'kmos_wave_band.fits')))
        logger.debug('-- Added {0} WAVE_BAND'.format(os.path.join(static_cal, 'kmos_wave_band.fits')))
        ofile.write('{0} REF_LINES\n'.format(os.path.join(static_cal, 'kmos_wave_ref_table.fits')))
        logger.debug('-- Added {0} REF_LINES'.format(os.path.join(static_cal, 'kmos_wave_ref_table.fits')))

    #execute spark script
    logger.debug('- Executing kmos_wave_cal')
    ecode = run_spark('kmos_wave_cal', sof=outfile, logger=logger)
    logger.debug('-- Complete')

    #check the exit status of the SPARK call
    if ecode != 0:
        return 1

    #move files to their final location
    products = ['LCAL_{0}{0}{0}.fits', 'DET_IMG_WAVE_{0}{0}{0}.fits']
    for product in products:
        clean_move(product.format(obsband.upper()), caldir)

    #move the .sof file for logging
    clean_move(outfile, os.path.join(kwargs['outputdir'], subdir, 'sof', '{0}_arc_{1}.sof'\
            .format(subdir, obsband.lower())))
    return 0


def spark_illum(logger=None, subdir=None, obsband=None, **kwargs):

    """ Generate (internal) illumination images for each of the angles
    present in the master flat.
    """
    
    #path for the output calibration directory
    caldir = os.path.join(kwargs['outputdir'], subdir, 'calib')
    
    #fetch rotator angles and filled extensions from the master flat file
    flat_file = fits.open(os.path.join(caldir, 'MASTER_FLAT_{0}{0}{0}.fits'.format(obsband)))
    pro_naangles = get_cal_rotator(flat_file)
    nrot = len(pro_naangles)
    flat_file.close()
    
    for irot, rot in enumerate(pro_naangles):
        logger.debug('- Processing frames at {0} deg rotation'.format(rot))
        frames = get_sof_list(filetype='flat', subdir=subdir, obsband=obsband, rotangle=rot, **kwargs)

        #write temporary .sof file to feed to
        outfile = '{0}.sof'.format(id_gen())
        logger.debug('-- Writing illumination frames to temporary file {0}'.format(outfile))

        with open(outfile, 'wt') as ofile:
            for frame, ftype in frames:
                if ftype == 'FLAT,LAMP':
                    ofile.write('{0} FLAT_ON\n'.format(frame))
                    logger.debug('--- Added {0} FLAT_ON'.format(frame))

            #include additional files required for SPARK
            ofile.write('{0} XCAL\n'.format(os.path.join(caldir, 'XCAL_{0}{0}{0}.fits'.format(obsband.upper()))))
            logger.debug('--- Added {0} XCAL'.format(os.path.join(caldir, 'XCAL_{0}{0}{0}.fits'.format(obsband.upper()))))
            ofile.write('{0} YCAL\n'.format(os.path.join(caldir, 'YCAL_{0}{0}{0}.fits'.format(obsband.upper()))))
            logger.debug('--- Added {0} YCAL'.format(os.path.join(caldir, 'YCAL_{0}{0}{0}.fits'.format(obsband.upper()))))
            ofile.write('{0} LCAL\n'.format(os.path.join(caldir, 'LCAL_{0}{0}{0}.fits'.format(obsband.upper()))))
            logger.debug('--- Added {0} LCAL'.format(os.path.join(caldir, 'LCAL_{0}{0}{0}.fits'.format(obsband.upper()))))
            ofile.write('{0} WAVE_BAND\n'.format(os.path.join(static_cal, 'kmos_wave_band.fits')))
            logger.debug('--- Added {0} WAVE_BAND'.format(os.path.join(static_cal, 'kmos_wave_band.fits')))


        #control args
        routine_args = '-cmethod=average'

        #execute spark script
        logger.debug('-- Executing kmos_illumination')
        ecode = run_spark('kmos_illumination', args=routine_args, sof=outfile, logger=logger)
        logger.debug('--- Complete')

        #check the exit status of the SPARK call
        if ecode != 0:
            return 1

        #move files to their final location
        products = ['ILLUM_CORR_{0}{0}{0}{1}.fits']
        for product in products:
            clean_move(product.format(obsband.upper(), ''), 
                       os.path.join(caldir, product.format(obsband.upper(), '_{0:03}'.format(int(rot)))))

        #move the .sof file for logging
        clean_move(outfile, os.path.join(kwargs['outputdir'], subdir, 'sof', '{0}_intflat_{1}_{2:03}.sof'\
                .format(subdir, obsband.lower(), int(rot))))

    return 0


def spark_reconstruct_arc(logger=None, subdir=None, obsband=None, **kwargs):

    """ Reconstruct a single arc frame to use for resolution measurement.
    """
    #path for the output calibration directory
    caldir = os.path.join(kwargs['outputdir'], subdir, 'calib')
    
    #fetch rotator angles and filled extensions from the master flat file
    flat_file = fits.open(os.path.join(caldir, 'MASTER_FLAT_{0}{0}{0}.fits'.format(obsband)))
    rot = get_cal_rotator(flat_file)[0]
    flat_file.close()

    logger.info('- Processing frames at {0} deg rotation'.format(rot))
    frames = get_sof_list(filetype='arc', subdir=subdir, obsband=obsband, rotangle=rot, **kwargs)

    #write temporary .sof file to feed to
    outfile = '{0}.sof'.format(id_gen())
    logger.debug('-- Writing illumination frames to temporary file {0}'.format(outfile))

    with open(outfile, 'wt') as ofile:
        #only reconstruct a single arc frame
        for frame, ftype in frames[:1]:
            if ftype == 'WAVE,LAMP':
                ofile.write('{0} ARC_ON\n'.format(frame))
                logger.debug('--- Added {0} ARC_ON'.format(frame))

        #include additional files required for SPARK
        ofile.write('{0} XCAL\n'.format(os.path.join(caldir, 'XCAL_{0}{0}{0}.fits'.format(obsband.upper()))))
        logger.debug('--- Added {0} XCAL'.format(os.path.join(caldir, 'XCAL_{0}{0}{0}.fits'.format(obsband.upper()))))
        ofile.write('{0} YCAL\n'.format(os.path.join(caldir, 'YCAL_{0}{0}{0}.fits'.format(obsband.upper()))))
        logger.debug('--- Added {0} YCAL'.format(os.path.join(caldir, 'YCAL_{0}{0}{0}.fits'.format(obsband.upper()))))
        ofile.write('{0} LCAL\n'.format(os.path.join(caldir, 'LCAL_{0}{0}{0}.fits'.format(obsband.upper()))))
        logger.debug('--- Added {0} LCAL'.format(os.path.join(caldir, 'LCAL_{0}{0}{0}.fits'.format(obsband.upper()))))
        ofile.write('{0} WAVE_BAND\n'.format(os.path.join(static_cal, 'kmos_wave_band.fits')))
        logger.debug('--- Added {0} WAVE_BAND'.format(os.path.join(static_cal, 'kmos_wave_band.fits')))

    

    #execute spark script
    logger.debug('-- Executing kmos_reconstruct')
    ecode = run_spark('kmos_reconstruct', sof=outfile, logger=logger)
    logger.debug('--- Complete')

    #check the exit status of the SPARK call
    if ecode != 0:
        return 1

    #move files to their final location
    products = ['CUBE_ARC.fits']
    clean_move('CUBE_ARC.fits', os.path.join(caldir, 'reconstructed_arc_{0}_{1}.fits'.format(obsband.lower(), subdir)))

    #move the .sof file for logging
    clean_move(outfile, os.path.join(kwargs['outputdir'], subdir, 'sof', '{0}_reconstruct_arc_{1}.sof'\
            .format(subdir, obsband.lower())))

    return 0



def make_spark_calibs(logger=None, subdir=None, make_dark=True, make_arc=True,
                      make_flat=True, make_illum=True, reconstruct_arc=True,
                      make_standard=True, obsband=None, **kwargs):
  
    """ Wrapper to run the requested calibration steps using SPARK.  By default
    runs all of the dark, flat, arc, and illumination steps, and copies the results
    to subdir/calib/* for compatibility with previous codes.

    """
    ecode = 0

    #a bit ugly on the error catching here
    if make_dark:
        logger.info('Processing darks')
        comp = 'kmos_dark'
        try:
            ecode = spark_dark(logger=logger, subdir=subdir, obsband=obsband, **kwargs)
            if ecode:
                return ecode, comp
        except:
            return 1, comp
        else:
            logger.info('- Finished darks')

    if make_flat:
        logger.info('Processing flats')
        comp = 'kmos_flat'
        try:
            ecode = spark_flat(logger=logger, subdir=subdir, obsband=obsband, **kwargs)
            if ecode:
                return ecode, comp
        except:
            return 1, comp
        else:
            logger.info('- Finished flats')
   
    if make_arc:
        logger.info('Processing arcs')
        comp = 'kmos_wave_cal'
        try:
            ecode = spark_arc(logger=logger, subdir=subdir, obsband=obsband, **kwargs)
            if ecode:
                return ecode, comp
        except:
            return 1, comp
        else:
            logger.info('- Finished arcs')
    
    if make_illum:
        logger.info('Processing illumination')
        comp = 'kmos_illumination'
        try:
            ecode = spark_illum(logger=logger, subdir=subdir, obsband=obsband, **kwargs)
            if ecode:
                return ecode, comp
        except:
            return 1, comp
        else:
            logger.info('- Finished illumination')

    if reconstruct_arc:
        logger.info('Reconstructing arc')
        comp = 'kmos_reconstruct'
        try:
            ecode = spark_reconstruct_arc(logger=logger, subdir=subdir, obsband=obsband, **kwargs)
            if ecode:
                return ecode, comp
        except:
            return 1, comp
        else:
            logger.info('- Finished reconstruction')

    if make_standard:
        logger.info('Processing standard stars')
        comp = 'kmos_std_star'
        try:
            ecode = spark_standard(logger=logger, subdir=subdir, obsband=obsband, **kwargs)
            if ecode:
                return ecode, comp
        except:
            return 1, comp
        else:
            logger.info('- Finished standard stars')


    return ecode, 'none'


