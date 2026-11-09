from __future__ import print_function
import os
import numpy as np
import astropy.units as u
import glob

from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord

try:
    from astroquery.simbad import Simbad
    from astroquery.irsa import Irsa
except ImportError:
    pass

from ..utils.fileutils import (get_sof_list, clean_move)
from ..utils.obsutils import (get_cal_rotator, run_spark, match_obs_rotator, sex_to_deg,
                              reconstruct_wave)
from ..utils.setuputils import id_gen


__all__ = ["spark_standard", "get_star_info", "trace", "get_instrument_response"]

# static calibration directory based on SPARK_HOME environment variable
static_cal = os.environ['KMOS_CALIB']

# band-specific VO table names in case of online query
VO_flux = {'YJ':'flux(J)', 'H':'flux(H)', 'K':'flux(K)'}
VO_table_flux = {'YJ':'FLUX_J', 'H':'FLUX_H', 'K':'FLUX_K'}
ir_table_flux = {'YJ':'j_m', 'H':'h_m', 'K':'k_m'}

def spark_standard(logger=None, subdir=None, obsband=None, **kwargs):

    """ Reconstruct all standard stars for this band.
    """

    #path for the output calibration directory
    caldir = os.path.join(kwargs['outputdir'], subdir, 'calib')
    stddir = os.path.join(kwargs['outputdir'], subdir, 'std')

    #fetch rotator angles and filled extensions from the master flat file
    flat_file = fits.open(os.path.join(caldir, 'MASTER_FLAT_{0}{0}{0}.fits'.format(obsband)))
    pro_naangles = get_cal_rotator(flat_file)
    nrot = len(pro_naangles)
    flat_file.close()


    #get list of flat frames for processing
    frames = get_sof_list(filetype='standard', subdir=subdir, obsband=obsband, **kwargs)

    #group frames by start time (these are unique OBs)
    group_dict = {}
    name_dict = {}
    dit_dict = {}
    rot_dict = {}
    for obs_file, obs_dit, obs_start, obs_name, obs_rot in frames:
        group_dict.setdefault(obs_start, []).append(obs_file)
        name_dict.setdefault(obs_name, []).append(obs_start)
        dit_dict[obs_start] = obs_dit #dits are always the same within the same OB
    
        if obs_rot < 0: obs_rot += 360
        rot_dict.setdefault(obs_start, []).append(obs_rot)
    
    name_dict = dict([(star, list(np.unique(start_list))) for star, start_list in name_dict.items()])
    rot_dict = dict([(ttime, np.mean(rot_list)) for ttime, rot_list in rot_dict.items()])

    # pass over each star, being careful to handle multiple observations
    for star, times in name_dict.items():
        dits = [dit_dict[ot] for ot in times]

        cnt_exp = 1
        for ttime in times:
            # include additional index to deal with multiple observations of same star in
            # the same band with the same exposure time
            base_out = '{0}_{1:.1f}s_{2}_{3}'.format(star, float(dit_dict[times[0]]), 
                                                          obsband.lower(), cnt_exp) 
            cnt_exp += 1
      
            # get matched rotation for the illumination correctoin
            _, matched_rot, this_star_rot = match_obs_rotator('dummy', pro_naangles, 
                                                              input_rot=float(rot_dict[ttime])) 

            #write temporary .sof file to feed to kmos
            outfile = '{0}.sof'.format(id_gen())
            logger.debug('- Writing standard star frames to temporary file {0}'.format(outfile))

            with open(outfile, 'wt') as ofile:
                for frame in group_dict[ttime]:
                    ofile.write('{0} STD\n'.format(frame))
                    logger.debug('-- Added {0} STD'.format(frame))

                ofile.write('{0}/ILLUM_CORR_{1}{1}{1}_{2:03}.fits ILLUM_CORR\n'.format(caldir, obsband.upper(), matched_rot))
                logger.debug('-- Added {0}/ILLUM_CORR_{1}{1}{1}_{2:03}.fits ILLUM_CORR'.format(caldir, obsband.upper(), matched_rot))
                ofile.write('{0}/MASTER_FLAT_{1}{1}{1}.fits MASTER_FLAT\n'.format(caldir, obsband.upper()))
                logger.debug('-- Added {0}/MASTER_FLAT_{1}{1}{1}.fits MASTER_FLAT'.format(caldir, obsband.upper()))
                ofile.write('{0}/XCAL_{1}{1}{1}.fits XCAL\n'.format(caldir, obsband.upper()))
                logger.debug('-- Added {0}/XCAL_{1}{1}{1}.fits XCAL'.format(caldir, obsband.upper()))
                ofile.write('{0}/YCAL_{1}{1}{1}.fits YCAL\n'.format(caldir, obsband.upper()))
                logger.debug('-- Added {0}/YCAL_{1}{1}{1}.fits YCAL'.format(caldir, obsband.upper()))
                ofile.write('{0}/LCAL_{1}{1}{1}.fits LCAL\n'.format(caldir, obsband.upper()))
                logger.debug('-- Added {0}/LCAL_{1}{1}{1}.fits LCAL'.format(caldir, obsband.upper()))
                ofile.write('{0}/kmos_wave_band.fits WAVE_BAND\n'.format(static_cal))
                logger.debug('-- Added {0}/kmos_wave_band.fits WAVE_BAND'.format(static_cal))
                ofile.write('{0}/kmos_spec_type.fits SPEC_TYPE_LOOKUP\n'.format(static_cal))
                logger.debug('-- Added {0}/kmos_spec_type.fits SPEC_TYPE_LOOKUP'.format(static_cal))
                ofile.write('{0}/kmos_atmos_{1}.fits ATMOS_MODEL\n'.format(static_cal, obsband.lower()))
                logger.debug('-- Added {0}/kmos_atmos_{1}.fits ATMOS_MODEL'.format(static_cal, obsband.lower()))

       
            #control args
            routine_args = '--no_noise --save_cubes=TRUE'

            #execute spark script
            logger.debug('- Executing kmos_std_star')
            ecode = run_spark('kmos_std_star', args=routine_args, sof=outfile, logger=logger)
            logger.debug('-- Complete')
        
            #check the exit status of the SPARK call
            if ecode != 0:
                return 1

            #move files to their final location
            products = ['TELLURIC_{0}{0}{0}{1}.fits', 'STAR_SPEC_{0}{0}{0}{1}.fits', 'STD_MASK_{0}{0}{0}{1}.fits',
                        'STD_IMAGE_{0}{0}{0}{1}.fits', 'STD_CUBE_{0}{0}{0}{1}.fits']
            for product in products:
                clean_move(product.format(obsband.upper(),''), 
                           os.path.join(stddir, product.format('', base_out)))
        
            #move the .sof file for logging
            clean_move(outfile, os.path.join(kwargs['outputdir'], subdir, 'sof', '{0}_stdstar_{1}.sof'\
                    .format(subdir, base_out)))
    return 0


def get_star_info(header=None, query=True):
    """ This routine returns the star name, appropriate in-band
    magnitude, and spectral + luminosity class.  Its default
    behaviour is to execute a position-based Simbad query, but 
    in case of failure it instead just returns the header information
    as written.
    """
    # parse header information
    date_obs = header['DATE-OBS']
    band = header['ESO INS FILT1 ID'].upper()  
  
    alpha = header['ESO OCS TARG ALPHA']
    delta = header['ESO OCS TARG DELTA']
    ra, dec = sex_to_deg(alpha, delta)

    # header star information (in case of query failure)
    hdr_star_mag = str(header['ESO OCS STDSTAR MAG']) 
    # catch rare case of multiple mags in the header
    if len(hdr_star_mag.split(',')) > 1:
        hdr_star_mag = float(hdr_star_mag.split(',')[0])
    else:
        hdr_star_mag = float(hdr_star_mag)

    hdr_star_type = str(header['ESO OCS STDSTAR TYPE'])
    hdr_star_name = header['ESO OBS TARG NAME']

    # allow just returning header information as written
    if not query:
        return hdr_star_name, hdr_star_type, hdr_star_mag

    # build time and coordinate objects 
    time = Time(date_obs, format='isot', scale='utc')
    skycoord = SkyCoord(ra, dec, frame='icrs', unit=(u.deg, u.deg), obstime=time)

    # execute the primary query and pull out stellar type and magnitude information
    query = Simbad()
    ir_query = Irsa()

    query.add_votable_fields(VO_flux[band], 'sptype')
    prim_res = query.query_region(skycoord, radius=3.*u.arcsecond)

    # query 2MASS catalog directly for magnitudes
    tmass_res = ir_query.query_region(skycoord, catalog='fp_psc', radius=3.0*u.arcsecond)

    if prim_res is not None:

        simbad_mag = prim_res[VO_table_flux[band]][0]
        star_mag = tmass_res[ir_table_flux[band]][0]
        star_type = prim_res['SP_TYPE'][0].decode('utf-8')

        #check for multiple classifications.  In this case just adopt the header type (otherwise really messy).
        if len(star_type.split('/')) > 1:
            star_type = hdr_star_type

        #check spectral type, if it doesn't include a luminosity class then assume MS
        if len(star_type) < 3:
            star_type += 'V'

        # resolve the hipparcos name if one exists, otherwise use MAIN_ID
        name_res = query.query_objectids(prim_res['MAIN_ID'][0].decode('utf-8'))
        hip_name = [qid.strip() for qid in name_res['ID'] if qid[:3] == 'HIP']
        if not len(hip_name):
            star_name = '_'.join(prim_res['MAIN_ID'][0].decode('utf-8').strip().split(None))
        else:
            star_name = '_'.join(hip_name[0].split(None))
        return star_name, star_type, star_mag
    else:
        return hdr_star_name, hdr_star_type, hdr_star_mag

def factors(n):
    return np.array(list({f for i in range(1, int(n**0.5)+1) if n % i == 0 for f in [i, n//i]}))


def trace(cube, wave):
    """ Trace an object through a spectral data cube.
    """
    
    #chunk up the cube to make images for better S/N
    npix = cube.shape[0]
    splits = factors(npix)
    use_split = splits[np.argmin(np.fabs(np.sqrt(npix)-splits))]
    wave_splits = np.linspace(0,npix-1,use_split+1, dtype=int)
    wave_limits = [(wave[ii], wave[ii+1]) for ii in wave_splits[:-1]]

#    for low, high in wave_limits:
    return

    






def get_instrument_response(logger=None, subdir=None, obsband=None, **kwargs):
    """ produces response curves for each IFU based on the observed standard
    star and pre-computed relative arm response curves.  Output includes factor
    to convert from counts to physical units (erg/s/cm^2/A).
    """

    #identify standard star cubes in this band
    files = glob.glob('{0}/*cube*_{1}_*.fits'.format(os.path.join(kwargs['outputdir'], subdir, 'std'),obsband.lower()))
   
    #iterate over each cube, trace the object and extract its spectrum
    for file in files[:1]:
        with fits.open(file) as ofile:

            for ifu in range(1,25):
                cube = np.array(ofile[ifu].data)
                
                try:
                    oz, ox, oy = cube.shape
                except ValueError:
                    continue
                else:
                    wave = reconstruct_wave(ofile[ifu].header)

                    #trace the object in the cube
                    trace_params = trace(cube, wave)




    #   
    return

def extract_spectrum(logger=None, subdir=None, obsband=None, **kwargs):
    """
    """
    return
