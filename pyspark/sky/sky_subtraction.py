from __future__ import print_function

import subprocess,copy
import contextlib
import numpy as np
import os, sys
import time
import bottleneck as bn
import astropy.io.fits as fits

import astropy.units as u
from scipy.optimize import minimize
#from astropy.modeling.blackbody import blackbody_lambda
from astropy.modeling import models

#from .zapv2 import (process, SVDoutput)
from .zapv2 import process as process_v2
from .zapv2 import Zap as ZapObj_v2
from .zapv2 import SVDoutput as SVDoutput_v2
from .zapv1 import process as process_v1
from .zapv1 import SVDoutput as SVDoutput_v1

from ..utils.obsutils import (find_lines, hsm, reconstruct_wave)
from ..utils.fileutils import (fetch_resource_path, duplicate, get_obs_exposures)
from .sky_combine import (parse_multisky_options, match_obj_skies,
                          combine_sky_cubes, scale_sky, provenance_cards)
from .hybrid import hybrid_ifu


__all__ = ["subtract_sky", "match_obj_sky", "match_obj_skies", "make_SVDs_v2", "build_custom_svd"]

sky_methods = ['zapv1', 'zapv2']



def match_obj_sky(exposures):
    """ Given a list of exposures generate by get_obs_exposures, return
    a list of object,sky tuples, where the matching is done by closest time.
    """
    #pull out times
    time_dict = dict([(efile, etime) for efile, _, etime, _ in exposures])

    #lookup table of times
    object_exposures = [(efile, dit) for efile, etype, _, dit in exposures if etype == 'OBJ'] 
    sky_exposures = [(efile, dit) for efile, etype, _, dit in exposures if etype == 'SKY'] 
    #sky_times = [time_dict[efile] for efile, _ in sky_exposures]
    #sky_dits = np.array([dit for _, dit in sky_exposures])

    obj_sky_pair = []
    for obj, odit in object_exposures:
        #only consider skys matched by exposure time
        matched_skys = [efile for efile, dit in sky_exposures if dit == odit]
        matched_times = [time_dict[efile] for efile, dit in sky_exposures if dit == odit]

        sky_idx = np.argmin(np.fabs(np.array(matched_times) - time_dict[obj]))
        obj_sky_pair.append((obj, matched_skys[sky_idx], odit))

    return obj_sky_pair


def make_SVDs_v2(spec_array, header, pcaoutfits=None, zlevel='none', cftype='med_smooth', 
              clean=False, cfwidth=101, boost=50, reject_factor=0.4, logger=None):
    """ Given an input (2D) array of spectra, generate PCA output following
    zap_v2.0 conventions.
    """
    #generate mask array for sky line rejection
    med_sky = bn.nanmedian(spec_array, axis=1)
    msky = find_lines(med_sky)

    #run zap instance
    zap_inst = SVDoutput_v2(None, cube='dummy', stack=spec_array, hdr=header, instrument='KMOS', 
                         zlevel=zlevel, cftype=cftype, clean=False, cfwidth=cfwidth, boost=boost, 
                         reject_factor=reject_factor, contmask=msky)

    if pcaoutfits is not None:
        zap_inst.writePCA(pcaoutputfits=pcaoutfits, overwrite=True)


def build_custom_svd(subdir=None, inputdir=None, obsband=None, pcaoutfits=None,
                     logger=None, **kwargs):
    """ Build a ZAPv2 eigenbasis (SVD) from this dataset's own SKY exposures,
    instead of using the generic bundled ZAPv2_SVD_*.fits file. Pools every
    spaxel's spectrum from every SKY-type exposure of `subdir`/`obsband`
    across all 24 IFUs, then hands that stack to make_SVDs_v2.

    Run this once per band before reduce_kmos.py, then point zap_sky_v2 at
    the result via svd_path=<pcaoutfits> (e.g. by setting svd_path in the
    [cubes] section of the .cfg -- subtract_sky forwards **kwargs straight
    through to zap_sky_v2).
    """
    exposures = get_obs_exposures(subdir=os.path.join(subdir, 'sky'), obsband=obsband, inputdir=inputdir, **kwargs)
    sky_exposures = [efile for efile, etype, _, _ in exposures if etype == 'SKY']
    if not len(sky_exposures):
        if logger is not None:
            logger.warning('-- No SKY exposures found for {0}, cannot build custom SVD'.format(subdir))
        return None

    spectra, header = [], None
    for sky_exp in sky_exposures:
        with fits.open(sky_exp) as sfile:
            for ifu in range(1, 25):
                cube = np.array(sfile[ifu].data)
                if cube.ndim != 3:
                    continue
                if header is None:
                    header = sfile[ifu].header
                flat = cube.reshape(cube.shape[0], -1)
                # spaxels routinely have a few NaN samples near the edge of
                # their own valid-wavelength coverage (see CUBE_EDGE_MARGIN
                # in telluric_check.ipynb) -- requiring every pixel finite
                # throws away nearly everything. Keep spaxels that are
                # mostly finite and zero-fill the small remaining gaps.
                finite_frac = np.isfinite(flat).mean(axis=0)
                usable = flat[:, finite_frac >= 0.9]
                if usable.size:
                    spectra.append(np.where(np.isfinite(usable), usable, 0.0))

    if not spectra:
        if logger is not None:
            logger.warning('-- No finite sky spectra found for {0}, cannot build custom SVD'.format(subdir))
        return None

    spec_array = np.concatenate(spectra, axis=1)
    if logger is not None:
        logger.info('- Building custom SVD from {0} sky spaxel spectra across {1} exposure(s)'.format(
            spec_array.shape[1], len(sky_exposures)))

    make_SVDs_v2(spec_array, header, pcaoutfits=pcaoutfits, logger=logger)
    return pcaoutfits


def sub_thermal(wave, cube, mask):
    #generate 1D spectrum from cube
    spec_1d = bn.nanmedian(cube[:,1:13,1:13].reshape(2048,-1), axis=1)

    #remove pixels with dodgy values in advance
    clean_mask = np.isfinite(spec_1d) & ~np.isnan(spec_1d) & (wave > 2.1) & (wave < 2.4)
    spec_1d = spec_1d[clean_mask]
    wave_init = wave[clean_mask]
    mask_init = ~mask[clean_mask]
   
    #iteratively fit for thermal component
    out_init = np.array([315., 3.30, 10.])
    for fiter in range(3):

        #initial fit using minizming the modal residual
        sol = minimize(res_bb, out_init, args=(wave_init, spec_1d, mask_init), method='L-BFGS-B', 
                            bounds=[(270,380), (0,10),(-200,300)])

        #store the output solution
        out_init = sol.x

        #generate best fit and modify the mask
        temp_bb = bb_model(out_init, wave_init)
        res = (spec_1d - temp_bb)[mask_init]
        med_offset = bn.nanmedian(res)
        var_offset = 1.4862 * bn.nanmedian(np.fabs(res-med_offset))
        
        #update mask
        mask_init = ((spec_1d - temp_bb) < 2.*var_offset + med_offset)

    ##remove subtract fitted thermal component and return
    fit_bb = bb_model([out_init[0], out_init[1], 0.], wave)
    corrected = np.copy(cube) - fit_bb[:,np.newaxis,np.newaxis]

    return corrected


def bb_model(params, wave):
    bb_spec = bb_lambda(wave, params[0]).value
    bb_spec /= bb_spec.max()
    return (10**params[1])*bb_spec + params[2]


def res_bb(p, x_in, y_in, mask_in):
    #generate blackbody
    model_spec = bb_model(p, x_in)

    #return rms residual
    residual = (y_in - model_spec)[mask_in]
    return np.sqrt(bn.nanmean(residual**2))


def bb_lambda(wave, temp):
    #generate blackbody spectrum given temperature.  In case we ever need it,
    #put it in physical units (similar to the calibrated spectrum)
    #temp_bb = (blackbody_lambda(wave*u.micron, temp) * u.sr).to(u.watt / u.meter**2 / u.micron) / 1e-17
    bb = models.BlackBody(temperature=temp*u.K)
    temp_bb = (bb(wave*u.micron) * u.sr).to(u.watt / u.meter**2 / u.micron, equivalencies=u.spectral_density(wave*u.micron)) / 1e-17
    return temp_bb

def salva(cube, filename):
    hdu0=fits.PrimaryHDU(data=cube)
    hdul=fits.HDUList([hdu0])
    hdul.writeto(filename,overwrite=True)
    


def zap_sky_v2(matched_exposures, logger=None, obsband=None, subdir=None,
            inputdir=None, suffix='s', tbsub=False, svd_path=None,
            sky_combine='nearest', sky_scale='none', sky_nan='renorm', hybrid_sky=False,
            hybrid_min_dangle=20., hybrid_wing=0, hybrid_oh_model='median', hybrid_free_model='tmpl', hybrid_edge=0,
            sky_align=False, **kwargs):

    for obj_exp, sky_exp, dit in matched_exposures:
        # sky_exp is either one SKY file (classic, from match_obj_sky) or a
        # list of (sky_file, weight, dt) from match_obj_skies (multi-sky)
        multi = not isinstance(sky_exp, str)
        if multi:
            sky_files = [s[0] for s in sky_exp]
            sky_wts = [s[1] for s in sky_exp]
        else:
            sky_files = [sky_exp]

        # hybrid sky subtraction (docs/hybrid-sky.md): reconstruct() wrote the DRP detector-level
        # sky-subtracted cube of this OBJECT frame to skysub_det/ if the OB has a rotator mismatch
        hyb_path = os.path.join(inputdir, subdir, 'skysub_det', os.path.basename(obj_exp))
        use_hyb = bool(hybrid_sky) and not multi and os.path.isfile(hyb_path)
        if use_hyb:
            # the detector-level cube must belong to this O - S: same SKY frame, made after the sky/ OBJECT
            # cube (a later reconstruction without hybrid_sky leaves old cubes behind), rotator offset above
            # the threshold
            dh = fits.getheader(hyb_path)
            det_sky = dh.get('HIERARCH ESO PYSPARK SKYSUB DET SKY')
            why = None
            if det_sky is None or 'SCI_RECONSTRUCTED_{0}'.format(det_sky) != os.path.basename(sky_exp):
                why = 'was made with SKY {0}, not {1}'.format(det_sky, os.path.basename(sky_exp))
            elif os.path.getmtime(hyb_path) < os.path.getmtime(obj_exp):
                why = 'is older than sky/{0}'.format(os.path.basename(obj_exp))
            elif float(dh.get('HIERARCH ESO PYSPARK SKYSUB DET DANGLE', 0.)) <= float(hybrid_min_dangle):
                why = 'has a rotator offset <= hybrid_min_dangle'
            if why is not None:
                if logger is not None:
                    logger.warning('-- skysub_det/{0} {1}: classic O - S'.format(os.path.basename(hyb_path), why))
                use_hyb = False

        # slitlet-end alignment (pyspark/cubes/sky_align.py): reconstruct() wrote the SKY cube aligned to this
        # OBJECT frame to sky_align/; it replaces the mean-angle SKY cube of sky/ in the classic O - S
        if sky_align and not multi:
            al_path = os.path.join(inputdir, subdir, 'sky_align', os.path.basename(obj_exp))
            why = None
            if not os.path.isfile(al_path):
                why = 'missing'
            else:
                al_sky = fits.getheader(al_path).get('HIERARCH ESO PYSPARK SKY ALIGN SKY')
                if al_sky is None or 'SCI_RECONSTRUCTED_{0}'.format(al_sky) != os.path.basename(sky_exp):
                    why = 'was made from SKY {0}, not {1}'.format(al_sky, os.path.basename(sky_exp))
                elif os.path.getmtime(al_path) < os.path.getmtime(obj_exp):
                    why = 'is older than sky/{0}'.format(os.path.basename(obj_exp))
            if why is None:
                sky_files = [al_path]
                if logger is not None:
                    logger.info('-- Aligned SKY cube sky_align/{0}'.format(os.path.basename(al_path)))
            elif logger is not None:
                logger.warning('-- sky_align/{0} {1}: mean-angle SKY cube {2}'.format(
                    os.path.basename(al_path), why, os.path.basename(sky_exp)))

        input_path = os.path.split(obj_exp)[0]
        output_path = os.path.join(inputdir, subdir, 'science')

        if svd_path is not None:
            # use a dataset-specific eigenbasis (see build_custom_svd below)
            # instead of the generic bundled one -- its OH-line residual
            # structure matches this run's own sky frames, not whatever
            # dataset ZAPv2_SVD_*_n500004.fits was originally built from.
            if logger is not None:
                logger.info('- Using custom SVD file {0}'.format(svd_path))
            fpath = svd_path
        else:
            #fetch the superdark reference frame for this detector
            if obsband.upper() == 'YJ':
                if int(dit) == 600:
                    logger.warning("- Using 600s SVD file")
                    svd_file = 'ZAPv2_SVD_{0}_n9504_e600.fits'.format(obsband.upper())
                else:
                    svd_file = 'ZAPv2_SVD_{0}_n500004.fits'.format(obsband.upper())
                #if int(dit) == 300:
                #    svd_file = 'ZAPv2_SVD_{0}_n500004.fits'.format(obsband.upper())
                #elif int(dit) == 600:
                #    logger.warning("- Using 600s SVD file")
                #    svd_file = 'ZAPv2_SVD_{0}_n9504_e600.fits'.format(obsband.upper())
            else:
                svd_file = 'ZAPv2_SVD_{0}_n500004.fits'.format(obsband.upper())

            fpath = fetch_resource_path('data/{0}'.format(svd_file))
        # copy object frame to output
        exp_sub = duplicate(obj_exp, outputdir=output_path, suffix=suffix) 

        if logger is not None:
            logger.info('- Subtracting sky for {0}'.format(obj_exp.split('/')[-1])) 
            if multi:
                logger.info('-- Sky ({0}): {1}'.format(sky_combine, ', '.join(
                    '{0} w={1:.3f} dt={2:+.0f}s'.format(s.split('/')[-1], w, d) for s, w, d in sky_exp)))
            else:
                logger.debug('-- Using {0} for initial subtraction'.format(sky_exp.split('/')[-1]))
            if use_hyb:
                logger.info('-- Hybrid sky subtraction with skysub_det/{0} (wing {1}, oh {2}, free {3}{4})'.format(
                    os.path.basename(hyb_path), int(hybrid_wing), str(hybrid_oh_model).strip(),
                    str(hybrid_free_model).strip(), ', edge {0}'.format(int(hybrid_edge)) if int(hybrid_edge) else ''))
        with fits.open(exp_sub, mode='update') as ofile, contextlib.ExitStack() as sstack:
            sfiles = [sstack.enter_context(fits.open(sf)) for sf in sky_files]
            rfile = sstack.enter_context(fits.open(hyb_path)) if use_hyb else None

            for ifu in range(1,25):
                if not multi:
                    scube = np.array(sfiles[0][ifu].data)
                else:
                    # SKY frames without data for this IFU are left out
                    sdata = [(np.array(s[ifu].data), w) for s, w in zip(sfiles, sky_wts)
                             if s[ifu].data is not None]
                    scube = combine_sky_cubes([c for c, _ in sdata], [w for _, w in sdata],
                                              sky_nan=sky_nan) if sdata else None
                ocube = np.array(ofile[ifu].data)
                header = ofile[ifu].header

                try:
                    oz, ox, oy = ocube.shape
                except ValueError:
                    continue
                else:
                    if logger is not None:
                        logger.info('-- Processing sky subtraction for IFU {0}'.format(ifu))
                    if multi and scube is None:
                        raise ValueError('IFU {0} of {1}: no SKY frame has data'.format(
                            ifu, obj_exp.split('/')[-1]))

                    #generate mask array for continuum
                    msky = find_lines(bn.nanmedian(scube[:,1:13,1:13].reshape(2048,-1), axis=1))
                    
                    #rescale the (combined) sky to this object frame (multi-sky only)
                    if multi:
                        scube, scale_cards = scale_sky(ocube, scube, header, sky_scale=sky_scale,
                                                       logger=logger)
                        for key, val, com in scale_cards:
                            header[key] = (val, com)

                    #apply thermal background correction if flagged
                    if tbsub:
                        if logger is not None:
                            logger.info('--- Removing thermal background from object and sky frames')
                        #reconstruct wavelength solution
                        #print('ricostruisci')
                        wave = reconstruct_wave(header)

                        #extract 1D spectra from the un-sky-subtracted cubes
                        scube_corr = sub_thermal(wave, scube, msky)
                        ocube_corr = sub_thermal(wave, ocube, msky)
                        sub_cube = np.copy(ocube_corr) - np.copy(scube_corr)
                    elif use_hyb and rfile[ifu].data is not None and rfile[ifu].data.shape == ocube.shape:
                        lam = (np.arange(oz) + 1 - header['CRPIX3']) * header['CDELT3'] + header['CRVAL3']
                        sub_cube = hybrid_ifu(ocube, scube, np.array(rfile[ifu].data), ifu, lam,
                                              wing=int(hybrid_wing), oh_model=str(hybrid_oh_model).strip(),
                                              free_model=str(hybrid_free_model).strip(), edge=int(hybrid_edge))
                        sub_cube = sub_cube.astype(np.float32).astype(float)
                    else:
                        if use_hyb and logger is not None:
                            logger.warning('--- IFU {0}: no detector-level cube, classic O - S'.format(ifu))
                        sub_cube = np.copy(ocube) - np.copy(scube)

                    nanmask = np.isfinite(sub_cube)
                    nanmask2=copy.deepcopy(nanmask)
                    sub_cube[~nanmask] = 0.

                    zap_obj = process_v2('dummy', cube=sub_cube, hdr=header, clean=True,
                                        zlevel='none', cftype='med_smooth', cfwidthSP=101,
                                        extSVD=fpath, interactive=True, overwrite=True,
                                        instrument='KMOS', boost=50, contmask=msky,
                                        reject_factor=0.4, logger=logger)

                    #reprocess the reconstructed sky spectrum to correct for any residual continuum drift
                    if logger is not None:
                        logger.debug('---- running second-order continuum cleaning')

                    cleaner_obj = ZapObj_v2('dummy', pca_class=None, n_components=None, instrument='KMOS',
                                            cube=zap_obj.recon.reshape(sub_cube.shape), hdr=header, logger=None)
                    cleaner_obj._prepare(clean=True, zlevel='none', cftype='med_smooth',
                                         cfwidth=101, extzlevel=None, mask=None, boost=50,
                                         contmask=msky, reject_factor=0.4)

                    #reincorporate residual continuum array
                    corr_cube = np.copy(zap_obj.cleancube+cleaner_obj.contarray.reshape(sub_cube.shape))
                    corr_cube[~nanmask2] = np.nan
                    ofile[ifu].data = np.asarray(corr_cube, dtype=np.float32)

            #record which SKY frames / weights were subtracted (multi-sky only)
            if multi:
                for key, val, com in provenance_cards(sky_exp, sky_combine, sky_scale, sky_nan):
                    ofile[0].header[key] = (val, com)
                ofile[0].header['HISTORY'] = 'Sky: {0} of {1} SKY frame(s), scale={2}, nan={3}'.format(
                    sky_combine, len(sky_exp), sky_scale, sky_nan)
            if sky_files[0] != sky_exp and not multi:
                ofile[0].header['HIERARCH ESO PYSPARK SKY ALIGN'] = (True, 'SKY cube aligned at slitlet ends')
                ofile[0].header['HISTORY'] = 'Sky: sky_align/{0}'.format(os.path.basename(sky_files[0]))
            if use_hyb:
                ofile[0].header['HIERARCH ESO PYSPARK SKY HYBRID'] = (True, 'detector-level O-S + sky-only model')
                ofile[0].header['HIERARCH ESO PYSPARK SKY HYBRID WING'] = (int(hybrid_wing), 'IFU-median in OH wings 2..W')
                if str(hybrid_oh_model).strip() != 'median':
                    ofile[0].header['HIERARCH ESO PYSPARK SKY HYBRID OHMODEL'] = (str(hybrid_oh_model).strip(),
                                                                                  'OH-core model of D')
                if str(hybrid_free_model).strip() != 'tmpl':
                    ofile[0].header['HIERARCH ESO PYSPARK SKY HYBRID FREEMODEL'] = (str(hybrid_free_model).strip(),
                                                                                    'off-OH model of D')
                if int(hybrid_edge):
                    ofile[0].header['HIERARCH ESO PYSPARK SKY HYBRID EDGE'] = (int(hybrid_edge),
                                                                               'slitlet-end level of D, N positions')
                ofile[0].header['HISTORY'] = 'Sky: hybrid (DRP detector-level O-S + slitlet model of the sky-only term)'


def zap_sky_v1(matched_exposures, logger=None, obsband=None, subdir=None, 
            inputdir=None, suffix='s', tbsub=None, **kwargs):

    #fetch the superdark reference frame for this detector
    svd_file = 'ZAPv1_SVD_{0}_n500004.fits'.format(obsband.upper())
    fpath = fetch_resource_path('data/{0}'.format(svd_file))
    
    #if logger is not None:
    #    logger.debug("- Subtracting sky using {0}".format(svd_file))

    for obj_exp, sky_exp in matched_exposures:
        input_path = os.path.split(obj_exp)[0]
        output_path = os.path.join(inputdir, subdir, 'science')

        # copy object frame to output
        exp_sub = duplicate(obj_exp, outputdir=output_path, suffix=suffix) 

        with fits.open(exp_sub, mode='update') as ofile, fits.open(sky_exp) as sfile:

            for ifu in range(1,25):
                scube = np.array(sfile[ifu].data)
                ocube = np.array(ofile[ifu].data)
                header = ofile[ifu].header

                try:
                    oz, ox, oy = ocube.shape
                except ValueError:
                    continue
                else:
                    if logger is not None:
                        logger.info('-- Processing sky subtraction for IFU {0}'.format(ifu))

                    #generate mask array for continuum
                    msky = find_lines(bn.nanmedian(scube[:,1:13,1:13].reshape(2048,-1), axis=1))
                    
                    #apply thermal background correction if flagged
                    if tbsub:
                        if logger is not None:
                            logger.info('--- Removing thermal background from object and sky frames')
                        #reconstruct wavelength solution
                        wave = reconstruct_wave(header)

                        #extract 1D spectra from the un-sky-subtracted cubes
                        scube_corr = sub_thermal(wave, scube, msky)
                        ocube_corr = sub_thermal(wave, ocube, msky)
                        sub_cube = np.copy(ocube_corr) - np.copy(scube_corr)
                    else:
                        sub_cube = np.copy(ocube) - np.copy(scube)
                    
                    nanmask = np.isfinite(sub_cube)
                    sub_cube[~nanmask] = 0.

                    zap_obj = process_v1('dummy', cube=sub_cube*exp_scale, hdr=header, clean=False, 
                                        zlevel='none', cftype='med_smooth', cfwidthSP=51, 
                                        extSVD=fpath, interactive=True, overwrite=True,
                                        instrument='KMOS', boost=50, contmask=msky,
                                        reject_factor=0.4, logger=logger)
                
                    corr_cube = np.copy(zap_obj.cleancube)
                    corr_cube[~nanmask] = np.nan

                    ofile[ifu].data = np.asarray(corr_cube, dtype=np.float32)


def subtract_sky(logger=None, subdir=None, obsband=None, 
                 sky_method='zapv2', search_suffix=None, 
                 tbsub=True, segment_sky=False, sky_combine='nearest',
                 sky_frames=None, sky_max_dt=None, sky_scale='none',
                 sky_nan='renorm', **kwargs):
    """ Driver function for sky subtraction.  Currently only supports
    subtraction using ZAP, but can be expanded to include other approaches.

    sky_combine/sky_frames/sky_max_dt/sky_scale/sky_nan ([cubes] options, see
    sky_combine.py) select a multi-sky subtraction; with the defaults
    (nearest, none) the classic single-nearest-sky code path runs unchanged.
    """
    #check that sky subtraction is handled by this routine
    if sky_method not in sky_methods:
        return 0

    sky_combine, sky_frames, sky_max_dt, sky_scale, sky_nan, classic = parse_multisky_options(
        sky_combine=sky_combine, sky_frames=sky_frames, sky_max_dt=sky_max_dt,
        sky_scale=sky_scale, sky_nan=sky_nan)
    if not classic and sky_method != 'zapv2':
        raise ValueError('multi-sky options are only implemented for sky_method = zapv2')

    #get list of frames to process. Assumes location of un-sky-subtracted frames in "sky" directory
    exposures = get_obs_exposures(subdir=os.path.join(subdir, 'sky'), obsband=obsband, 
                                  suffix=search_suffix, **kwargs) 
    if not len(exposures):
        if logger is not None:
            logger.warning('-- No reconstructed frames found, skipping sky subtraction')
        return 1

    #generate associated list of objects and skies
    if classic:
        obj_sky = match_obj_sky(exposures)
    else:
        obj_sky = match_obj_skies(exposures, sky_combine=sky_combine, sky_frames=sky_frames,
                                  sky_max_dt=sky_max_dt, logger=logger)
    #print(obj_sky)
    #only apply thermal background correction for k-band
    if tbsub and obsband == 'K':
        tbsub = True
    else:
        tbsub = False
    if tbsub and sky_scale != 'none':
        # sub_thermal would run after the rescaling; the DRP removes the thermal
        # component before scaling (kmos_priv_sky_tweak) - not implemented
        raise ValueError('sky_scale is not implemented for the K-band tbsub path')
    if kwargs.get('sky_align', False) and (not classic or sky_method != 'zapv2' or kwargs.get('hybrid_sky', False)):
        raise ValueError('sky_align works only with the classic nearest-sky zapv2 subtraction (no multi-sky, no hybrid)')
    if kwargs.get('hybrid_sky', False) and (not classic or tbsub or sky_method != 'zapv2'):
        raise ValueError('hybrid_sky works only with the classic nearest-sky zapv2 subtraction (no multi-sky, no K tbsub)')


    if sky_method == 'zapv2':
        zap_sky_v2(obj_sky, logger=logger, obsband=obsband, subdir=subdir, 
                   tbsub=tbsub, segment_sky=segment_sky, sky_combine=sky_combine,
                   sky_scale=sky_scale, sky_nan=sky_nan, **kwargs)    

    elif sky_method == 'zapv1':
        zap_sky_v1(obj_sky, logger=logger, obsband=obsband, subdir=subdir, tbsub=tbsub, **kwargs)    

    return 0
