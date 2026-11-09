""" This code corrects an observed KMOS exposure for barycentric velocity.

The core code was adapted from Jane Rigby's code for computing barycentric
corrections, which was in turn adapted from
https://gist.github.com/StuartLittlefair/5aaf476c5d7b52d20aa9544cfaa936a1
"""
from __future__ import print_function, division
import os
import numpy as np
import astropy.units as u
import astropy.constants as c
from astropy.io import fits
from astropy.coordinates import (SkyCoord, EarthLocation, solar_system, 
        UnitSphericalRepresentation, CartesianRepresentation)
from astropy.time import Time


from ..utils.fileutils import (get_obs_exposures, duplicate)
from ..utils.obsutils import reconstruct_wave

__all__ = ["rvcorr_cube", "get_rvcorr", "calc_vbary", "velocity_correct"]


#def apply_barycentric_correction(df, barycor_vel,  colwav='wave', colwavnew='newwave'):
#    ''' Applies the barycentric velocity correction to the wavelength of a spectrum.  Assumes Pandas.
#    Inputs:
#    df:           the spectrum, as a pandas DataFrame
#    barycor_vel:  barycentric correction velocity, from compute_barycentric_correction()
#    colwav:       column name that holds the wavelength
#    colwavenew:   column name for new wavelength.  Have not tested what happens if you try to overwrite colwav'''
#    A_c = constants.c  # speed of light
#    df[colwavnew] = df[colwav] * (1.0 + (barycor_vel / A_c))
#    return(0)

#fhead = fits.getheader('/Users/jtmendel/astro/kmos_redux/new_dc/KMOS_SPEC_OBS304_0024.fits')
#print( get_velcorr(fhead))

def calc_vbary(time, skycoord, location):
    """Barycentric velocity correction.  was named 'velcorr'. Should return
    barycentric correction factor, in km/s, same sign convention as in jskycalc 
    
    Uses the ephemeris set with
    ``astropy.coordinates.solar_system_ephemeris.set`` for corrections.  For
    more information see `~astropy.coordinates.solar_system_ephemeris`.
    
    Parameters
    ----------
    mytime : `~astropy.time.Time`
      The time of observation.
    skycoord: `~astropy.coordinates.SkyCoord`
      The sky location to calculate the correction for.
    location: `~astropy.coordinates.EarthLocation`, optional
      The location of the observatory to calculate the correction for.
      If no location is given, the ``location`` attribute of the Time
      object is used
      
    Returns
    -------
    vel_corr : `~astropy.units.Quantity`
      The velocity correction to convert to Barycentric velocities. Should be
      added to the original velocity.
    """
    # ICRS position and velocity of Earth's geocenter     
    ep, ev = solar_system.get_body_barycentric_posvel('earth', time)
    
    # GCRS position and velocity of observatory
    op, ov = location.get_gcrs_posvel(time)
    
    # ICRS and GCRS are axes-aligned. Can add the velocities
    velocity = ev + ov # relies on PR5434 being merged
    
    # get unit ICRS vector in direction of SkyCoord
    sc_cartesian = skycoord.represent_as(UnitSphericalRepresentation).represent_as(CartesianRepresentation)
   
    return sc_cartesian.dot(velocity).to(u.km/u.s) # similarly requires PR5434


def get_rvcorr(header):
    """Computes barycentric velocity correction.

    Parameters
    ----------
    header : dict
        Fits header object (from, for example `~astropy.io.fits`) from
        KMOS observations

    Returns
    -------
    vbary: `~astropy.units.Quantity`
        Barycentric velocity correction for the input frame
    """
    date_obs = header['DATE-OBS']
    ra = header['RA']
    dec = header['DEC']
    
    lat = header['ESO TEL GEOLAT']
    lon = header['ESO TEL GEOLON']
    elev = header['ESO TEL GEOELEV']

    location = EarthLocation(lon=lon, lat=lat, height=elev)
    time = Time(date_obs, format='isot', scale='utc')
    skycoord = SkyCoord(ra, dec, frame='icrs', unit=(u.deg, u.deg), obstime=time)

    vbary = calc_vbary(time, skycoord, location)
    return vbary


def rvcorr_cube(wave_in, array_in, vbary):
    """ Apply barycentric velocity correction to input KMOS data cube

    Parameters
    ----------
    wave_in : ndarray
        Input wavelength array.
    array_in : ndarray
        Input datacube, oriented as (wavelength, x, y).
    vbary: `~astropy.units.Quantity`
        Barycentric velocity, computed from get_velcorr.  Should be
        added to the original velocity. 

    Returns
    -------
    array_out: ndarray
        Datacube corrected for barycentric velocity.
    """
    wave_rvcorr = wave_in * (1.0 + (vbary / c.c))

    #make sure we're working with doubles for the interpolation
    idt = array_in.dtype

    array_in = np.asarray(array_in, dtype=np.float64)
    wave_in = np.asarray(wave_in, dtype=np.float64)

    #interpolate data from shifted wavelength grid back on to the input grid
    _, nx, ny = array_in.shape
    array_out = np.full_like(array_in, np.nan)
    goodpix_in = np.isfinite(array_in)
    for ix in range(nx):
        for iy in range(ny):
            old_spec = array_in[:, ix, iy]
            goodpix = goodpix_in[:, ix, iy]
            if np.count_nonzero(goodpix):
                new_spec = np.interp(wave_in, wave_rvcorr, old_spec,
                                     left=np.nan, right=np.nan)
                new_spec[~goodpix | ~np.isfinite(new_spec)] = np.nan
                array_out[:,ix,iy] = new_spec
    
    return np.asarray(array_out, dtype=idt)



def velocity_correct(logger=None, subdir=None, obsband=None, rvcorr=True,
		     search_suffix=None, rv_mode='interp', **kwargs):
    """rv_mode 'interp' (default): resample every spaxel onto the barycentric wavelength grid (linear interpolation;
    smooths by sqrt(f(1-f)) pixels, up to about 20-27 km/s, and correlates neighbouring planes).
    'header' (docs/rv-correction.md): leave the data in the telescope frame and only store VHELIO; the frames are combined
    in the telescope frame (the barycentric velocities of the frames of one combination differ by 0.2-0.4 km/s rms,
    at most 4-12 km/s) and the combined cube's wavelength axis is relabelled lambda (1 + <v>/c) afterwards, with the
    telluric correction applied unshifted before that."""


    # check that velocity correction is requested
    if not rvcorr:
        return 0
    mode = str(rv_mode).split(';')[0].split('#')[0].strip().lower()
    if mode not in ('interp', 'header'):
        raise ValueError("rv_mode must be 'interp' or 'header', got %r" % rv_mode)

    if logger is not None:
        logger.debug('- Running heliocentric corrections')

    # get list of frames to process for barycentric correction.  Assumes 
    # working with sky-subtracted frames in science directory
    exposures = get_obs_exposures(subdir=os.path.join(subdir, 'science'), 
                                  obsband=obsband, suffix=search_suffix, **kwargs)

    if not len(exposures):
        if logger is not None:
            logger.warning('-- No exposures found for heliocentric correction')
        return 1

    # iterate over each exposure, apply rv correction
    for exposure, _, _, _ in exposures:
        fpath = os.path.split(exposure)[0]

        #copy object to output
        exp_corr = duplicate(exposure, outputdir=fpath, suffix='rv')

        with fits.open(exp_corr, mode='update') as ofile:
            vbary = get_rvcorr(ofile[0].header)
            if mode == 'header':
                ofile[0].header['HISTORY'] = "Heliocentric correction {0:.2f} km/s stored, data not resampled".format(
                    float(vbary.value))
                ofile[0].header['HIERARCH ESO PYSPARK VHELIO'] = vbary.value
                ofile[0].header['HIERARCH ESO PYSPARK RV MODE'] = ('header', 'topocentric; relabel after combine')
                continue
            for ifu in range(1,25):
                cube = np.array(ofile[ifu].data)
                try:
                    oz, ox, oy = cube.shape
                except ValueError:
                    continue
                else:
                    wave = reconstruct_wave(ofile[ifu].header)
                    corr_cube = rvcorr_cube(wave, cube, vbary)
                    ofile[ifu].data = np.asarray(corr_cube, dtype=np.float32)

            #fluxh
            ofile[0].header['HISTORY'] = "Applied heliocentric correction {0:.2f} km/s".format(float(vbary.value))
            ofile[0].header['HIERARCH ESO PYSPARK VHELIO'] = vbary.value 
    
    return 0
