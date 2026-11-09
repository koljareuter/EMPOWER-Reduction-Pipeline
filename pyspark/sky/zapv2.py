# ZAP - Zurich Atmosphere Purge
#
# Copyright (c) 2014-2016 Kurt Soto
# Copyright (c) 2015-2017 Simon Conseil
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the
# following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

####################################################################
# Nov. 2017 - This version of the code has been modified by Trevor Mendel 
# for use with KMOS data and inclusion in the pySPARK software package 
# including adaptations for working with many sky exposures.
####################################################################


from __future__ import absolute_import, division, print_function

import astropy.units as u
import logging
import numpy as np
import os
import scipy.ndimage as ndi
import sys

from astropy.io import fits
from astropy.utils import minversion
from astropy.wcs import WCS
from astropy.stats import (sigma_clip, mad_std)

from functools import wraps
from multiprocessing import cpu_count, Manager, Process, freeze_support
from scipy.stats import sigmaclip
from time import time

from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
from sklearn.decomposition._base import _BasePCA

from ._zap_version import __version2__ as __version__

__all__ = ['process', 'SVDoutput', 'nancleanfits', 'contsubfits', 'Zap',
           'wmedian']

# Limits of the segments in Angstroms. Use single region for the optical
##SKYSEG = [0, 10140, 10670, 11250, 11960, 12450, 12890, 14000, 14720, 
##          15650, 16000, 16356, 17253, 18400, 19570, 20950, 22000, 
##          23500, 24600, 26000]
##SKYSEG = [0, 26000]
##CONTSEG = [0, 10140, 10670, 11250, 11960, 12450, 12890, 14000, 14720, 
##           15650, 16000, 16356, 17253, 18400, 19570, 20950, 22000, 
##           23500, 24600, 26000]
##CONTFLAG = [0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0]


# segment, see below for the old values.
#SKYSEG_MUSE = [0, 10000]

# These are the old limits from the original zap. Keeping them for reference,
# and may be useful for further testing.
# [0, 5400, 5850, 6440, 6750, 7200, 7700, 8265, 8602, 8731, 9275, 10000]

# range where the NaD notch filters absorbs significant flux
NOTCH_FILTER_RANGES = {
    'WFM-AO-E': [5755, 6008],
    'WFM-AO-N': [5805, 5966]
}

# List of allowed values for cftype (continuum filter)
CFTYPE_OPTIONS = ('weight', 'median', 'fit', 'none', 'med_smooth')

# Number of available CPUs
NCPU = cpu_count()
NCPS = 1
PY2 = sys.version_info[0] == 2
ASTROPY_LT_1_3 = not minversion('astropy', '1.3')

if not PY2:
    text_type = str
    string_types = (str,)
else:
    text_type = unicode
    string_types = (str, unicode)


# ================= Top Level Functions =================

def process(cubefits, cube=None, hdr=None, outcubefits='DATACUBE_ZAP.fits', clean=True,
            zlevel='median', cftype='weight', cfwidthSVD=100, cfwidthSP=100,
            nevals=[], extSVD=None, skycubefits=None, mask=None,
            interactive=False, ncpu=None, pca_class=None, n_components=None,
            overwrite=False, varcurvefits=None, instrument='MUSE', boost=50,
            reject_factor=0.0, contmask=None, logger=None, segment_sky=False):
    """ Performs the entire ZAP sky subtraction algorithm.

    This is the main ZAP function. It works on an input FITS file and
    optionally writes the product to an output FITS file.

    Parameters
    ----------
    cubefits : str
        Input FITS file, containing a cube with data in the first extension.
    outcubefits : str
        Output FITS file, based on the input one to propagate all header
        information and other extensions. Default to `DATACUBE_ZAP.fits`.
    clean : bool
        If True (default value), the NaN values are cleaned. Spaxels with more
        then 25% of NaN values are removed, the others are replaced with an
        interpolation from the neighbors. The NaN values are reinserted into
        the final datacube. If set to False, any spaxel with a NaN value will
        be ignored.
    zlevel : str
        Method for the zeroth order sky removal: `none`, `sigclip` or `median`
        (default).
    cftype : {'weight', 'median', 'fit', 'none'}
        Method for the continuum filter. For the `weight` method, a zeroth
        order sky is required (see `zlevel`).
    cfwidthSVD : int or float
        Window size for the continuum filter, for the SVD computation.
        Default to 300.
    cfwidthSP : int or float
        Window size for the continuum filter used to remove the continuum
        features for calculating the eigenvalues per spectrum. Smaller values
        better trace the sources. An optimal range of is typically
        20 - 50 pixels. Default to 300.
    nevals : list
        Allow to specify the number of eigenspectra used for each segment.
        Provide either a single value that will be used for all of the
        segments, or a list of 11 values that will be used for each of the
        segments.
    extSVD : Zap object
        Can be a ``Zap`` object output from :func:`~zap.SVDoutput`.
        If given, the SVD from this object will be used, otherwise the SVD is
        computed. So this allows to compute the SVD on an other field or with
        different settings.
    skycubefits : str
        Path for the optional output of the sky that is subtracted from the
        cube. This is simply the input cube minus the output cube.
    mask : str
        A 2D fits image to exclude regions that may contaminate the zlevel or
        eigenspectra. This image should be constructed from the datacube itself
        to match the dimensionality. Sky regions should be marked as 0, and
        astronomical sources should be identified with an integer greater than
        or equal to 1. Default to None.
    interactive : bool
        If True, a :class:`~zap.Zap` object containing all information on
        the ZAP process is returned, and can be used to explore the
        eigenspectra and recompute the output (with the
        :meth:`~zap.Zap.reprocess` method). In this case, the output files
        are not saved (`outcubefits` and `skycubefits` are ignored). Default
        to False.
    varcurvefits : str
        Path for the optional output of the explained variance curves.

    """
    if logger is not None:
        logger.debug('--- Running ZAP %s', __version__)
    t0 = time()
    if not isinstance(cubefits, string_types):
        raise TypeError('The process method only accepts a single datacube '
                        'filename.')

    # check if outcubefits/skycubefits exists before beginning
    if not overwrite:
        _check_file_exists(outcubefits)
        _check_file_exists(skycubefits)

    if ncpu is not None:
        global NCPU
        NCPU = ncpu

    if extSVD is not None and mask is not None:
        raise ValueError('extSVD and mask parameters are incompatible: if mask'
                         ' must be used, then the SVD has to be recomputed')

    if mask is not None or (extSVD is None and cfwidthSVD != cfwidthSP):
        # Compute the SVD separately, only if a mask is given, or if the
        # cfwidth values differ and extSVD is not given. Otherwise, the SVD
        # will be computed in the _run method, which allows to avoid running
        # twice the zlevel and continuumfilter steps.
        extSVD = SVDoutput(cubefits, clean=clean, zlevel=zlevel,
                           cftype=cftype, cfwidth=cfwidthSVD, mask=mask,
                           instrument=instrument, boost=boost, segment_sky=segment_sky)

    zobj = Zap(cubefits, pca_class=pca_class, n_components=n_components,
               instrument=instrument, cube=cube, hdr=hdr, logger=logger,
               segment_sky=segment_sky)
    
    zobj._run(clean=clean, zlevel=zlevel, cfwidth=cfwidthSP, cftype=cftype,
              nevals=nevals, extSVD=extSVD, boost=boost, contmask=contmask,
              reject_factor=reject_factor)

    if interactive:
        # Return the zobj object without saving files
        return zobj

    if skycubefits is not None:
        zobj.writeskycube(skycubefits=skycubefits, overwrite=overwrite)

    if varcurvefits is not None:
        zobj.writevarcurve(varcurvefits=varcurvefits, overwrite=overwrite)

    zobj.mergefits(outcubefits, overwrite=overwrite)
    if logger is not None:
        logger.info('Zapped! (took %.2f sec.)', time() - t0)


def SVDoutput(cubefits, cube=None, hdr=None, stack=None, clean=True, zlevel='median', 
              cftype='weight', cfwidth=100, mask=None, ncpu=None, pca_class=None,
              n_components=None, instrument='MUSE', boost=50, contmask=None,
              reject_factor=0.0, logger=None, segment_sky=False):
    """Performs the SVD decomposition of a datacube.

    This allows to use the SVD for a different datacube. It used to allow to
    save the SVD to a file but this is no more possible. Instead it returns
    a ``Zap`` which can be given to the :func:`~zap.process` function.

    Parameters
    ----------
    cubefits : str
        Input FITS file, containing a cube with data in the first extension.
    clean : bool
        If True (default value), the NaN values are cleaned. Spaxels with more
        then 25% of NaN values are removed, the others are replaced with an
        interpolation from the neighbors.
    zlevel : str
        Method for the zeroth order sky removal: `none`, `sigclip` or `median`
        (default).
    cftype : {'weight', 'median', 'fit', 'none'}
        Method for the continuum filter. For the `weight` method, a zeroth
        order sky is required (see `zlevel`).
    cfwidth : int or float
        Window size for the continuum filter, default to 300.
    mask : str
        Path of a FITS file containing a mask (1 for objects, 0 for sky).

    """
    if logger is not None:
        logger.info('Processing %s to compute the SVD', cubefits)

    if ncpu is not None:
        global NCPU
        NCPU = ncpu

    zobj = Zap(cubefits, cube=cube, hdr=hdr, pca_class=pca_class, 
               n_components=n_components, instrument=instrument, 
               logger=logger, segment_sky=segment_sky)
    zobj._prepare(clean=clean, zlevel=zlevel, cftype=cftype,
                  cfwidth=cfwidth, mask=mask, stack=stack)
    zobj._msvd()

    #extend this to write out critical information?

    return zobj


def contsubfits(cubefits, outfits='CONTSUB_CUBE.fits', ncpu=None,
                cftype='median', cfwidth=300, clean_nan=True, zlevel='median',
                overwrite=False, logger=None):
    """A standalone implementation of the continuum removal."""
    if ncpu is not None:
        global NCPU
        NCPU = ncpu

    zobj = Zap(cubefits)
    zobj._prepare(clean=clean_nan, zlevel=zlevel, cftype=cftype,
                  cfwidth=cfwidth)
    cube = zobj.make_contcube()

    outhead = _newheader(zobj)
    outhdu = fits.PrimaryHDU(data=cube, header=outhead)
    write_hdulist_to(outhdu, outfits, overwrite=overwrite)
    if logger is not None:
        logger.info('Continuum cube file saved to %s', outfits)


def nancleanfits(cubefits, outfn='NANCLEAN_CUBE.fits', rejectratio=0.25,
                 boxsz=1, overwrite=False, logger=None):
    """Interpolates NaN values from the nearest neighbors.

    Parameters
    ----------
    cubefits : str
        Input FITS file, containing a cube with data in the first extension.
    outfn : str
        Output FITS file. Default to ``NANCLEAN_CUBE.fits``.
    rejectratio : float
        Defines a cutoff for the ratio of NAN to total pixels in a spaxel
        before the spaxel is avoided completely. Default to 0.25
    boxsz : int
        Defines the number of pixels around the offending NaN pixel.
        Default to 1, which looks for the 26 nearest neighbors which
        is a 3x3x3 cube.

    """
    with fits.open(cubefits) as hdu:
        hdu[1].data = _nanclean(hdu[1].data, rejectratio=rejectratio,
                                boxsz=boxsz)[0]
        write_hdulist_to(hdu, outfn, overwrite=overwrite)


def timeit(func, logger=None):
    @wraps(func)
    def wrapped(*args, **kwargs):
        t0 = time()
        res = func(*args, **kwargs)
        if logger is not None:
            logger.info('%s - Time: %.2f sec.', func.__name__, time() - t0)
        return res
    return wrapped


# ================= Main class =================

class Zap(object):

    """ Main class to run each of the steps of ZAP.

    Attributes
    ----------

    cleancube : numpy.ndarray
        The final datacube after removing all of the residual features.
    contarray : numpy.ndarray
        A 2D array containing the subtracted continuum per spaxel.
    cube : numpy.ndarray
        The original cube with the zlevel subtraction performed per spaxel.
    laxis : numpy.ndarray
        A 1d array containing the wavelength solution generated from the header
        parameters.
    wcs : astropy.wcs.WCS
        WCS object with the wavelength solution.
    lranges : list
        A list of the wavelength bin limits used in segmenting the sepctrum
        for SVD.
    nancube : numpy.ndarray
        A 3d boolean datacube containing True in voxels where a NaN value was
        replaced with an interpolation.
    nevals : numpy.ndarray
        A 1d array containing the number of eigenvalues used per segment to
        reconstruct the residuals.
    normstack : numpy.ndarray
        A normalized version of the datacube decunstructed into a 2d array.
    pranges : numpy.ndarray
        The pixel indices of the bounding regions for each spectral segment.
    recon : numpy.ndarray
        A 2d array containing the reconstructed emission line residuals.
    run_clean : bool
        Boolean that indicates that the NaN cleaning method was used.
    run_zlevel : bool
        Boolean indicating that the zero level correction was used.
    stack : numpy.ndarray
        The datacube deconstructed into a 2d array for use in the the SVD.
    variancearray : numpy.ndarray
        A list of length nsegments containing variances calculated per spaxel
        used for normalization
    y,x : numpy.ndarray
        The position in the cube of the spaxels that are in the 2d
        deconstructed stack
    zlsky : numpy.ndarray
        A 1d array containing the result of the zero level subtraction

    """

    def __init__(self, cubefits, cube=None, hdr=None, pca_class=None, 
                 n_components=None, instrument='MUSE', logger=None, 
                 segment_sky=False):

        self.logger = logger

        if cube is not None: #in case a cube is already supplied
            self.cube = cube
            self.header = hdr
            self.ins_mode = None
        else: #default behaviour 
            self.cubefits = cubefits
            with fits.open(cubefits) as hdul:
                self.ins_mode = hdul[0].header.get('HIERARCH ESO INS MODE')
                self.cube = hdul[1].data
                self.header = hdul[1].header

        # Workaround for floating points errors in wcs computation: if cunit is
        # specified, wcslib will convert in meters instead of angstroms, so we
        # remove cunit before creating the wcs object
        header = self.header.copy()
        unit = u.Unit(header.pop('CUNIT3'))
        self.wcs = WCS(header).sub([3])

        # Create Lambda axis
        #wlaxis = np.arange(self.cube.shape[0])
        try:
            wlaxis = np.arange(self.cube.shape[0])
        except AttributeError:
            wlaxis = np.arange(self.header['NAXIS3'])
        
        self.laxis = self.wcs.all_pix2world(wlaxis, 0)[0]
        if unit != u.angstrom:
            # Make sure lambda is in angstroms
            self.laxis = (self.laxis * unit).to(u.angstrom).value

        # Change laser region into zeros if AO
        if self.ins_mode in NOTCH_FILTER_RANGES:
            if self.logger is not None:
                self.logger.info('Cleaning laser region for AO, mode=%s, limits=%s',
                            self.ins_mode, NOTCH_FILTER_RANGES[self.ins_mode])
            self.notch_limits = self.wcs.all_world2pix(
                NOTCH_FILTER_RANGES[self.ins_mode], 0)[0].astype(int)
            lmin, lmax = self.notch_limits
            self.cube[lmin:lmax + 1] = 0.0

        # NaN Cleaning
        self.run_clean = False
        self.nancube = None
        self._boxsz = 1
        self._rejectratio = 0.25

        # Mask file
        self.maskfile = None

        # zlevel parameters
        self.run_zlevel = False
        self.zlsky = np.zeros_like(self.laxis)

        # Extraction results
        self.stack = None
        self.sigma_mask = None
        self.y = None
        self.x = None

        # Normalization Maps
        self.contarray = None
        self.variancearray = None
        self.normstack = None

        # identify the spectral range of the dataset
        laxmin = min(self.laxis)
        laxmax = max(self.laxis)

        ## List of segmentation limits
        skyseg = np.array([0, 26000])
        contseg = np.array([0, 10140, 10670, 11250, 11960, 12450, 12890, 14000, 14720,
                            15650, 16000, 16356, 17253, 18400, 19450, 20000, 20815, 
                            22000, 23500, 24600, 26000])
        contflag = np.array([0,0,0,0,0,1,0,0,0,1,0,0,0,0,1,0,1,0,0,0])

        #skyseg = np.array(SKYSEG)
        crop_sky = (skyseg > laxmin) & (skyseg < laxmax)
        crop_cont = (contseg > laxmin) & (contseg < laxmax)
        skyseg = skyseg[crop_sky]
        contseg = contseg[crop_cont]
        contflag = contflag[crop_cont[1:]]

        # sky segment limit in angstroms
        self.lranges = (np.vstack([np.append(laxmin - 10, skyseg),
                                   np.append(skyseg, laxmax + 10)])).T
       
        # continuum segment limit in angstroms
        self.cranges = (np.vstack([np.append(laxmin - 10, contseg),
                                   np.append(contseg, laxmax + 10)])).T

        # flags for continuum estimation
        self.franges = [flag for ii, flag in enumerate(contflag) if contseg[ii]]

        # segment limit in pixels
        laxis = self.laxis
        lranges = self.lranges
        pranges = []
        for i in range(len(lranges)):
            paxis = wlaxis[(laxis > lranges[i, 0]) & (laxis <= lranges[i, 1])]
            pranges.append((np.min(paxis), np.max(paxis) + 1))
        self.pranges = np.array(pranges)

        # continuum segment limit in pixels
        laxis = self.laxis
        cranges = self.cranges
        cpranges = []
        for i in range(len(cranges)):
            cpaxis = wlaxis[(laxis > cranges[i, 0]) & (laxis <= cranges[i, 1])]
            cpranges.append((np.min(cpaxis), np.max(cpaxis) + 1))
        self.cpranges = np.array(cpranges)

        # eigenspace Subset
        if pca_class is not None:
            if self.logger is not None:
                self.logger.debug('Using %s', pca_class)
            self.pca_class = pca_class
        else:
            self.pca_class = PCA

        # Reconstruction of sky features
        self.n_components = n_components
        self.recon = None
        self.cleancube = None

        #for masking, require an external array
        self.flag_extSVD = False

    @timeit
    def _prepare(self, clean=True, zlevel='median', cftype='weight',
                 cfwidth=300, extzlevel=None, mask=None, boost=50,
                 contmask=None, reject_factor=0.0, stack=None):
        # Check for consistency between weighted median and zlevel keywords
        if cftype == 'weight' and zlevel == 'none':
            raise ValueError('Weighted median requires a zlevel calculation')
        # clean up the nan values
        if clean:
            self._nanclean()

        # if mask is supplied, apply it
        if mask is not None:
            self._applymask(mask)

        # Extract the spectra that we will be working with
        # unless extracted stack is already provided
        if stack is None:
           self._extract()
        else:
           self.stack = stack

        # remove the median along the spectral axis
        if zlevel.lower() != 'none':
            if extzlevel is None:
                self._zlevel(calctype=zlevel)
            else:
                self._externalzlevel(extzlevel)
                
        # handling rejection of spectral channels in continuum estimate (1=bad)
        if contmask is None:
            contmask = np.zeros(self.stack.shape[0], dtype=int)
        elif contmask.shape[0] != self.stack.shape[0]:
            raise ValueError('Input spectra and continuum mask must have same first dimension')

        # remove the continuum level - this is multiprocessed to speed it up
        self._continuumfilter(cfwidth=cfwidth, cftype=cftype, boost=boost,
                              contmask=contmask, reject_factor=reject_factor)
        #flag deviant pixels in the continuum-cleaned spectra
        self._flag_pixels()

        # normalize the variance in the segments.
        self._normalize_variance()

    def _run(self, clean=True, zlevel='median', cftype='weight',
             cfwidth=300, nevals=[], extSVD=None, boost=50,
             contmask=None, reject_factor=0.0):
        """ Perform all steps to ZAP a datacube:

        - NaN re/masking,
        - deconstruction into "stacks",
        - zerolevel subraction,
        - continuum removal,
        - normalization,
        - singular value decomposition,
        - eigenvector selection,
        - residual reconstruction and subtraction,
        - data cube reconstruction.

        """
        self._prepare(clean=clean, zlevel=zlevel, cftype=cftype,
                      cfwidth=cfwidth, extzlevel=extSVD, boost=boost,
                      contmask=contmask, reject_factor=reject_factor)

        # do the multiprocessed SVD calculation
        if extSVD is None:
            self._msvd()
        elif isinstance(extSVD, Zap):
            self.models = extSVD.models
        else:
            self.models = self.readPCA(extSVD)
            self.flag_extSVD = True
        self.components = [m.components_.copy() for m in self.models]

        # choose some fraction of eigenspectra or some finite number of
        # eigenspectra
        if nevals == []:
            self.optimize()
            self.chooseevals(nevals=self.nevals)
        else:
            self.chooseevals(nevals=nevals)

        # reconstruct the sky residuals using the subset of eigenspace
        self.reconstruct()

        # stuff the new spectra back into the cube
        self.remold()

    def _nanclean(self):
        """
        Detects NaN values in cube and removes them by replacing them with an
        interpolation of the nearest neighbors in the data cube. The positions
        in the cube are retained in nancube for later remasking.
        """
        self.cube, self.nancube = _nanclean(
            self.cube, rejectratio=self._rejectratio, boxsz=self._boxsz)
        self.run_clean = True

    @timeit
    def _extract(self):
        """Deconstruct the datacube into a 2d array.

        Since spatial information is not required, and the linear algebra
        routines require 2d arrays. The operation rejects any spaxel with even
        a single NaN value, since this would cause the linear algebra routines
        to crash.

        Adds the x and y data of these positions into the Zap class

        """
        # make a map of spaxels with NaNs
        #badmap = (np.logical_not(np.isfinite(self.cube))).sum(axis=0)
        badmap = np.zeros_like(self.cube, dtype=bool).sum(axis=0)

        # get positions of those with no NaNs
        self.y, self.x = np.where(badmap == 0)
        
        # extract those positions into a 2d array
        self.stack = self.cube[:, self.y, self.x]

        if self.logger is not None:
            self.logger.debug('--- Extract to 2D, %d valid spaxels (%d%%)', len(self.x),
                        len(self.x) / np.prod(self.cube.shape[1:]) * 100)


    def _flag_pixels(self):
        #setup masked array for inversion
        mstack = np.nanmedian(self.normstack, axis=1)
        dev = np.clip(mad_std(self.normstack, axis=1, ignore_nan=True), 1e-10, None)
        self.sigma_mask = (np.fabs(self.normstack-mstack[:,np.newaxis])/dev[:,np.newaxis] > 6.)
       
        #iteratively clean pixels for reconstruction
        for ii in range(5):
            old_mask = np.copy(self.sigma_mask)
            tarray = np.copy(self.normstack)
            tarray[self.sigma_mask] = np.nan

            mstack = np.nanmedian(tarray, axis=1)
            dev = np.clip(mad_std(tarray, axis=1, ignore_nan=True), 1e-10, None)
            self.sigma_mask = (np.fabs(self.normstack-mstack[:,np.newaxis])/dev[:,np.newaxis] > 6.)
            if np.all(self.sigma_mask == old_mask):
                break

    def _externalzlevel(self, extSVD):
        """Remove the zero level from the extSVD file."""
        if self.logger is not None:
            self.logger.debug('Using external zlevel from %s', extSVD)
        if isinstance(extSVD, Zap):
            self.zlsky = np.array(extSVD.zlsky, copy=True)
            self.run_zlevel = extSVD.run_zlevel
        else:
            self.zlsky = fits.getdata(extSVD, 0)
            self.run_zlevel = 'extSVD'
        self.stack -= self.zlsky[:, np.newaxis]

    @timeit
    def _zlevel(self, calctype='median'):
        """
        Removes a 'zero' level from each spectral plane. Spatial information is
        not required, so it operates on the extracted stack.

        Operates on stack, leaving it with this level removed and adds the data
        'zlsky' to the class. zlsky is a spectrum of the zero levels.

        This zero level is currently calculated with a median.

        Experimental operations -

        - exclude top quartile
        - run in an iterative sigma clipped mode

        """
        self.run_zlevel = calctype
        if calctype == 'none':
            if self.logger is not None:
                self.logger.debug('Skipping zlevel subtraction')
            return

        if calctype == 'median':
            if self.logger is not None:
                self.logger.debug('Median zlevel subtraction')
            func = _imedian
        elif calctype == 'sigclip':
            if self.logger is not None:
                self.logger.debug('Iterative Sigma Clipping zlevel subtraction')
            func = _isigclip
        else:
            raise ValueError('Unknow zlevel type, must be none, median, or '
                             'sigclip')

        self.zlsky = np.hstack(parallel_map(func, self.stack, NCPU, axis=0))
        self.stack -= self.zlsky[:, np.newaxis]

    @timeit
    def _continuumfilter(self, cfwidth=300, cftype='weight', boost=50, 
                         contmask=None, reject_factor=0.0):
        """A multiprocessed implementation of the continuum removal.

        This process distributes the data to many processes that then
        reassemble the data.  Uses two filters, a small scale (less than the
        line spread function) uniform filter, and a large scale median filter
        to capture the structure of a variety of continuum shapes.

        added to class
        contarray - the removed continuua
        normstack - "normalized" version of the stack with the continuua
            removed

        """
        if cftype not in CFTYPE_OPTIONS:
            raise ValueError("cftype must be weight, median, fit or none, "
                             "got {}".format(cftype))
        if self.logger is not None:
            self.logger.debug('--- Applying Continuum Filter, cftype=%s', cftype)
        self._cftype = cftype
        self._cfwidth = cfwidth

        weight = None
        mboost = None
        if cftype == 'weight':
            weight = np.abs(self.zlsky - (np.max(self.zlsky) + 1))
        elif cftype == 'med_smooth':
            mboost = boost
        # remove continuum features
        if cftype == 'none':
            self.normstack = self.stack.copy()
        else:
            self.contarray = _continuumfilter(self.stack, cftype, weight=weight, 
                                              cfwidth=cfwidth, boost=mboost)

            # for reject_factor > 0, interpolate continuum for pixels with fewer
            # than reject_factor*cfwidth "good" pixels
            if contmask is not None and reject_factor > 0.0:

                if contmask.shape[0] != self.stack.shape[0]:
                    raise ValueError("Input spectra and continuum mask must have the same first dimension")

                # set fraction of window which needs to be "good"
                reject_threshold = self._cfwidth*reject_factor

                # filter the continuum mask
                pad_mask = np.pad(contmask, (cfwidth//2, cfwidth//2), 'constant', constant_values=0)
                fmask = np.sum(rolling_window(pad_mask, cfwidth), axis=1)

                # identify pixels with too few "good" continuum pixels
                flag_bad = (fmask >= reject_threshold)

                self.contarray[flag_bad,:] = interp1d(np.arange(contmask.shape[0])[~flag_bad],
                        self.contarray[~flag_bad,:], axis=0, bounds_error=False,
                        fill_value='extrapolate')(np.arange(contmask.shape[0])[flag_bad])
 
            # check continuum against segment flags to avoid very broad features
            if any(self.franges):
               for sflag, spix in zip(self.franges, self.cpranges):
                    if sflag:
                        parray = np.vstack((self.contarray[spix[0],:], self.contarray[spix[1],:]))
                        self.contarray[spix[0]:spix[1],:] = \
                                interp1d(np.array(spix),parray, axis=0)(np.arange(spix[0],spix[1]))


            self.normstack = self.stack - self.contarray

    def _normalize_variance(self):
        """Normalize the variance in the segments."""
        if self.logger is not None:
            self.logger.debug('--- Normalizing variances')

        nseg = len(self.pranges)
        self.variancearray = var = np.zeros((nseg, self.stack.shape[1]))
        for i in range(nseg):
            pmin, pmax = self.pranges[i]
            var[i, :] = np.maximum(np.nanvar(self.normstack[pmin:pmax, :], axis=0), 1.0e-10)
            self.normstack[pmin:pmax, :] /= var[i, :]

    @timeit
    def _msvd(self):
        """Multiprocessed singular value decomposition.

        Takes the normalized, spectral segments and distributes them
        to the individual svd methods.

        """
        if self.logger is not None:
            self.logger.info('--- Calculating SVD on %d segments (%s)', len(self.pranges),
                    self.pranges)
        indices = [x[0] for x in self.pranges[1:]]
        # normstack = self.stack - self.contarray
        Xarr = np.array_split(self.normstack.T, indices, axis=1)

        self.models = []
        for i, x in enumerate(Xarr):
            if self.n_components is not None:
                ncomp = max(x.shape[1] * self.n_components, 60)
                if self.logger is not None:
                    self.logger.debug('--- Segment %d, computing %d eigenvectors out of %d',
                                i, ncomp, x.shape[1])
            else:
                ncomp = None

            self.models.append(self.pca_class(n_components=ncomp).fit(x))


    def chooseevals(self, nevals=[]):
        """Choose the number of eigenspectra/evals to use for reconstruction.

        User supplies the number of eigen spectra to be used (neval) or the
        percentage of the eigenspectra that were calculated (peval) from each
        spectral segment to be used.

        The user can either provide a single value to be used for all segments,
        or provide an array that defines neval or peval per segment.

        """
        nranges = len(self.pranges)
        nevals = np.atleast_1d(nevals)

        # deal with an input list
        if len(nevals) > 1:
            if len(nevals) != nranges:
                nevals = np.array([nevals[0]])
                if self.logger is not None:
                    self.logger.warning('--- Chosen eigenspectra array does not correspond to '
                                'number of segments')
            else:
                if self.logger is not None:
                    self.logger.info('--- Choosing %s eigenspectra for segments', nevals)

        if len(nevals) == 1:
            if self.logger is not None:
                self.logger.debug('--- Choosing %s eigenspectra for all segments', nevals)
            nevals = np.zeros(nranges, dtype=int) + nevals

        if nevals.ndim == 1:
            start = np.zeros(nranges, dtype=int)
            end = nevals
        else:
            start, end = nevals.T

        self.nevals = nevals
        for i, model in enumerate(self.models):
            model.components_ = self.components[i][start[i]:end[i]]

    @timeit
    def reconstruct(self):
        """Reconstruct the residuals from a given set of eigenspectra and
        eigenvalues
        """
        if self.logger is not None:
            self.logger.info('--- Reconstructing Sky Residuals')
        indices = [x[0] for x in self.pranges[1:]]

        normstack_ma = np.ma.masked_array(self.normstack, mask=self.sigma_mask)

        #use custom transform and inverse transform methods for masked array
        if self.flag_extSVD:
            Xarr = np.array_split(normstack_ma.T, indices, axis=1)
            Xnew = [model.transform_ma(x)
                    for model, x in zip(self.models, Xarr)]
            Xinv = [model.inverse_transform_ma(x)
                    for model, x in zip(self.models, Xnew)]
        else:
            Xarr = np.array_split(self.normstack.T, indices, axis=1)
            Xnew = [model.transform(x)
                    for model, x in zip(self.models, Xarr)]
            Xinv = [model.inverse_transform(x)
                    for model, x in zip(self.models, Xnew)]
        
        self.recon = np.concatenate([x.T * self.variancearray[i, :]
                                     for i, x in enumerate(Xinv)])

    def make_cube_from_stack(self, stack, with_nans=False):
        """Stuff the stack back into a cube."""
        cube = self.cube.copy()
        cube[:, self.y, self.x] = stack
        if with_nans:
            cube[self.nancube] = np.nan
        if self.ins_mode in NOTCH_FILTER_RANGES:
            lmin, lmax = self.notch_limits
            cube[lmin:lmax + 1] = np.nan
        return cube

    def remold(self):
        """ Subtracts the reconstructed residuals and places the cleaned
        spectra into the duplicated datacube.
        """
        if self.logger is not None:
            self.logger.info('--- Applying correction and reshaping data product')
        self.cleancube = self.make_cube_from_stack(self.stack - self.recon,
                                                   with_nans=self.run_clean)

    def reprocess(self, nevals=[]):
        """ A method that redoes the eigenvalue selection, reconstruction, and
        remolding of the data.
        """
        self.chooseevals(nevals=nevals)
        self.reconstruct()
        self.remold()

    def optimize(self):
        """Compute the optimal number of components needed to characterize
        the residuals.

        This function calculates the variance per segment with an increasing
        number of eigenspectra/eigenvalues. It then deterimines the point at
        which the second derivative of this variance curve reaches zero. When
        this occurs, the linear reduction in variance is attributable to the
        removal of astronomical features rather than emission line residuals.

        """
        if self.logger is not None:
            self.logger.debug('--- Compute number of components')
        ncomp = []
        for model in self.models:
            ##original ZAPv2
            #var = model.explained_variance_
            #deriv, mn1, std1 = _compute_deriv(var, nsigma=2)
            #cross = np.append([False], deriv >= (mn1 - std1))
            #ncomp.append(np.where(cross)[0][0])
    
            #modified to mimic ZAPv1
            var = model.explained_variance_
            deriv, mn1, std1 = _compute_deriv(var, nsigma=2)
            cross = np.append([False], deriv >= (mn1 - std1))
            deriv2, mn2, std2 = _compute_deriv(deriv, nsigma=2)
            cross2 = np.append([False,False], np.fabs(deriv2) <= (mn2 + std2))
            arrs = np.logical_or(cross, cross2)
            ncomp.append(np.where(arrs)[0][0])

        self.nevals = np.array(ncomp)

    def make_contcube(self):
        """ Remold the continuum array so it can be investigated.

        Takes the continuum stack and returns it into a familiar cube form.
        """
        contcube = self.cube.copy() * np.nan
        contcube[:, self.y, self.x] = self.contarray
        return contcube

    def _applymask(self, mask):
        """Apply a mask to the input data to provide a cleaner basis set.

        mask is >1 for objects, 0 for sky so that people can use sextractor.
        The file is read with ``astropy.io.fits.getdata`` which first tries to
        read the primary extension, then the first extension is no data was
        found before.

        """
        if self.logger is not None:
            self.logger.debug('-- Applying Mask for SVD Calculation from %s', mask)
        self.maskfile = mask
        mask = fits.getdata(mask).astype(bool)
        nmasked = np.count_nonzero(mask)
        if self.logger is not None:
            self.logger.debug('-- Masking %d pixels (%d%%)', nmasked,
                    nmasked / np.prod(mask.shape) * 100)
        self.cube[:, mask] = np.nan

    def writecube(self, outcubefits='DATACUBE_ZAP.fits', overwrite=False):
        """Write the processed datacube to an individual fits file."""
        outhead = _newheader(self)
        outhdu = fits.PrimaryHDU(data=self.cleancube, header=outhead)
        write_hdulist_to(outhdu, outcubefits, overwrite=overwrite)
        if self.logger is not None:
            self.logger.info('--- Cube file saved to %s', outcubefits)

    def writeskycube(self, skycubefits='SKYCUBE_ZAP.fits', overwrite=False):
        """Write the processed datacube to an individual fits file."""
        outcube = self.cube - self.cleancube
        outhead = _newheader(self)
        outhdu = fits.PrimaryHDU(data=outcube, header=outhead)
        write_hdulist_to(outhdu, skycubefits, overwrite=overwrite)
        if self.logger is not None:
            self.logger.info('--- Sky cube file saved to %s', skycubefits)

    def writevarcurve(self, varcurvefits='VARCURVE_ZAP.fits', overwrite=False):
        """Write the explained variance curves to an individual fits file."""
        from astropy.table import Table
        table = Table([m.explained_variance_ for m in self.models])
        hdu = fits.table_to_hdu(table)
        _newheader(self, hdu.header)
        write_hdulist_to(hdu, varcurvefits, overwrite=overwrite)
        if self.logger is not None:
            self.logger.info('--- Variance curve file saved to %s', varcurvefits)

    def mergefits(self, outcubefits, overwrite=False):
        """Merge the ZAP cube into the full muse datacube and write."""
        # make sure it has the right extension
        outcubefits = outcubefits.split('.fits')[0] + '.fits'
        with fits.open(self.musecubefits) as hdu:
            hdu[1].header = _newheader(self)
            hdu[1].data = self.cleancube
            write_hdulist_to(hdu, outcubefits, overwrite=overwrite)
        if self.logger is not None:
            self.logger.info('--- Cube file saved to %s', outcubefits)

    def writePCA(self, pcaoutputfits='ZAP_PCA.fits', overwrite=False):
        """Write critical PCA information to a fits file."""
        if not overwrite:
            _check_file_exists(pcaoutputfits)

        header = fits.Header()
        header['ZAPvers'] = (__version__, 'ZAP version')
        header['ZAPzlvl'] = (self.run_zlevel, 'ZAP zero level correction')
        header['ZAPclean'] = (self.run_clean,
                          'ZAP NaN cleaning performed for calculation')
        header['ZAPcftyp'] = (self._cftype, 'ZAP continuum filter type')
        header['ZAPcfwid'] = (self._cfwidth, 'ZAP continuum filter size')
        header['ZAPmask'] = (self.maskfile, 'ZAP mask used to remove sources')
        nseg = len(self.models)
        header['ZAPnseg'] = (nseg, 'Number of segments used for ZAP SVD')
    
        hdu = fits.HDUList([fits.PrimaryHDU(data=self.zlsky, header=header)])
        for i, model in enumerate(self.models):
            thead = fits.Header()
            thead['Whiten'] = model.whiten
            hdu.append(fits.ImageHDU(data=model.mean_, header=thead))
            hdu.append(fits.ImageHDU(data=model.explained_variance_))
            hdu.append(fits.ImageHDU(data=model.components_))
        
        # write for later use
        hdu.writeto(pcaoutputfits,overwrite=overwrite)
        if self.logger is not None:
            self.logger.info('--- PCA components saved to %s', pcaoutputfits)

    def readPCA(self, pcainputfits):
        """Read int stored PCA information."""

        if not os.path.exists(pcainputfits):
            raise IOError("The requested PCA file does not exist")

        hdu = fits.open(pcainputfits)

        #check that the input array has the same number of segments
        nseg = hdu[0].header['ZAPnseg']
        if nseg != len(self.pranges):
            raise AttributeError("The input PCA array was generated with different segments")

        self.zlsky = hdu[0].data
        models = []
        for i in range(nseg):
            wht = hdu[(3*i)+1].header['whiten']
            temp_mean = hdu[(3*i)+1].data
            temp_var = hdu[(3*i)+2].data
            temp_comp = hdu[(3*i)+3].data

            models.append(PCAobj(mean=temp_mean, var=temp_var, comp=temp_comp, whiten=wht))
        hdu.close()
        
        if self.logger is not None:
            self.logger.info('--- PCA components read from %s', os.path.split(pcainputfits)[1])
        
        return models

    def plotvarcurve(self, i=0, ax=None):
        var = self.models[i].explained_variance_
        deriv, mn1, std1 = _compute_deriv(var)

        if ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(3, 1, figsize=[10, 15])

        ax1, ax2, ax3 = ax
        ax1.plot(var, linewidth=3)
        ax1.plot([self.nevals[i], self.nevals[i]], [min(var), max(var)])
        ax1.set_ylabel('Variance')

        ax2.plot(np.arange(deriv.size), deriv)
        ax2.hlines([mn1, mn1 - std1], 0, len(deriv), colors=('k', '0.5'))
        ax2.plot([self.nevals[i] - 1, self.nevals[i] - 1],
                 [min(deriv), max(deriv)])
        ax2.set_ylabel('d/dn Var')

        deriv2 = np.diff(deriv)
        ax3.plot(np.arange(deriv2.size), np.abs(deriv2))
        ax3.plot([self.nevals[i] - 2, self.nevals[i] - 2],
                 [min(deriv2), max(deriv2)])
        ax3.set_ylabel('(d^2/dn^2) Var')
        # ax3.set_xlabel('Number of Components')

        ax1.set_title('Segment {0}, {1} - {2} Angstroms'.format(
            i, self.lranges[i][0], self.lranges[i][1]))

    def plotvarcurves(self):
        nseg = len(self.models)
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(nseg, 3, figsize=(16, nseg * 2),
                                 tight_layout=True)
        for i in range(nseg):
            self.plotvarcurve(i=i, ax=axes[i])


# ================= Helper Functions =================

if ASTROPY_LT_1_3:
    # the 'clobber' parameter was renamed to 'overwrite' in 1.3
    def write_hdulist_to(hdulist, fileobj, overwrite=False, **kwargs):
        hdulist.writeto(fileobj, clobber=overwrite, **kwargs)
else:
    def write_hdulist_to(hdulist, fileobj, overwrite=False, **kwargs):
        hdulist.writeto(fileobj, overwrite=overwrite, **kwargs)


def worker(f, i, chunk, out_q, err_q, kwargs):
    try:
        result = f(i, chunk, **kwargs)
    except Exception as e:
        err_q.put(e)
        return

    # output the result and task ID to output queue
    out_q.put((i, result))


def parallel_map(func, arr, indices, logger=None, **kwargs):
    if logger is not None:
        logger.debug('Running function %s with indices: %s',
                     func.__name__, indices)
    axis = kwargs.pop('axis', None)
    if isinstance(indices, (int, np.integer)) and indices == 1:
        return [func(0, arr, **kwargs)]
    freeze_support()
    manager = Manager()
    out_q = manager.Queue()
    err_q = manager.Queue()
    jobs = []
    chunks = np.array_split(arr, indices, axis=axis)
    if 'split_arrays' in kwargs:
        split_arrays = [np.array_split(a, indices, axis=axis)
                        for a in kwargs.pop('split_arrays')]
    else:
        split_arrays = None

    for i, chunk in enumerate(chunks):
        if split_arrays:
            kwargs['split_arrays'] = [s[i] for s in split_arrays]
        p = Process(target=worker, args=(func, i, chunk, out_q, err_q, kwargs))
        jobs.append(p)
        p.start()

    # gather the results
    for proc in jobs:
        proc.join()

    if not err_q.empty():
        # kill all on any exception from any one slave
        raise err_q.get()

    # Processes finish in arbitrary order. Process IDs double
    # as index in the resultant array.
    results = [None] * len(jobs)
    while not out_q.empty():
        idx, result = out_q.get()
        results[idx] = result

    return results


def _compute_deriv(arr, nsigma=5):
    """Compute statistics on the derivatives"""
    #ZAPv1 implementation
    deriv = np.diff(arr)
    ind = int(.5 * (arr.size-2))
    mn1 = deriv[ind:].mean()
    std1 = deriv[ind:].std() * nsigma
    
    ##original ZAPv2 implementation
    #npix = int(0.25 * arr.shape[0])
    #deriv = np.diff(arr[:npix])
    #ind = int(.15 * deriv.size)
    #mn1 = deriv[ind:].mean()
    #std1 = deriv[ind:].std() * nsigma
    return deriv, mn1, std1


def _continuumfilter(stack, cftype, weight=None, cfwidth=300, boost=None, logger=None):
    freeze_support()
    if cftype == 'fit':
        x = np.arange(stack.shape[0])
        # Excluding the very red part for the fit. This is Muse-specific,
        # but anyway for another instrument this method should probably
        # not be used as is.
        res = np.polynomial.polynomial.polyfit(x[:3600], stack[:3600], deg=5)
        ret = np.polynomial.polynomial.polyval(x, res, tensor=True)
        return ret.T

    if cftype == 'median':
        func = _icfmedian
        weight = None
    elif cftype == 'weight':
        func = _icfweight
    elif cftype == 'med_smooth':
        func = _icfmed_smooth
        weight = None
    if logger is not None:    
        logger.debug('-- Using cfwidth=%d', cfwidth)
    c = parallel_map(func, stack, NCPU, axis=1, weight=weight, cfwidth=cfwidth,
                     boost=boost)
    return np.concatenate(c, axis=1)


def _icfweight(i, stack, weight=None, cfwidth=None, boost=None):
    return np.array([wmedian(row, weight, cfwidth=cfwidth)
                     for row in stack.T]).T


def _icfmedian(i, stack, weight=None, cfwidth=None, boost=None):
    ufilt = 3  # set this to help with extreme over/under corrections
    return ndi.median_filter(
        ndi.uniform_filter(stack, (ufilt, 1)), (cfwidth, 1))


def _icfmed_smooth(i, stack, weight=None, cfwidth=None, boost=None):
    ufilt = cfwidth//2
    return ndi.uniform_filter(
        ndi.percentile_filter(stack, boost, (cfwidth, 1)), (ufilt, 1))


def rolling_window(a, window):  # function for striding to help speed up
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def wmedian(spec, wt, cfwidth=300):
    """Performs a weighted median filtering of a 1d spectrum.

    Operates using a cumulative sum curve.

    Parameters
    ----------
    spec : numpy.ndarray
        Input 1d spectrum to be filtered
    wt : numpy.ndarray
        A spectrum of equal length as the input array to provide the weights.
    cfwidth : int or float
        Window size for the continuum filter, for the SVD computation.
        Default to 300.

    """
    # ignore the warning (feature not a bug)
    old_settings = np.seterr(divide='ignore')
    spec = np.pad(spec, (cfwidth, cfwidth), 'constant', constant_values=0)
    wt = np.abs(wt)
    wt = np.pad(wt, (cfwidth, cfwidth), 'constant',
                constant_values=(np.min(wt) / 1000., np.min(wt) / 1000.))

    # do some striding for speed
    swin = rolling_window(spec, cfwidth)  # create window container array
    wwin = rolling_window(wt, cfwidth)  # create window container array

    # sort based on data
    srt = np.argsort(swin, axis=-1)
    ind = np.ogrid[0:swin.shape[0], 0:swin.shape[1]]
    sdata = swin[ind[0], srt]
    swt = wwin[ind[0], srt]

    # calculate accumulated weights
    awt = np.cumsum(swt, axis=-1)

    # new weightsort for normalization and consideration of data
    nw = (awt - 0.5 * swt) / awt[:, -1][:, np.newaxis]

    # find the midpoint in the new weight sort
    s = np.argmin(np.abs(nw - 0.5), axis=-1)
    sl = np.arange(len(s))
    nws = nw[sl, s]
    nws1 = nw[sl, s - 1]

    f1 = (nws - 0.5) / (nws - nws1)
    f2 = (0.5 - nws1) / (nws - nws1)
    wmed = sdata[sl, s - 1] * f1 + sdata[sl, s] * f2
    width = cfwidth // 2
    wmed = wmed[width:-width - 1]
    np.seterr(old_settings['divide'])

    return wmed


def _newheader(zobj, header=None):
    """Put the pertinent zap parameters into the header"""
    header = header or zobj.header.copy()
    header['COMMENT'] = 'These data have been ZAPped!'
    header.append(('ZAPvers', __version__, 'ZAP version'), end=True)
    # zlevel removal performed
    header.append(('ZAPzlvl', zobj.run_zlevel, 'ZAP zero level correction'))
    # Nanclean performed
    header['ZAPclean'] = (zobj.run_clean,
                          'ZAP NaN cleaning performed for calculation')
    # Continuum Filtering
    header['ZAPcftyp'] = (zobj._cftype, 'ZAP continuum filter type')
    header['ZAPcfwid'] = (zobj._cfwidth, 'ZAP continuum filter size')

    # number of segments
    nseg = len(zobj.pranges)
    header['ZAPnseg'] = (nseg, 'Number of segments used for ZAP SVD')

    # per segment variables
    if hasattr(zobj, 'nevals'):
        for i in range(nseg):
            header['ZAPseg{}'.format(i)] = (
                '{}:{}'.format(zobj.pranges[i][0], zobj.pranges[i][1] - 1),
                'spectrum segment (pixels)')
            header['ZAPnev{}'.format(i)] = (zobj.nevals[i],
                                            'number of eigenvals/spectra used')

    return header


def _isigclip(i, istack):
    mn = []
    for col in istack:
        clipped = sigma_clip(col, sigma_lower=3, sigma_upper=3, stdfunc=mad_std)
        mn.append(clipped.mean())
    return np.array(mn)


def _imedian(i, istack):
    return np.median(istack, axis=1)


@timeit
def _nanclean(cube, rejectratio=0.25, boxsz=1, logger=None):
    """
    Detects NaN values in cube and removes them by replacing them with an
    interpolation of the nearest neighbors in the data cube. The positions in
    the cube are retained in nancube for later remasking.

    """
    if logger is not None:
        logger.info('Cleaning NaN values in the cube')
    cleancube = cube.copy()
    badcube = np.logical_not(np.isfinite(cleancube))        # find NaNs
    badmap = badcube.sum(axis=0)  # map of total nans in a spaxel

    # choose some maximum number of bad pixels in the spaxel and extract
    # positions
    badmask = badmap > (rejectratio * cleancube.shape[0])
    if logger is not None:
        logger.info('Rejected %d spaxels with more than %.1f%% NaN pixels',
                    np.count_nonzero(badmask), rejectratio * 100)

    # make cube mask of bad spaxels
    badcube &= (~badmask[np.newaxis, :, :])
    z, y, x = np.where(badcube)

    neighbor = np.zeros((z.size, (2 * boxsz + 1)**3))
    icounter = 0
    if logger is not None:
        logger.info("Fixing %d remaining NaN pixels", len(z))

    # loop over samplecubes
    nz, ny, nx = cleancube.shape
    for j in range(-boxsz, boxsz + 1, 1):
        for k in range(-boxsz, boxsz + 1, 1):
            for l in range(-boxsz, boxsz + 1, 1):
                iz, iy, ix = z + l, y + k, x + j
                outsider = ((ix <= 0) | (ix >= nx - 1) |
                            (iy <= 0) | (iy >= ny - 1) |
                            (iz <= 0) | (iz >= nz - 1))
                ins = ~outsider
                neighbor[ins, icounter] = cleancube[iz[ins], iy[ins], ix[ins]]
                neighbor[outsider, icounter] = np.nan
                icounter = icounter + 1

    cleancube[z, y, x] = np.nanmean(neighbor, axis=1)
    return cleancube, badcube


def _check_file_exists(filename):
    if filename is not None and os.path.exists(filename):
        raise IOError('Output file "{0}" exists'.format(filename))

##### new functions for dealing with PCA input/output ##

class PCAobj(_BasePCA):
    def __init__(self, mean=None, var=None, comp=None, whiten=False):
        self.components_ = comp
        self.explained_variance_ = var
        self.mean_ = mean
        self.whiten=whiten

    def fit(self):
        pass

    #include transform methods which work on masked arrays to deal with catastrophic
    #outliers in the spectra
    def transform_ma(self, X):
        """Apply dimensionality reduction to X.
        X is projected on the first principal components 
        previously extracted from a training set.

        Modified from the standard sklearn routine to use
        masked array routines
        """
        if self.mean_ is not None:
            X = X - self.mean_
        X_transformed = np.array(np.ma.dot(X, self.components_.T))
        if self.whiten:
            X_transformed /= np.sqrt(self.explained_variance_)
        return X_transformed

    def inverse_transform_ma(self, X):
        """Transform data back to its original space.
        In other words, return an input X_original whose transform would be X.

        Modified from the standard sklearn routine to use
        masked array routines
        """
        if self.whiten:
            return np.array(np.ma.dot(X, np.sqrt(self.explained_variance_[:, np.newaxis]) *
                            self.components_)) + self.mean_
        else:
            return np.array(np.ma.dot(X, self.components_)) + self.mean_


