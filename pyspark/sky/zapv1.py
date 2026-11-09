# ZAP - Zurich Atmosphere Purge
#
#    Copyright (c) 2016 Kurt Soto
#
#    Permission is hereby granted, free of charge, to any person obtaining
#    a copy of this software and associated documentation files (the
#    "Software"), to deal in the Software without restriction, including
#    without limitation the rights to use, copy, modify, merge, publish,
#    distribute, sublicense, and/or sell copies of the Software, and to permit
#    persons to whom the Software is furnished to do so, subject to the
#    following conditions:
#
#    The above copyright notice and this permission notice shall be included in
#    all copies or substantial portions of the Software.
#
#    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
#    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#    DEALINGS IN THE SOFTWARE.
#
####################################################################
# Nov. 2017 - This version of the code has been modified by Trevor Mendel 
# for use with KMOS data and inclusion in the pySPARK software package 
# including adaptations for working with many sky exposures.
####################################################################

from __future__ import absolute_import, division, print_function

import astropy.units as u
import numpy as np
import os
import sys

from astropy.io import fits
from astropy.wcs import WCS
from functools import wraps
from multiprocessing import cpu_count, Manager, Process
from scipy import ndimage as ndi
from scipy.stats import sigmaclip
from astropy.stats import sigma_clip
from time import time

from scipy.interpolate import interp1d
from scipy import linalg

import matplotlib.pyplot as pl

from ._zap_version import __version1__ as __version__

__all__ = ['process', 'SVDoutput']


# Limits of the segments in Angstroms
SKYSEG_MUSE = [0, 5400, 5850, 6440, 6750, 7200, 7700, 8265, 8602, 8731, 9275, 10000]
SKYFLAG_MUSE = np.zeros(len(SKYSEG_MUSE)-1, dtype=int)
SKYSEG_KMOS = [0, 7800, 8240, 8730, 9260, 9640, 10140, 10670, 11250, 11960, 12450, \
        12890, 14000, 14720, 15650, 16000, 16356, 17253, 18400, 19570, 20950, 22000, 23500, 24600, 26000] 
SKYFLAG_KMOS = [0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0]

# Number of available CPUs
NCPU_default = cpu_count()
PY2 = sys.version_info[0] == 2

if not PY2:
    text_type = str
    string_types = (str,)
else:
    text_type = unicode
    string_types = (str, unicode)


###############################################################################
################################### Top Level Functions #######################
###############################################################################


def process(cubefits=None, cube=None, hdr=None, outcubefits='DATACUBE_ZAP.fits', clean=True,
            zlevel='median', cftype='weight', cfwidthSVD=100, cfwidthSP=50,
            pevals=[], nevals=[], optimizeType='normal', extSVD=None,
            skycubefits=None, svdoutputfits='ZAP_SVD.fits', mask=None,
            interactive=False, overwrite=False, instrument='MUSE', boost=50, 
            ncpu=NCPU_default, reject_factor=0.0, contmask=None, logger=None):
    """ Performs the entire ZAP sky subtraction algorithm.

    Work on an input FITS file and optionally writes the product to an output
    FITS file.

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
    cftype : str
        Method for the continuum filter: `median` or `weight` (default). For
        the `weight` method, a zeroth order sky is required (see `zlevel`).
    cfwidthSVD : int or float
        Window size for the continuum filter, for the SVD computation.
        Default to 100.
    cfwidthSP : int or float
        Window size for the continuum filter used to remove the continuum
        features for calculating the eigenvalues per spectrum. Smaller values
        better trace the sources. An optimal range of is typically
        20 - 50 pixels. Default to 50.
    optimizeType : str
        Optimization method to compute the number of eigenspectra used for each
        segment: `none`, `normal` (default), `enhanced`. If `none`, the number
        of eigenspectra must be specified with `nevals` or `pevals`, otherwise
        `normal` is used.
    pevals : list
        Allow to specify the percentage of eigenspectra used for each segment.
        If this is used, the pevals is ignored. Provide either a single value
        that will be used for all of the segments, or a list of 11 values that
        will be used for each of the segments.
    nevals : list
        Allow to specify the number of eigenspectra used for each segment.
        If this is used, the pevals is ignored. Provide either a single value
        that will be used for all of the segments, or a list of 11 values that
        will be used for each of the segments.
    extSVD : str
        Path of an input FITS file containing a SVD computed by the
        :func:`~zap.SVDoutput` function. Otherwise the SVD is computed.
    skycubefits : str
        Path for the optional output of the sky that is subtracted from the
        cube. This is simply the input cube minus the output cube.
    svdoutputfits : str
        Output FITS file containing the eigenbasis. Default to `ZAP_SVD.fits`.
    mask : str
        A 2D fits image to exclude regions that may contaminate the zlevel or
        eigenspectra. This image should be constructed from the datacube itself
        to match the dimensionality. Sky regions should be marked as 0, and
        astronomical sources should be identified with an integer greater than
        or equal to 1. Default to None.
    interactive : bool
        If True, a :class:`~zap.zclass` object containing all information on
        the ZAP process is returned, and can be used to explore the
        eigenspectra and recompute the output (with the
        :meth:`~zap.zclass.reprocess` method). In this case, the output files
        are not saved (`outcubefits` and `skycubefits` are ignored). Default to
        False.
    overwrite : bool
        If True (default False) write over any files.
    """
    #if not isinstance(cubefits, string_types):
    #    raise TypeError('The process method only accepts a single datacube '
    #                    'filename.')

    # make sure it has the right extension
    outcubefits = outcubefits.split('.fits')[0] + '.fits'

    # check if outcubefits/skycubefits exists before beginning
    check_file_exists(outcubefits, overwrite=overwrite)
    check_file_exists(skycubefits, overwrite=overwrite)
    if extSVD is None:
        check_file_exists(svdoutputfits, overwrite=overwrite)

    # Check for consistency between weighted median and zlevel keywords
    if cftype == 'weight' and zlevel == 'none':
        raise ValueError('Weighted median requires a zlevel calculation')

    if optimizeType not in ('none', 'normal', 'enhanced'):
        raise ValueError('Invalid value for optimizeType')

    if extSVD is not None and mask is not None:
        raise ValueError('extSVD and mask parameters are incompatible: if mask'
                         ' must be used, then the SVD has to be recomputed')

    if mask is not None or (extSVD is None and cfwidthSVD != cfwidthSP):
        # Compute the SVD separately, only if a mask is given, or if the
        # cfwidth values differ and extSVD is not given. Otherwise, the SVD
        # will be computed in the _run method, which allows to avoid running
        # twice the zlevel and continuumfilter steps.
        SVDoutput(cubefits=cubefits, svdoutputfits=svdoutputfits,
                  clean=clean, zlevel=zlevel, cftype=cftype,
                  cfwidth=cfwidthSVD, mask=mask, instrument=instrument, 
                  boost=boost, ncpu=ncpu)
        extSVD = svdoutputfits

    zobj = zclass(cubefits=cubefits, instrument=instrument, cube=cube, hdr=hdr,
                  logger=logger)
    zobj._run(clean=clean, zlevel=zlevel, cfwidth=cfwidthSP, cftype=cftype,
              pevals=pevals, nevals=nevals, optimizeType=optimizeType,
              extSVD=extSVD, boost=boost, ncpu=ncpu, 
              contmask=contmask, reject_factor=reject_factor)
    if interactive:
        # Return the zobj object without saving files
        return zobj

    if zobj.run_zlevel != 'extSVD' and svdoutputfits is not None:
        # Save SVD only if it was computed in _run, i.e. if an external SVD
        # was not given
        zobj.writeSVD(svdoutputfits=svdoutputfits, overwrite=overwrite)
    if skycubefits is not None:
        zobj.writeskycube(skycubefits=skycubefits)

    zobj.mergefits(outcubefits, overwrite=overwrite)


def SVDoutput(cubefits=None, cube=None, hdr=None, stack=None, svdoutputfits='ZAP_SVD.fits', clean=True,
              zlevel='median', cftype='weight', cfwidth=100, mask=None, maskfits=False, overwrite=False, 
              instrument='MUSE', boost=50, ncpu=NCPU_default, contmask=None, reject_factor=0.0, logger=None):
    """ Performs the SVD decomposition of a datacube.

    This allows to use the SVD for a different datacube.

    Parameters
    ----------

    cubefits : str
        Input FITS file, containing a cube with data in the first extension.
    svdoutputfits : str
        Output FITS file. Default to ZAP_SVD.fits
    clean : bool
        If True (default value), the NaN values are cleaned. Spaxels with more
        then 25% of NaN values are removed, the others are replaced with an
        interpolation from the neighbors.
    zlevel : str
        Method for the zeroth order sky removal: `none`, `sigclip` or `median`
        (default).
    cftype : str
        Method for the continuum filter: `median` or `weight` (default). For
        the `weight` method, a zeroth order sky is required (see `zlevel`).
    cfwidth : int or float
        Window size for the continuum filter, default to 300.
    mask : str
        Path of a FITS file containing a mask (1 for objects, 0 for sky).

    """
    if logger is not None:
        logger.info('Running ZAP %s !', __version__)
        logger.info('Processing %s to compute the SVD', cubefits)
    check_file_exists(svdoutputfits, overwrite=overwrite)

    # Check for consistency between weighted median and zlevel keywords
    if cftype == 'weight' and zlevel == 'none':
        raise ValueError('Weighted median requires a zlevel calculation')

    zobj = zclass(cubefits=cubefits, cube=cube, hdr=hdr, instrument=instrument)

    # clean up the nan values
    if clean:
        zobj._nanclean()

    # if mask is supplied, apply it
    if mask is not None:
        zobj._applymask(mask, maskfits=maskfits)

    # Extract the spectra that we will be working with
    if stack is None:
        zobj._extract()
    else:
        #direct stack input, needs to have been NaNs pre-cleaned
        zobj.stack = stack

    # remove the median along the spectral axis
    if zlevel.lower() != 'none':
        zobj._zlevel(calctype=zlevel, ncpu=ncpu)

    #mask used for rejection of spectral channels in continuum estimate (1 = bad)
    if contmask is None:
        contmask = np.zeros(zobj.stack.shape[0], dtype=int)
    elif contmask.shape[0] != zobj.stack.shape[0]:
        raise ValueError("Input spectra and continuum mask must have the same first dimension")

    # remove the continuum level - this is multiprocessed to speed it up
    zobj._continuumfilter(cftype=cftype, cfwidth=cfwidth, boost=boost, ncpu=ncpu, 
            contmask=contmask, reject_factor=reject_factor)

    # do the multiprocessed SVD calculation
    zobj._msvd(ncpu=ncpu)

    # write to file
    zobj.writeSVD(svdoutputfits=svdoutputfits, overwrite=overwrite)


def contsubfits(cubefits, contsubfn='CONTSUB_CUBE.fits', overwrite=False, cfwidth=100):
    """ A multiprocessed implementation of the continuum removal.

    This process distributes the data to many processes that then reassemble
    the data. Uses two filters, a small scale (less than the line spread
    function) uniform filter, and a large scale median filter to capture the
    structure of a variety of continuum shapes.

    """
    check_file_exists(contsubfn)
    hdu = fits.open(cubefits)
    data = hdu[1].data
    stack = data.reshape(data.shape[0], (data.shape[1] * data.shape[2]))
    contarray = _continuumfilter(stack, 'median', cfwidth=cfwidth)

    # remove continuum features
    stack -= contarray
    hdu[1].data = stack.reshape(data.shape[0], data.shape[1], data.shape[2])
    hdu.writeto(contsubfn,overwrite=overwrite)
    hdu.close()


def nancleanfits(cubefits, outfn='NANCLEAN_CUBE.fits', overwrite=False, rejectratio=0.25,
                 boxsz=1):
    """
    Detects NaN values in cube and removes them by replacing them with an
    interpolation of the nearest neighbors in the data cube. The positions in
    the cube are retained in nancube for later remasking.

    Parameters
    ----------

    cubefits : str
        Input FITS file, containing a cube with data in the first extension.
    outfn : str
        Output FITS file. Default to NANCLEAN_CUBE.fits
    rejectratio : float
        Defines a cutoff for the ratio of NAN to total pixels in a spaxel
         before the spaxel is avoided completely. Default to 0.25
    boxsz : int
        Defines the number of pixels around the offending NaN pixel.
        Default to 1, which looks for the 26 nearest neighbors which
        is a 3x3x3 cube.

    """
    check_file_exists(outfn)
    hdu = fits.open(cubefits)
    cleancube = _nanclean(hdu[1].data, rejectratio=rejectratio, boxsz=boxsz)
    hdu[1].data = cleancube[0]
    hdu.writeto(outfn,overwrite=overwrite)
    hdu.close()


def check_file_exists(filename, overwrite=False, logger=None):
    if filename is not None and os.path.exists(filename):
        if overwrite is False:
            raise IOError('Output file "{0}" exists'.format(filename))
        else:
            if logger is not None:
                logger.info('Output file "{0}" exists, rewriting'.format(filename))            


def timeit(func, logger=None):
    @wraps(func)
    def wrapped(*args, **kwargs):
        t0 = time()
        res = func(*args, **kwargs)
        if logger is not None:
            logger.info('%s - Time: %.2f sec.', func.__name__, time() - t0)
        return res
    return wrapped


###############################################################################
##################################### Process Steps ###########################
###############################################################################

class zclass(object):

    """ Main class to run each of the steps of ZAP.

    Attributes
    ----------

    cleancube : numpy.ndarray
        The final datacube after removing all of the residual features.
    contarray : numpy.ndarray
        A 2D array containing the subtracted continuum per spaxel.
    cube : numpy.ndarray
        The original cube with the zlevel subtraction performed per spaxel.
    especeval : list of (eigenspectra, eval)
        A list containing the full set of eigenspectra and eigenvalues
        generated by the SVD calculation that is used toy reconstruct the
        entire datacube.
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
    varlist : numpy.ndarray
        An array for each segment with the variance curve, calculated for the
        optimize method.
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
    subespeceval : list of (eigenspectra, eval)
        The subset of eigenvalues and eigenspectra used to reconstruct the sky
        residuals.
    variancearray : numpy.ndarray
        A list of length nsegments containing variances calculated per spaxel
        used for normalization
    y,x : numpy.ndarray
        The position in the cube of the spaxels that are in the 2d
        deconstructed stack
    zlsky : numpy.ndarray
        A 1d array containing the result of the zero level subtraction

    """

    def __init__(self, cubefits=None, cube=None, hdr=None, instrument='MUSE', logger=None):
        """ Initialization of the zclass.

        Pulls the datacube into the class and trims it based on the known
        optimal spectral range of MUSE.

        """
        self.logger = logger

        # If hdu cube + header is supplied
        if cube is not None:
            self.cube     = cube
            self.header   = hdr
            #self.cubefits = cubefits
        
        # If there is only a cubefits file
        else:
            hdu = fits.open(cubefits)
            self.cube     = hdu[1].data
            self.header   = hdu[1].header
            self.cubefits = cubefits
            hdu.close()

        # Workaround for floating points errors in wcs computation: if cunit is
        # specified, wcslib will convert in meters instead of angstroms, so we
        # remove cunit before creating the wcs object
        header = self.header.copy()
        unit = u.Unit(header.pop('CUNIT3'))
        self.wcs = WCS(header).sub([3])

        # Create Lambda axis
        wlaxis = np.arange(self.header['NAXIS3'])
        self.laxis = self.wcs.all_pix2world(wlaxis, 0)[0]
        if unit != u.angstrom:
            # Make sure lambda is in angstroms
            self.laxis = (self.laxis * unit).to(u.angstrom).value

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
        self.y = None
        self.x = None

        # Normalization Maps
        self.contarray = None
        self.variancearray = None
        self.normstack = None

        # identify the spectral range of the dataset
        laxmin = min(self.laxis)
        laxmax = max(self.laxis)
        
        if self.logger is not None:
            self.logger.debug('Minimum wavelength=%d, maximum wavelength=%d', laxmin, laxmax)

        # List of segmentation limits in the optical
        if instrument == 'MUSE':
            if self.logger is not None:
                self.logger.info('Setup for MUSE')
            skyseg = np.array(SKYSEG_MUSE)
            skyflag = np.array(SKYFLAG_MUSE)
        elif instrument == 'KMOS':
            if self.logger is not None:
                self.logger.info('Setup for KMOS')
            skyseg = np.array(SKYSEG_KMOS)
            skyflag = np.array(SKYFLAG_KMOS)
        else:
            raise ValueError('None or invalid instrument name given')

        crop_seg = (skyseg > laxmin) & (skyseg < laxmax)
        skyseg = skyseg[crop_seg]
        skyflag = skyflag[crop_seg[1:]]

        # segment limit in angstroms
        self.lranges = (np.vstack([np.append(laxmin - 10, skyseg),
                                   np.append(skyseg, laxmax + 10)])).T
        
        # flag for continuum estimation
        self.franges = [flag for ii, flag in enumerate(skyflag) if skyseg[ii]] 

        # segment limit in pixels
        laxis = self.laxis
        lranges = self.lranges

        pranges = []
        for i in range(len(lranges)):
            paxis = wlaxis[(laxis > lranges[i, 0]) & (laxis <= lranges[i, 1])]
            pranges.append((np.min(paxis), np.max(paxis) + 1))

        self.pranges = np.array(pranges)
    
        # eigenspace Subset
        self.especeval = []
        self.subespeceval = []

        # Reconstruction of sky features
        self.recon = None
        self.cleancube = None
        self.varlist = None  # container for variance curves

    @timeit
    def _run(self, clean=True, zlevel='median', cftype='weight',
             cfwidth=100, pevals=[], nevals=[], optimizeType='normal',
             extSVD=None, boost=50, ncpu=NCPU_default, 
             contmask=None, reject_factor=0.0):
        """ Perform all zclass to ZAP a datacube:

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
        if self.logger is not None:
            self.logger.info('Running ZAP %s !', __version__)

        self.optimizeType = optimizeType

        # clean up the nan values
        if clean:
            self._nanclean()

        # Extract the spectra that we will be working with
        self._extract()

        # remove the median along the spectral axis
        if extSVD is None:
            if zlevel.lower() != 'none':
                self._zlevel(calctype=zlevel)
        else:
            if zlevel.lower() != 'none':
                self._externalzlevel(extSVD)

        #mask used for rejection of spectral channels in continuum estimate (1 = bad)
        if contmask is None:
            contmask = np.zeros(self.stack.shape[0], dtype=int)
        elif contmask.shape[0] != self.stack.shape[0]:
            raise ValueError("Input spectra and continuum mask must have the same first dimension")

        # remove the continuum level - this is multiprocessed to speed it up
        self._continuumfilter(cftype=cftype, cfwidth=cfwidth, boost=boost, ncpu=ncpu, 
                contmask=contmask, reject_factor=reject_factor)


        # do the multiprocessed SVD calculation
        if extSVD is None:
            self._msvd(ncpu=ncpu)
        else:
            self._externalSVD(extSVD)

        # choose some fraction of eigenspectra or some finite number of
        # eigenspectra
        if optimizeType != 'none' or (nevals == [] and pevals == []):
            self.optimize()
            self.chooseevals(nevals=self.nevals)
        else:
            self.chooseevals(pevals=pevals, nevals=nevals)

        # reconstruct the sky residuals using the subset of eigenspace
        self.reconstruct()

        # stuff the new spectra back into the cube
        self.remold()

    # Clean up the nan value spaxels
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
        """
        Deconstruct the datacube into a 2d array, since spatial information is
        not required, and the linear algebra routines require 2d arrays.

        The operation rejects any spaxel with even a single NaN value, since
        this would cause the linear algebra routines to crash.

        Adds the x and y data of these positions into the zclass

        """
        if self.logger is not None:
            self.logger.debug('Extracting to 2D')
        # make a map of spaxels with NaNs
        badmap = (np.logical_not(np.isfinite(self.cube))).sum(axis=0)
        # get positions of those with no NaNs
        self.y, self.x = np.where(badmap == 0)
        # extract those positions into a 2d array
        self.stack = self.cube[:, self.y, self.x]
        if self.logger is not None:
            self.logger.debug('%d valid spaxels', len(self.x))

    def _externalzlevel(self, extSVD):
        """Remove the zero level from the extSVD file."""
        if self.logger is not None:
            self.logger.info('Using external zlevel')
        self.zlsky = fits.getdata(extSVD, 0)
        self.stack -= self.zlsky[:, np.newaxis]
        self.run_zlevel = 'extSVD'

    @timeit
    def _zlevel(self, calctype='median', ncpu=NCPU_default):
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

        if calctype != 'none':
            if self.logger is not None:
                self.logger.info('Subtracting Zero Level')

            zlstack = self.stack

            if calctype == 'median':
                if self.logger is not None:
                    self.logger.info('Median zlevel calculation')
                func = _imedian
            elif calctype == 'sigclip':
                if self.logger is not None:
                    self.logger.info('Iterative Sigma Clipping zlevel calculation')
                func = _isigclip

            self.zlsky = np.hstack(parallel_map(func, zlstack, ncpu, axis=0, logger=self.logger))
            self.stack -= self.zlsky[:, np.newaxis]
        else:
            if self.logger is not None:
                self.logger.info('Skipping zlevel subtraction')

    def _continuumfilter(self, cfwidth=100, cftype='weight', boost=50, ncpu=NCPU_default,
            contmask=None, reject_factor=0.0):
        """ A multiprocessed implementation of the continuum removal.

        This process distributes the data to many processes that then
        reassemble the data.  Uses two filters, a small scale (less than the
        line spread function) uniform filter, and a large scale median filter
        to capture the structure of a variety of continuum shapes.

        added to class
        contarray - the removed continuua
        normstack - "normalized" version of the stack with the continuua
            removed

        """
        if self.logger is not None:
            self.logger.info('Applying Continuum Filter, cfwidth=%d', cfwidth)
        if cftype not in ('weight', 'median', 'med_smooth', 'med_smooth_reject'):
            raise ValueError("cftype must be 'weight' or 'median', got {}"
                             .format(cftype))
        self._cftype = cftype
        self._cfwidth = cfwidth

        if cftype == 'median':
            weight = None
            boost = None
        elif cftype == 'weight':
            weight = np.abs(self.zlsky - (np.max(self.zlsky) + 1))
            boost = None
        elif cftype == 'med_smooth':
            weight = None
            boost = boost
        
        # get filtered continuum estimate
        self.contarray = _continuumfilter(self.stack, cftype, weight=weight, cfwidth=cfwidth, boost=boost, 
                                          ncpu=ncpu)

        #for reject_fractor > 0, interpolate continuum for pixels with < reject_factor*cfwidth "good" pixels
        if contmask is not None and reject_factor > 0.0:
            if contmask.shape[0] != self.stack.shape[0]:
                raise ValueError("Input spectra and continuum mask must have the same first dimension")
       
            #set fraction of window which needs to be "good"
            reject_threshold = self._cfwidth*reject_factor
        
            #filter the continuum mask
            pad_mask = np.pad(contmask, (cfwidth//2, cfwidth//2), 'constant', constant_values=0)
            fmask = np.sum(rolling_window(pad_mask, cfwidth), axis=1)

            #identify pixels with too few "good" continuum pixels
            flag_bad = (fmask >= reject_threshold)

            self.contarray[flag_bad,:] = interp1d(np.arange(contmask.shape[0])[~flag_bad], 
                    self.contarray[~flag_bad,:], axis=0, bounds_error=False, 
                    fill_value='extrapolate')(np.arange(contmask.shape[0])[flag_bad])

        #check against against segment flags
        if any(self.franges): 
            for sflag, spix in zip(self.franges,self.pranges):
                if sflag:
                    parray = np.vstack((self.contarray[spix[0],:],self.contarray[spix[1],:]))
                    self.contarray[spix[0]:spix[1],:] = interp1d(np.array(spix),parray, axis=0)(np.arange(spix[0],spix[1]))

        self.normstack = self.stack - self.contarray

    @timeit
    def _msvd(self, ncpu=NCPU_default):
        """ Multiprocessed singular value decomposition.

        First the normstack is normalized per segment per spaxel by the
        variance.  Takes the normalized, spectral segments and distributes them
        to the individual svd methods.

        """
        if self.logger is not None:
            self.logger.info('Calculating SVD')

        # normalize the variance in the segments
        nseg = len(self.pranges)
        self.variancearray = var = np.zeros((nseg, self.stack.shape[1]))

        for i in range(nseg):
            pmin, pmax = self.pranges[i]
            var[i, :] = np.maximum(np.var(self.normstack[pmin:pmax, :], axis=0), 1.0e-10)
            self.normstack[pmin:pmax, :] /= var[i, :]

        if self.logger is not None:
            self.logger.debug('Beginning SVD on %d segments', nseg)
        indices = [x[0] for x in self.pranges[1:]]
        
        self.especeval = parallel_map(_isvd, self.normstack, indices, axis=0, logger=self.logger)


    def chooseevals(self, nevals=[], pevals=[]):
        """ Choose the number of eigenspectra/evals to use for reconstruction.

        User supplies the number of eigen spectra to be used (neval) or the
        percentage of the eigenspectra that were calculated (peval) from each
        spectral segment to be used.

        The user can either provide a single value to be used for all segments,
        or provide an array that defines neval or peval per segment.

        """
        nranges = len(self.especeval)
        nevals = np.atleast_1d(nevals)
        pevals = np.atleast_1d(pevals)
        nespec = np.array([self.especeval[i][0].shape[1]
                           for i in range(nranges)])

        # deal with no selection
        if len(nevals) == 0 and len(pevals) == 0:
            if self.logger is not None:
                self.logger.info('Number of modes not selected')
            nevals = np.array([1])

        # deal with an input list
        if len(nevals) > 1:
            if len(nevals) != nranges:
                nevals = np.array([nevals[0]])
                if self.logger is not None:
                    self.logger.info('Chosen eigenspectra array does not correspond to '
                                'number of segments')
            else:
                if self.logger is not None:
                    self.logger.info('Choosing %s eigenspectra for segments', nevals)

        if len(pevals) > 1:
            if len(pevals) != nranges:
                pevals = np.array([pevals[0]])
                if self.logger is not None:
                    self.logger.info('Chosen eigenspectra array does not correspond to '
                               'number of segments')
            else:
                if self.logger is not None:
                    self.logger.info('Choosing %s%% of eigenspectra for segments',
                                pevals)
                nevals = (pevals * nespec / 100.).round().astype(int)

        # deal with single value entries
        if len(pevals) == 1:
            if self.logger is not None:
                self.logger.info('Choosing %s%% of eigenspectra for all segments',
                            pevals)
            nevals = (pevals * nespec / 100.).round().astype(int)
        elif len(nevals) == 1:
            if self.logger is not None:
                self.logger.info('Choosing %s eigenspectra for all segments', nevals)
            nevals = np.zeros(nranges, dtype=int) + nevals

        # take subset of the eigenspectra and put them in a list
        subespeceval = []
        for i in range(nranges):
            eigenspectra, evals = self.especeval[i]
            tevals = (evals[0:nevals[i], :]).copy()
            teigenspectra = (eigenspectra[:, 0:nevals[i]]).copy()
            subespeceval.append((teigenspectra, tevals))

        self.subespeceval = subespeceval
        self.nevals = nevals

    @timeit
    def reconstruct(self):
        """Reconstruct the residuals from a given set of eigenspectra and
        eigenvalues
        """

        if self.logger is not None:
            self.logger.info('Reconstructing Sky Residuals')
        nseg = len(self.especeval)
        rec = [(eig[:, :, np.newaxis] * ev[np.newaxis, :, :]).sum(axis=1)
               for eig, ev in self.subespeceval]

        # rescale to correct variance
        for i in range(nseg):
            rec[i] *= self.variancearray[i, :]
        self.recon = np.concatenate(rec)

    # stuff the stack back into a cube
    def remold(self):
        """ Subtracts the reconstructed residuals and places the cleaned
        spectra into the duplicated datacube.
        """
        if self.logger is not None:
            self.logger.info('Applying correction and reshaping data product')
        self.cleancube = self.cube.copy()
        self.cleancube[:, self.y, self.x] = self.stack - self.recon
        if self.run_clean:
            self.cleancube[self.nancube] = np.nan

    # redo the residual reconstruction with a different set of parameters
    def reprocess(self, pevals=[], nevals=[]):
        """
        A method that redoes the eigenvalue selection, reconstruction, and
        remolding of the data.
        """

        self.chooseevals(pevals=pevals, nevals=nevals)
        self.reconstruct()
        self.remold()

    @timeit
    def optimize(self):
        """ Function to optimize the number of components used to characterize
        the residuals.

        This function calculates the variance per segment with an increasing
        number of eigenspectra/eigenvalues. It then deterimines the point at
        which the second derivative of this variance curve reaches zero. When
        this occurs, the linear reduction in variance is attributable to the
        removal of astronomical features rather than emission line residuals.

        """
        if self.logger is not None:
            self.logger.info('Optimizing')

        normstack = self.stack - self.contarray
        nseg = len(self.especeval)
        self.nevals = np.zeros(nseg, dtype=int)
        indices = [x[0] for x in self.pranges[1:]]
        self.varlist = parallel_map(_ivarcurve, normstack, indices, axis=0,
                                    especeval=self.especeval,
                                    variancearray=self.variancearray, logger=self.logger)

        if self.optimizeType == 'enhanced':
            if self.logger is not None:
                self.logger.info('Enhanced Optimization')
        else:
            if self.logger is not None:
                self.logger.info('Normal Optimization')

        for i in range(nseg):
            # optimize
            varlist = self.varlist[i]
            deriv = varlist[1:] - varlist[:-1]
            deriv2 = deriv[1:] - deriv[:-1]
            noptpix = varlist.size

            if self.optimizeType != 'enhanced':
                # statistics on the derivatives
                ind = int(.5 * (noptpix - 2))
                mn1 = deriv[ind:].mean()
                std1 = deriv[ind:].std() * 2
                mn2 = deriv2[ind:].mean()
                std2 = deriv2[ind:].std() * 2
                # look for crossing points. When they get within 1 sigma of
                # mean in settled region.
                # pad by 1 for 1st deriv
                cross1 = np.append([False], deriv >= (mn1 - std1))
                # pad by 2 for 2nd
                cross2 = np.append([False, False],
                                   np.abs(deriv2) <= (mn2 + std2))
                cross = np.logical_or(cross1, cross2)
            else:
                # statistics on the derivatives
                ind = int(.75 * (noptpix - 2))
                mn1 = deriv[ind:].mean()
                std1 = deriv[ind:].std()
                mn2 = deriv2[ind:].mean()
                std2 = deriv2[ind:].std()
                # pad by 1 for 1st deriv
                cross = np.append([False], deriv >= (mn1 - std1))

            self.nevals[i] = np.where(cross)[0][0]

    # #########################################################################
    # #################################### Extra Functions ####################
    # #########################################################################

    def make_contcube(self):
        """ Remold the continuum array so it can be investigated.

        Takes the continuum stack and returns it into a familiar cube form.
        """
        contcube = self.cube.copy() * np.nan
        contcube[:, self.y, self.x] = self.contarray
        return contcube

    def _externalSVD(self, extSVD):
        if self.logger is not None:
            self.logger.info('Calculating eigenvalues for input eigenspectra')
        hdu = fits.open(extSVD)
        nseg = len(self.pranges)

        # normalize the variance in the segments
        self.variancearray = np.zeros((nseg, self.stack.shape[1]))

        for i in range(nseg):
            pmin, pmax = self.pranges[i]
            self.variancearray[i, :] = np.maximum(np.var(self.normstack[pmin:pmax, :],
                                              axis=0), 1e-10)
            self.normstack[pmin:pmax, :] /= self.variancearray[i, :]

        especeval = []
        for i in range(nseg):
            eigenspectra = hdu[i + 1].data
            ns = self.normstack[self.pranges[i][0]:self.pranges[i][1]]
            evals = np.transpose(np.transpose(ns).dot(eigenspectra))
            especeval.append([eigenspectra, evals])

        self.especeval = especeval
        hdu.close()

    def _applymask(self, mask, maskfits=True):
        """Apply a mask to the input data to provide a cleaner basis set.

        mask is >1 for objects, 0 for sky so that people can use sextractor.
        The file is read with ``astropy.io.fits.getdata`` which first tries to
        read the primary extension, then the first extension is no data was
        found before.

        If maskfits is False, input mask can be 2D array, not fits file

        """
        if self.logger is not None:
            self.logger.info('Applying Mask for SVD Calculation from %s', mask)
        if maskfits:
            self.maskfile = mask
            mask = fits.getdata(mask).astype(bool)
        nmasked = np.count_nonzero(mask)
        if self.logger is not None:
            self.logger.info('Masking %d pixels (%d%%)', nmasked,
                        nmasked / np.prod(mask.shape) * 100)
        self.cube[:, mask] = np.nan

    ###########################################################################
    ##################################### Output Functions ####################
    ###########################################################################

    def writecube(self, outcubefits='DATACUBE_ZAP.fits', overwrite=False):
        """Write the processed datacube to an individual fits file."""

        check_file_exists(outcubefits,overwrite=overwrite)
        # fix up for writing
        outhead = _newheader(self)

        # create hdu and write
        outhdu = fits.PrimaryHDU(data=self.cleancube, header=outhead)
        outhdu.writeto(outcubefits,overwrite=overwrite)
        if self.logger is not None:
            self.logger.info('Cube file saved to %s', outcubefits)

    def writeskycube(self, skycubefits='SKYCUBE_ZAP.fits', overwrite=False):
        """Write the processed datacube to an individual fits file."""

        check_file_exists(skycubefits,overwrite=overwrite)
        # fix up for writing
        outcube = self.cube - self.cleancube
        outhead = _newheader(self)

        # create hdu and write
        outhdu = fits.PrimaryHDU(data=outcube, header=outhead)
        outhdu.writeto(skycubefits,overwrite=overwrite)
        if self.logger is not None:
            self.logger.info('Sky cube file saved to %s', skycubefits)

    def mergefits(self, outcubefits, overwrite=False, extension=1):
        """Merge the ZAP cube into the full muse datacube and write."""

        # make sure it has the right extension
        outcubefits = outcubefits.split('.fits')[0] + '.fits'
        check_file_exists(outcubefits,overwrite=overwrite)
        hdu = fits.open(self.cubefits)
        hdu[extension].header = _newheader(self)
        hdu[extension].data = self.cleancube
        hdu.writeto(outcubefits,overwrite=overwrite)
        hdu.close()
        if self.logger is not None:
            self.logger.info('Cube file saved to %s', outcubefits)

    def writeSVD(self, svdoutputfits='ZAP_SVD.fits', overwrite=False):
        """Write the SVD to an individual fits file."""

        check_file_exists(svdoutputfits,overwrite=overwrite)
        header = fits.Header()
        header['ZAPvers'] = (__version__, 'ZAP version')
        header['ZAPzlvl'] = (self.run_zlevel, 'ZAP zero level correction')
        header['ZAPclean'] = (self.run_clean,
                              'ZAP NaN cleaning performed for calculation')
        header['ZAPcftyp'] = (self._cftype, 'ZAP continuum filter type')
        header['ZAPcfwid'] = (self._cfwidth, 'ZAP continuum filter size')
        header['ZAPmask'] = (self.maskfile, 'ZAP mask used to remove sources')
        nseg = len(self.pranges)
        header['ZAPnseg'] = (nseg, 'Number of segments used for ZAP SVD')

        hdu = fits.HDUList([fits.PrimaryHDU(self.zlsky)])
        for i in range(len(self.pranges)):
            hdu.append(fits.ImageHDU(self.especeval[i][0]))
        # write for later use
        hdu.writeto(svdoutputfits,overwrite=overwrite)
        if self.logger is not None:
            self.logger.info('SVD file saved to %s', svdoutputfits)

    def plotvarcurve(self, i=0, ax=None):
        if len(self.varlist) == 0:
            if self.logger is not None:
                self.logger.warning('No varlist found. The optimize method must be '
                                   'run first.')
            return

        # optimize
        deriv = (np.roll(self.varlist[i], -1) - self.varlist[i])[:-1]
        deriv2 = (np.roll(deriv, -1) - deriv)[:-1]
        noptpix = self.varlist[i].size

        if self.optimizeType == 'normal':
            # statistics on the derivatives
            mn1 = deriv[.5 * (noptpix - 2):].mean()
            std1 = deriv[.5 * (noptpix - 2):].std() * 2
            mn2 = deriv2[.5 * (noptpix - 2):].mean()
            std2 = deriv2[.5 * (noptpix - 2):].std() * 2
        else:
            # statistics on the derivatives
            mn1 = deriv[.75 * (noptpix - 2):].mean()
            std1 = deriv[.75 * (noptpix - 2):].std()
            mn2 = deriv2[.75 * (noptpix - 2):].mean()
            std2 = deriv2[.75 * (noptpix - 2):].std()

        if ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(3, 1, figsize=[10, 15])

        ax1, ax2, ax3 = ax
        ax1.plot(self.varlist[i], linewidth=3)
        ax1.plot([self.nevals[i], self.nevals[i]],
                 [min(self.varlist[i]), max(self.varlist[i])])
        ax1.set_ylabel('Variance')

        ax2.plot(np.arange(deriv.size), deriv)
        ax2.plot([0, len(deriv)], [mn1, mn1], 'k')
        ax2.plot([0, len(deriv)], [mn1 - std1, mn1 - std1], '0.5')
        ax2.plot([self.nevals[i] - 1, self.nevals[i] - 1],
                 [min(deriv), max(deriv)])
        ax2.set_ylabel('d/dn Var')

        ax3.plot(np.arange(deriv2.size), np.abs(deriv2))
        ax3.plot([0, len(deriv2)], [mn2, mn2], 'k')
        ax3.plot([0, len(deriv2)], [mn2 + std2, mn2 + std2], '0.5')
        ax3.plot([self.nevals[i] - 2, self.nevals[i] - 2],
                 [min(deriv2), max(deriv2)])
        ax3.set_ylabel('(d^2/dn^2) Var')
        # ax3.set_xlabel('Number of Components')

        ax1.set_title('Segment {0}, {1} - {2} Angstroms'.format(
            i, self.lranges[i][0], self.lranges[i][1]))


###############################################################################
##################################### Helper Functions ########################
###############################################################################


def worker(f, i, chunk, out_q, err_q, kwargs):
    try:
        result = f(i, chunk, **kwargs)
    except Exception as e:
        err_q.put(e)
        return

    # output the result and task ID to output queue
    out_q.put((i, result))


def parallel_map(func, arr, indices, **kwargs):
    manager = Manager()
    out_q = manager.Queue()
    err_q = manager.Queue()
    jobs = []
    axis = kwargs.pop('axis', None)
    chunks = np.array_split(arr, indices, axis=axis)

    for i, chunk in enumerate(chunks):
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


##### Continuum Filtering #####

@timeit
def _continuumfilter(stack, cftype, weight=None, cfwidth=300, boost=None, ncpu=NCPU_default):
    if cftype == 'median':
        func = _icfmedian
        weight = None
        boost = None
    elif cftype == 'weight':
        func = _icfweight
        boost = None
    elif cftype == 'med_smooth':
        func = _icfmed_smooth
        weight = None

    c = parallel_map(func, stack, ncpu, axis=1, weight=weight, cfwidth=cfwidth, 
            boost=boost)
    return np.concatenate(c, axis=1)


def _icfweight(i, stack, weight=None, cfwidth=None):
    return np.array([wmedian(row, weight, cfwidth=cfwidth)
                     for row in stack.T]).T

def _icfmedian(i, stack, weight=None, cfwidth=None):
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


def wmedian(spec, wt, cfwidth=100):
    """ Performs a weighted median filtering of a 1d spectrum

    Operates using a cumulative sum curve

    Parameters
    ----------

    spec : numpy.ndarray
        Input 1d spectrum to be filtered
    wt : numpy.ndarray
        A spectrum of equal length as the input array to provide the weights.
    cfwidth : int or float
        Window size for the continuum filter, for the SVD computation.
        Default to 100.

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


# ### SVD #####

def _isvd(i, normstack, logger=None):
    """
    Perform single value decomposition and Calculate PC amplitudes (projection)
    outputs are eigenspectra operates on a 2D array.

    eigenspectra = [nbins, naxes]
    evals = [naxes, nobj]
    data = [nbins, nobj]
    """

    inormstack = normstack.T

    #center the data
    _mean = np.mean(inormstack, axis=0)
    inormstack -= _mean

    U, s, V = linalg.svd(inormstack, full_matrices=False)
    U, V = svd_flip(U, V)

    eigenspectra = np.transpose(V)
    evals = inormstack.dot(eigenspectra)
    if logger is not None:
        logger.info('Finished SVD Segment %d', i)
    return [eigenspectra, evals.T]


# ### OPTIMIZE #####

def _ivarcurve(i, istack, especeval=None, variancearray=None, logger=None):
    """
    Reconstruct the residuals from a given set of eigenspectra and eigenvalues.

    this is a special version for caculating the variance curve. It adds the
    contribution of a single mode to an existing reconstruction.

    """

    iprecon = np.zeros_like(istack)
    eigenspectra, evals = especeval[i]
    variance = variancearray[i]
    ivarlist = []
    totalnevals = int(np.round(evals.shape[0] * 0.25))

    progress_step = int(totalnevals * .2)
    to_percent = 100. / (totalnevals - 1.)
    if logger is not None:
        info = logger.info

    for nevals in range(totalnevals):
        if nevals and (nevals % progress_step) == 0 and logger is not None:
            info('Seg %d: %d%% complete ', i, int(nevals * to_percent))

        eig = eigenspectra[:, nevals]
        ev = evals[nevals, :]
        # broadcast evals on evects and sum
        iprecon += (eig[:, np.newaxis] * ev[np.newaxis, :])

        icleanstack = istack - (iprecon * variance)
        # calculate the variance on the cleaned segment
        ivarlist.append(np.var(icleanstack))

    return np.array(ivarlist)


def _newheader(zobj):
    """Put the pertinent zap parameters into the header"""
    header = zobj.header.copy()
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
    for i in range(nseg):
        header['ZAPseg{0}'.format(i)] = (
            '{0}:{1}'.format(zobj.pranges[i][0], zobj.pranges[i][1] - 1),
            'spectrum segment (pixels)')
        header['ZAPnev{0}'.format(i)] = (zobj.nevals[i],
                                         'number of eigenvals/spectra used')

    return header


#def _isigclip(i, istack):
#    mn = []
#    for col in istack:
#        clipped, bot, top = sigmaclip(col, low=3, high=3)
#        print( clipped.mean())
#        mn.append(clipped.mean())
#    print( mn)
#    return np.array(mn)

def _isigclip(i, istack):
    mn = []
    mfunc = np.ma.median
    sfunc = lambda x: 1.4826*np.ma.median(np.fabs(x-np.ma.median(x)))
    for col in istack:
        clipped = sigma_clip(col, sigma_lower=3, sigma_upper=3, cenfunc=mfunc, stdfunc=sfunc)
        mn.append(np.ma.mean(clipped))
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
    # badcube  = np.logical_not(np.isfinite(cleancube))        # find NaNs
    badcube  = np.logical_or(np.isnan(cleancube), cleancube == 0.) # Cut NaNs and unphysical zero flux
    badmap   = badcube.sum(axis=0)                           # map of total nans in a spaxel

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

    mn = np.ma.masked_invalid(neighbor)
    cleancube[z, y, x] = 0.
    return cleancube, badcube


