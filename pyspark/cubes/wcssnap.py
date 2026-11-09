"""[cubes] snap_wcs = False | True : integer-pixel registration for the cube combination (docs/noise-improvements-2026-10.md).

kmos_combine (EDPS cubes_combination, method=header) aligns the frames of a target by their WCS and applies
sub-pixel spatial shifts with a hard-coded bicubic spline ('BCS', kmclipm_shift). The O frames are dithered
by half a spaxel, so 4 of 6 frames are interpolated; the spline spreads single-frame outliers (OH residual
spikes, noisy pixels) into many voxels below the k-sigma rejection threshold. With integer shifts only,
line-flux S/N of 1"x1" apertures was +5.0 % (IZ), +4.7 % (YJ), +2.1 % (H) in esorex tests, at the cost of
a 1.4-5.6 % larger PSF FWHM (registration error up to half a spaxel = 0.1").

The snap is header-only: in every IFU extension CRPIX1/2 are moved by at most half a spaxel so that the IFU's
target position (ESO OCS ARMi ALPHA/DELTA, identical in all frames of a target) falls on the grid phase of
the reconstruction (CRPIX modulo 1, the same in all frames); undithered frames are not changed. The data
are not touched; the original values are kept in HIERARCH PYSPARK CRPIX1/2 ORIG.
"""
from __future__ import print_function

import glob
import os

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

__all__ = ["snap_frame", "snap_wcs"]

SNAPPED = "HIERARCH ESO PYSPARK WCS SNAPPED"


def _hms2deg(v):
    s = "%010.3f" % abs(float(v))
    return (int(s[0:2]) + int(s[2:4]) / 60.0 + float(s[4:]) / 3600.0) * 15.0


def _dms2deg(v):
    sign = -1.0 if float(v) < 0 else 1.0
    s = "%010.3f" % abs(float(v))
    return sign * (int(s[0:2]) + int(s[2:4]) / 60.0 + float(s[4:]) / 3600.0)


def snap_frame(path):
    """Snap the IFU WCS of one frame in place (header only). Returns [(ifu, dx, dy), ...] in pixels;
    a frame already snapped is left alone (returns None)."""
    with fits.open(path, mode="update") as hl:
        h0 = hl[0].header
        if h0.get(SNAPPED, False):
            return None
        moved = []
        for ifu in range(1, min(25, len(hl))):
            if hl[ifu].data is None:
                continue
            ra = h0.get("HIERARCH ESO OCS ARM%d ALPHA" % ifu)
            dec = h0.get("HIERARCH ESO OCS ARM%d DELTA" % ifu)
            if ra is None or dec is None:
                continue
            h = hl[ifu].header
            px, py = WCS(h).celestial.world_to_pixel_values(_hms2deg(ra), _dms2deg(dec))
            pos = np.array([float(px), float(py)]) + 1.0                   # FITS (1-based) pixel
            phase = np.array([h["CRPIX1"] % 1.0, h["CRPIX2"] % 1.0])
            d = (np.round(pos - phase) + phase) - pos
            h["HIERARCH PYSPARK CRPIX1 ORIG"] = h["CRPIX1"]
            h["HIERARCH PYSPARK CRPIX2 ORIG"] = h["CRPIX2"]
            h["CRPIX1"] = h["CRPIX1"] + d[0]
            h["CRPIX2"] = h["CRPIX2"] + d[1]
            moved.append((ifu, round(float(d[0]), 3), round(float(d[1]), 3)))
        h0[SNAPPED] = (True, "IFU WCS snapped to integer registration (wcssnap)")
        h0["HISTORY"] = "pyspark snap_wcs: CRPIX1/2 moved by <= 0.5 px for integer-shift combination"
    return moved


def snap_wcs(logger=None, subdir=None, obsband=None, snap_wcs=False, search_suffix='s_i_bkg2d_rv', **kwargs):
    """reduce_kmos step after the velocity correction: snap the final frames of one OB and band."""
    if not snap_wcs:
        return 0
    sci = os.path.join(kwargs['outputdir'], subdir, 'science')
    n = 0
    for f in sorted(glob.glob(os.path.join(sci, 'SCI_RECONSTRUCTED_*_{0}.fits'.format(search_suffix)))):
        if str(fits.getheader(f).get('HIERARCH ESO INS GRAT1 ID', '')).strip() != obsband:
            continue
        moved = snap_frame(f)
        n += 1
        if logger is not None and moved:
            logger.debug('-- snap_wcs {0}: max shift {1:.2f} px'.format(
                os.path.basename(f), max(max(abs(dx), abs(dy)) for _, dx, dy in moved)))
    if logger is not None:
        logger.info('- snap_wcs: {0} frames'.format(n))
    return 0 if n else 1
