"""Row-wise common-mode correction of KMOS science frames (row-correlated H2RG readout noise).

After the reference-column correction of fix_detector_levels (channel_correct) every frame still carries a
per-row offset common to all readout channels: 0.25-0.43 x the pixel noise in IZ/YJ, 0.19-0.27 in H,
independent between frames and between detectors. A detector row is one wavelength across all IFUs of the
detector, so in the cubes it becomes an offset coherent along the slitlets that raises the noise of aperture
spectra. Measured on the worst IZ OB: 5-6 % lower noise of 6-frame-stacked 5x5 aperture spectra (6-11 % for
IFUs on detectors 2 and 3), none on detector 1 and none in H (docs/noise-improvements-2026-10.md).

Per detector, the OB-median frame (static pattern, sky, targets) is subtracted from every frame; c(y) is the
median of that residual over the unilluminated pixels of row y (no slitlet in any rotator-angle extension of
XCAL, dilated by `grow` pixels along x, reference pixels excluded), and c(y) is subtracted from the whole row.
Unilluminated pixels carry no target signal. Detector 1 has only 40-75 such pixels per row (two edge
channels), which is why the default leaves it alone.

The corrected frames are written as COPIES with a new prefix (e.g. dcr_KMOS.*.fits next to dc_KMOS.*.fits);
the input frames are never modified (they may be links into another PROCDIR).
"""
from __future__ import print_function

import glob
import os

import numpy as np
from astropy.io import fits
from scipy.ndimage import binary_dilation

__all__ = ["unillum_mask", "row_offsets", "fix_row_commonmode", "row_commonmode_ob"]

ROWS = slice(4, 2044)          # rows 0-3 and 2044-2047 are reference pixels


def unillum_mask(xcal_file, det, grow=1):
    """Pixels of detector `det` that no slitlet illuminates at any rotator angle (NaN in every DET.<det>.DATA
    extension of XCAL), dilated by `grow` pixels along x, reference pixels excluded."""
    ill = np.zeros((2048, 2048), bool)
    with fits.open(xcal_file) as xcal:
        for ext in xcal[1:]:
            if ext.name == 'DET.{0}.DATA'.format(det) and ext.data is not None:
                ill |= np.isfinite(ext.data)
    if grow:
        ill = binary_dilation(ill, structure=np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], bool),
                              iterations=int(grow))
    un = ~ill
    un[:4] = un[-4:] = False
    un[:, :4] = un[:, -4:] = False
    return un


def row_offsets(stack, mask, min_pix=150):
    """c(y) for every frame of `stack` (nframe, 2048, 2048): median over the `mask` pixels of row y of
    (frame - per-pixel median of all frames); 0 for rows with fewer than `min_pix` valid pixels."""
    med = np.nanmedian(stack, axis=0)
    use = mask & np.isfinite(med)
    out = np.zeros(stack.shape[:2], np.float32)
    for i in range(stack.shape[0]):
        r = np.where(use, stack[i] - med, np.nan)[ROWS]
        n = np.isfinite(r).sum(axis=1)
        with np.errstate(all='ignore'):
            c = np.nanmedian(np.where(n[:, None] > 0, r, 0.0), axis=1)
        out[i, ROWS] = np.where((n >= max(int(min_pix), 1)) & np.isfinite(c), c, 0.0)
    return out


def fix_row_commonmode(frames, xcal_file, out_prefix, in_prefix='dc', dets=(2, 3), grow=1, min_pix=150,
                       logger=None):
    """Write row-corrected copies of `frames` ('<dir>/<in_prefix>_KMOS.*.fits', all of one OB and band) as
    '<dir>/<out_prefix>_KMOS.*.fits'. Returns the output paths. With fewer than 3 frames the copies are
    written unchanged (no median template), with a HISTORY note."""
    if out_prefix == in_prefix:
        raise ValueError('row common-mode correction: output prefix must differ from input prefix')
    frames = sorted(frames)
    outs = []
    for f in frames:
        d, name = os.path.split(f)
        if not name.startswith(in_prefix + '_'):
            raise ValueError('{0} does not start with {1}_'.format(name, in_prefix))
        outs.append(os.path.join(d, out_prefix + name[len(in_prefix):]))
    corr = {}
    if len(frames) >= 3:
        for det in dets:
            mask = unillum_mask(xcal_file, det, grow=grow)
            stack = np.array([fits.getdata(f, det).astype(np.float32) for f in frames])
            corr[det] = row_offsets(stack, mask, min_pix=min_pix)
            del stack
            if logger is not None:
                logger.info('-- row common mode det{0}: {1} unilluminated px/row (median), correction rms '
                            '{2:.3f}-{3:.3f} ADU'.format(det, int(np.median(mask[ROWS].sum(1))),
                                                         float(np.min(np.std(corr[det], 1))),
                                                         float(np.max(np.std(corr[det], 1)))))
    elif logger is not None:
        logger.warning('-- row common mode: only {0} frame(s), copies written uncorrected'.format(len(frames)))
    for i, (f, out) in enumerate(zip(frames, outs)):
        with fits.open(f) as hl:
            for det in dets:
                if det in corr:
                    hl[det].data = (hl[det].data.astype(np.float32) - corr[det][i][:, None]).astype(np.float32)
                    hl[0].header['HIERARCH ESO PYSPARK ROWCM DET{0} RMS'.format(det)] = (
                        round(float(np.std(corr[det][i])), 5), '[ADU] rms of the row correction')
            if corr:
                hl[0].header['HISTORY'] = 'pyspark row common-mode correction, dets {0}, grow {1}'.format(
                    ','.join(str(x) for x in dets), grow)
            else:
                hl[0].header['HISTORY'] = 'pyspark row common-mode correction skipped (< 3 frames)'
            hl.writeto(out, overwrite=True)
    return outs


def _as_dets(value):
    if isinstance(value, str):
        return tuple(int(x) for x in value.replace(' ', '').split(',') if x)
    return tuple(int(x) for x in value)


def row_commonmode_ob(outputdir=None, subdir=None, obsband=None, prefix=None, row_commonmode_input='dc',
                      row_commonmode_dets='2,3', row_commonmode_grow=1, row_commonmode_minpix=150,
                      logger=None, **kwargs):
    """reduce_kmos step: correct the science frames '<input>_KMOS.*.fits' of one OB and band in
    <outputdir>/<subdir> into '<prefix>_KMOS.*.fits' (prefix = the [detector] prefix that the
    reconstruction reads). Needs <outputdir>/<subdir>/calib/XCAL_<band x3>.fits."""
    if not prefix or prefix == row_commonmode_input:
        raise ValueError('row_commonmode_correct needs a [detector] prefix (read by the reconstruction) '
                         'different from row_commonmode_input, got {0!r}'.format(prefix))
    obd = os.path.join(outputdir, subdir)
    xcal = os.path.join(obd, 'calib', 'XCAL_{0}{0}{0}.fits'.format(obsband.upper()))
    if not os.path.isfile(xcal):
        raise IOError('row common-mode correction needs {0}'.format(xcal))
    frames = []
    for f in sorted(glob.glob(os.path.join(obd, '{0}_KMOS.*.fits'.format(row_commonmode_input)))):
        h = fits.getheader(f)
        band = str(h.get('HIERARCH ESO INS GRAT1 ID', '')).strip()
        catg = str(h.get('HIERARCH ESO DPR CATG', '')).strip()
        if band == obsband and catg == 'SCIENCE':
            frames.append(f)
    if not frames:
        raise IOError('no {0}_KMOS science frames for band {1} in {2}'.format(row_commonmode_input, obsband, obd))
    if logger is not None:
        logger.info('- Row common-mode correction of {0} frames: {1}_ -> {2}_'.format(
            len(frames), row_commonmode_input, prefix))
    return fix_row_commonmode(frames, xcal, out_prefix=prefix, in_prefix=row_commonmode_input,
                              dets=_as_dets(row_commonmode_dets), grow=int(row_commonmode_grow),
                              min_pix=int(row_commonmode_minpix), logger=logger)
