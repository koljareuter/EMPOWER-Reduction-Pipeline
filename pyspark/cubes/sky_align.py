"""[cubes] sky_align = False | True : align each SKY frame to its OBJECT frame at the slitlet ends
(docs/hybrid-sky.md, 'Sky lines at the slitlet ends'). Needs sky_cal_angle = object.

Cause. The slitlet traces move rigidly along detector x (= along the slitlet) with the rotator angle, by up to
~1.5 px over 360 deg (flexure). The flat and the slitlet geometry (XCAL/YCAL) of one 60-deg calibration bin then
put the slitlet end of a frame taken at another angle slightly off, and O - S keeps a scaled copy of every sky
line in the 2-3 cube rows/columns at the slitlet ends (IFUs 1-16: cube rows 0/13, IFUs 17-24: columns 0/13).
sky_cal_angle = object makes it worse (the SKY frame, taken up to ~90 deg away, is reconstructed with the
OBJECT's geometry although its traces sit elsewhere).

Fix (prototype 'shs4', haln/align2, 2026-11-04). Per OBJECT frame O and its SKY frame S (pyspark's pairing:
nearest DATE-OBS, same DIT), without moving S's pixels (a raw shift of S moves the detector's fixed pattern
off the pixels it cancels on in O - S: interior 5x5 noise x1.16-1.31):
  dx_IFU  shift along detector x that puts S's traces onto O's, per IFU (sky_align_dx below)
  D       = shift(S_rep, dx_IFU) - S_rep  (S_rep: S with the flat's bad pixels repaired, cubic B-spline)
  per slitlet end, the pixels 0-2 inside it, bins of 40 mas of the calibrated along-slitlet coordinate
  (YCAL on detectors 1/2, XCAL on detector 3; calibration bin nearest to O's angle), every slitlet row:
          D = c0 + c1*spec + c2*spec' + c3*spec''   (spec = median of S_rep over the interior pixels 3-7 of the
                                                     same row and end, ' = derivative along the dispersion)
  S'      = S_raw + c1*spec + c2*spec' + c3*spec''  (the pedestal c0 is not applied; no other pixel changes)
S' is reconstructed with O's rotator angle (one SKY cube per OBJECT frame) into <OB>/sky_align/
SCI_RECONSTRUCTED_<OBJECT frame>, and subtract_sky uses it instead of the mean-angle SKY cube of sky/ for the
classic O - S + ZAP. The OBJECT frames and sky/ are not touched.

sky_align_dx = measure (default) | flat :
  flat     the slitlet-end positions of the OB's MASTER_FLAT at the six calibration angles (per IFU and end,
           median over rows and slitlets), a 2nd-order harmonic in angle, evaluated at the mean exposure angle
           (ADA ABSROT START/END) of O and of S, dx = mean over the two ends of e(O) - e(S)
  measure  the flat value refined from the frames' own OH lines: the dx (searched within +-0.6 px of the flat
           value) that nulls the slitlet-end/interior ratio of O/S at OH-line peaks (end pixels 0 and 1), per
           end; used when the search does not end at its border, the two ends agree within 0.2 px and the
           result is within 0.4 px of the flat value, otherwise the flat value
The offsets used are recorded in the SKY cube headers (HIERARCH ESO PYSPARK SKY ALIGN DX<ifu>; DXF<ifu> = flat
value, SRC<ifu> = MEAS | FLAT).
Measured on the local H OB KMOS.2026-06-07T07:24:38.627 (6 O-S pairs, S 76-89 deg from O, 6-frame stacks):
oh_rows median edge maximum 15.4 sigma (own angle) / 27.7 (object angle) -> 1.3; edge level of the +-3-plane line
windows 24.3 / 75.4 -> 4.5; interior noise and star fluxes as object angle.
Measured vs flat offsets there: rms 0.086 px. Cost: one extra kmos_sci_red run (one SKY
reconstruction per OBJECT frame) plus ~1 min of Python per O-S pair.
"""
from __future__ import print_function

import glob
import os

import numpy as np
from astropy.io import fits
from scipy.ndimage import shift as ndshift, median_filter, maximum_filter

from ..utils.fileutils import patch_header
from ..utils.obsutils import run_spark
from ..utils.setuputils import id_gen
from .sky_calangle import _info, ROT, ORIG, USED

__all__ = ["AlignCalib", "align_sky_data", "align_skies", "ALIGN_SKY", "ALIGN_OBJ", "ALIGN_DX"]

CAL_ROT = "HIERARCH ESO PRO ROT NAANGLE"
ABSROT = ("HIERARCH ESO ADA ABSROT START", "HIERARCH ESO ADA ABSROT END")
CAL_ANGLES = np.array([0., 60., 120., 180., 240., 300.])
PFX = "HIERARCH ESO PYSPARK SKY ALIGN "
ALIGN_SKY = PFX + "SKY"
ALIGN_OBJ = PFX + "OBJ"
ALIGN_DX = PFX + "DX"
HDR_KEYS = ['DATE', 'DATE-OBS', 'OBJECT', 'RA', 'DEC', 'MJD-OBS', 'UTC', 'LST']

EDGE_PIX = (0, 1, 2)          # pixels inside each slitlet end that are corrected
SPEC_PIX = range(3, 8)        # interior pixels of the same row giving the local sky spectrum
BIN_MAS = 40.0                # bin width of the along-slitlet coordinate for the fit
REF_ANGLE = 60.0              # calibration angle of the fixed IFU column blocks
DX_HALF, DX_STEP = 0.6, 0.1   # 'measure': search window around the flat value
DX_LR, DX_FLAT = 0.2, 0.4     # 'measure': max |dL - dR| and max |dx - flat value|


def _angle_bin(angle):
    return float(CAL_ANGLES[np.argmin(np.abs((CAL_ANGLES - angle + 180.) % 360. - 180.))])


def _harmonic(theta):
    t = np.radians(np.atleast_1d(np.asarray(theta, float)))
    return np.c_[np.ones_like(t), np.cos(t), np.sin(t), np.cos(2 * t), np.sin(2 * t)]


def _exposure_angle(path):
    h = fits.getheader(path)
    try:
        return 0.5 * (float(h[ABSROT[0]]) + float(h[ABSROT[1]]))
    except KeyError:
        return float(h[ROT])


class AlignCalib(object):
    """XCAL, YCAL and MASTER_FLAT of one OB and band at the six calibration angles, with the derived
    slitlet geometry, bad-pixel maps and the flat-edge offset model."""

    def __init__(self, caldir, obsband):
        b = obsband.upper()
        self._hdus = dict((k, fits.open(os.path.join(caldir, '{0}_{1}{1}{1}.fits'.format(k, b))))
                          for k in ('XCAL', 'YCAL', 'MASTER_FLAT'))
        self._geo, self._along, self._bad, self._dig = {}, {}, {}, {}
        self._edge_coef = None

    def close(self):
        for h in self._hdus.values():
            h.close()

    def cal(self, kind, det, angle):
        for x in self._hdus[kind][1:]:
            if x.header.get('EXTNAME') == 'DET.{0}.DATA'.format(det) and abs(float(x.header[CAL_ROT]) - angle) < .01:
                return np.array(x.data, float)
        raise KeyError('{0} has no DET.{1}.DATA at {2} deg'.format(kind, det, angle))

    def ifu_digit(self, det):
        """IFU number within the detector (1-8) per pixel, from the XCAL decimal digit at REF_ANGLE."""
        if det not in self._dig:
            xc = self.cal('XCAL', det, REF_ANGLE)
            with np.errstate(invalid='ignore'):
                self._dig[det] = np.rint(10 * np.abs(xc - np.trunc(xc)))
        return self._dig[det]

    def bad_pixels(self, det):
        """(along x, along y) masks of the isolated flat-NaN pixels (any angle) that repair() interpolates."""
        if det not in self._bad:
            g = np.array([np.isfinite(np.asarray(x.data, float)) for x in self._hdus['MASTER_FLAT'][1:]
                          if x.header.get('EXTNAME') == 'DET.{0}.DATA'.format(det)])
            bx = np.zeros(g.shape[1:], bool)
            by = np.zeros(g.shape[1:], bool)
            for G in g:
                L, R, L2, R2 = np.roll(G, 1, 1), np.roll(G, -1, 1), np.roll(G, 2, 1), np.roll(G, -2, 1)
                bx |= (~G & L & R) | (~G & ((L & ~R & R2) | (R & ~L & L2)))
                by |= ~G & ~L & ~R & np.roll(G, 1, 0) & np.roll(G, -1, 0)
            self._bad[det] = (bx, by & ~bx)
        return self._bad[det]

    def repair(self, D, det):
        """Copy of a raw detector image with the flat's bad pixels interpolated (runs of <= 2 px along x from
        the flanking pixels, otherwise the mean of the rows above and below); remaining NaN: x-neighbour mean."""
        bx, by = self.bad_pixels(det)
        out = D.copy()
        good = ~(bx | by) & np.isfinite(D)
        n = D.shape[1]
        for y, x in zip(*np.nonzero(bx)):
            l = x - 1
            while l >= 0 and not good[y, l] and x - l < 3:
                l -= 1
            r = x + 1
            while r < n and not good[y, r] and r - x < 3:
                r += 1
            if 0 <= l and r < n and good[y, l] and good[y, r]:
                out[y, x] = D[y, l] + (D[y, r] - D[y, l]) * (x - l) / float(r - l)
        for y, x in zip(*np.nonzero(by)):
            if 0 < y < D.shape[0] - 1:
                out[y, x] = 0.5 * (D[y - 1, x] + D[y + 1, x])
        nb = ~np.isfinite(out)
        if nb.any():
            out[nb] = np.nan_to_num(0.5 * (np.roll(out, 1, 1) + np.roll(out, -1, 1))[nb])
        return out

    def geometry(self, det, angle):
        """Slitlet runs per detector row at a calibration angle: label (IFU digit * 100 + slitlet), row, first
        and last column."""
        if (det, angle) not in self._geo:
            xc, yc = self.cal('XCAL', det, angle), self.cal('YCAL', det, angle)
            fin = np.isfinite(xc) & np.isfinite(yc)
            xz, yz = np.nan_to_num(xc), np.nan_to_num(yc)
            dig = np.rint(10 * np.abs(xz - np.trunc(xz))).astype(int)
            across = np.trunc(xz if det < 3 else yz)
            lab = np.where(fin, dig * 100 + np.floor(across / 200.).astype(int) + 50, -1)
            yy, xx = np.nonzero(lab >= 0)
            ll = lab[yy, xx]
            o = np.lexsort((xx, yy, ll))
            yy, xx, ll = yy[o], xx[o], ll[o]
            key = ll.astype(np.int64) * 4096 + yy
            st = np.r_[0, np.nonzero(np.diff(key))[0] + 1]
            en = np.r_[st[1:], len(key)]
            self._geo[(det, angle)] = dict(sl=ll[st], y=yy[st], x0=xx[st], x1=xx[en - 1])
        return self._geo[(det, angle)]

    def along(self, det, angle):
        """Calibrated along-slitlet coordinate (mas): YCAL on detectors 1/2, XCAL on detector 3."""
        if (det, angle) not in self._along:
            self._along[(det, angle)] = np.trunc(np.nan_to_num(self.cal('YCAL' if det < 3 else 'XCAL', det, angle)))
        return self._along[(det, angle)]

    # ---------------------------------------------------------------- flat-edge offset model
    def _edge_model(self):
        """Harmonic coefficients of the MASTER_FLAT slitlet-end position (px, relative to 0 deg) per (IFU, end)."""
        if self._edge_coef is not None:
            return self._edge_coef
        coef = {}
        rows_all = np.arange(2048)
        for det in (1, 2, 3):
            lab = {}
            for a in CAL_ANGLES:
                x, y = self.cal('XCAL', det, a), self.cal('YCAL', det, a)
                ok = np.isfinite(x) & np.isfinite(y)
                xz, yz = np.nan_to_num(x), np.nan_to_num(y)
                dig = np.where(ok, np.rint(10 * np.abs(xz - np.trunc(xz))), -1).astype(int)
                u = np.trunc(xz) if det < 3 else np.trunc(yz)
                sl = np.where(ok, np.rint((u + 1300.0) / 200.0), -1).astype(int)
                good = ok & (sl >= 0) & (sl <= 13) & (dig >= 1) & (dig <= 8)
                lab[a] = np.where(good, dig * 100 + sl, -1)
            fl = dict((a, np.nan_to_num(self.cal('MASTER_FLAT', det, a))) for a in CAL_ANGLES)
            for dig in range(1, 9):
                m60 = (lab[REF_ANGLE] // 100) == dig
                cols = np.nonzero(m60.sum(0) > 200)[0]
                if len(cols) < 50:
                    continue
                c0, c1 = max(cols.min() - 4, 0), min(cols.max() + 5, 2048)
                ends = {'L': [], 'H': []}
                for sl in range(14):
                    key = dig * 100 + sl
                    mk = lab[REF_ANGLE][:, c0:c1] == key
                    anyr = mk.any(1)
                    lo = np.where(anyr, mk.argmax(1) + c0, -1)
                    hi = np.where(anyr, c1 - 1 - mk[:, ::-1].argmax(1), -1)
                    okr = anyr & (hi - lo >= 11) & (rows_all >= 8) & (rows_all < 2040)
                    if not okr.any():
                        continue
                    r, lo_, hi_ = rows_all[okr], lo[okr], hi[okr]
                    ii = np.clip(lo_[:, None] + 3 + np.arange(8)[None, :], 0, 2047)
                    wl = np.clip(lo_[:, None] + np.arange(-1, 3)[None, :], 0, 2047)
                    wh = np.clip(hi_[:, None] + np.arange(-2, 2)[None, :], 0, 2047)
                    eL, eH = [], []
                    for a in CAL_ANGLES:
                        Z = fl[a] * (lab[a] == key)
                        A = np.median(Z[r[:, None], ii], axis=1)
                        with np.errstate(all='ignore'):
                            eL.append(np.nanmedian(lo_ + 2.5 - Z[r[:, None], wl].sum(1) / A))
                            eH.append(np.nanmedian(hi_ - 2.5 + Z[r[:, None], wh].sum(1) / A))
                    ends['L'].append(np.array(eL) - eL[0])
                    ends['H'].append(np.array(eH) - eH[0])
                for e, v in ends.items():
                    if not v:
                        continue
                    with np.errstate(all='ignore'):
                        m = np.nanmedian(np.array(v), 0)
                    if np.all(np.isfinite(m)):
                        coef[((det - 1) * 8 + dig, e)] = np.linalg.lstsq(_harmonic(CAL_ANGLES), m, rcond=None)[0]
        self._edge_coef = coef
        return coef

    def flat_offsets(self, obj, sky):
        """{ifu: dx} predicted from the flat-edge model for the exposure angles of obj and sky (IFUs without a
        model: detector median)."""
        coef = self._edge_model()
        to, ts = _exposure_angle(obj), _exposure_angle(sky)
        dx = {}
        for i in range(1, 25):
            if (i, 'L') in coef and (i, 'H') in coef:
                dx[i] = float(np.mean([(_harmonic(to) @ coef[(i, e)])[0] - (_harmonic(ts) @ coef[(i, e)])[0]
                                       for e in ('L', 'H')]))
        for det in (1, 2, 3):
            v = [dx[i] for i in dx if (i - 1) // 8 == det - 1]
            for i in range((det - 1) * 8 + 1, det * 8 + 1):
                dx.setdefault(i, float(np.median(v)) if v else 0.0)
        return dx

    # ---------------------------------------------------------------- offsets from the frames' OH lines
    def measure_offsets(self, obj, sky, prior):
        """{ifu: (dx, 'MEAS' | 'FLAT', dL, dR)}: the shift of sky along x that nulls the slitlet-end/interior
        ratio of obj/sky at OH-line peaks, searched within +-DX_HALF px of prior[ifu] (see module doc)."""
        ang = _angle_bin(float(fits.getheader(obj)[ROT]))
        out = {}

        def hp(a):
            return a - median_filter(np.nan_to_num(a), size=(15, 1))

        def lin_shift(A, B):
            ok = np.isfinite(A) & np.isfinite(B)
            dA = np.gradient(np.where(ok, A, 0.0))
            M = np.c_[A[ok], dA[ok], np.ones(ok.sum())]
            c = np.linalg.lstsq(M, B[ok], rcond=None)[0]
            return -c[1] / c[0]

        for det in (1, 2, 3):
            O = np.nan_to_num(np.asarray(fits.getdata(obj, det), float))
            S = np.nan_to_num(np.asarray(fits.getdata(sky, det), float))
            xc, yc = self.cal('XCAL', det, ang), self.cal('YCAL', det, ang)
            M = np.isfinite(yc)
            left = M & ~np.roll(M, 1, 1)
            right = M & ~np.roll(M, -1, 1)
            pos = np.full(M.shape, -99)
            posr = np.full(M.shape, -99)
            for r in range(M.shape[0]):
                for l, rr in zip(np.nonzero(left[r])[0], np.nonzero(right[r])[0]):
                    n = rr - l + 1
                    pos[r, l:rr + 1] = np.arange(n)
                    posr[r, l:rr + 1] = np.arange(n)[::-1]
            with np.errstate(invalid='ignore'):
                digit = np.rint(10 * np.abs(xc - np.trunc(xc)))
            Ohp = hp(O)
            for k in range(1, 9):
                ifu = (det - 1) * 8 + k
                try:
                    r = self._measure_ifu(O, S, Ohp, digit, pos, posr, k, prior[ifu], hp, lin_shift)
                except Exception:                       # e.g. an IFU without signal: flat value
                    r = None
                if r is not None:
                    out[ifu] = r
        for i in range(1, 25):
            out.setdefault(i, (float(prior[i]), 'FLAT', np.nan, np.nan))
        return out

    @staticmethod
    def _measure_ifu(O, S, Ohp, digit, pos, posr, k, prior_ifu, hp, lin_shift):
        """One IFU of measure_offsets: (dx, 'MEAS' | 'FLAT', dL, dR), or None (too few pixels)."""
        cols = np.nonzero((digit == k).sum(0) > 200)[0]
        if len(cols) < 50:
            return None
        sub = (slice(None), slice(max(cols.min() - 4, 0), min(cols.max() + 5, 2048)))
        msk = digit[sub] == k
        msk[:60] = False
        msk[1990:] = False
        inner = msk & (pos[sub] > 3) & (posr[sub] > 3)
        if inner.sum() < 1000:
            return None
        Shp0 = hp(S[sub])
        ro = np.array([np.median(Ohp[sub][r][inner[r]]) if inner[r].any() else np.nan for r in range(2048)])
        rs = np.array([np.median(Shp0[r][inner[r]]) if inner[r].any() else np.nan for r in range(2048)])
        ok = np.isfinite(ro) & np.isfinite(rs)
        best = max(range(-4, 5), key=lambda s: np.nansum(ro[ok][10:-10] * np.roll(rs[ok], s)[10:-10]))
        dy = best + lin_shift(np.roll(np.nan_to_num(rs), best), np.nan_to_num(ro))
        Sy = ndshift(S[sub], (dy, 0), order=3, mode='nearest')
        thr = np.percentile(Shp0[inner], 98)
        Sref = maximum_filter(np.where(inner, Shp0, 0), size=(1, 15))
        pk = (Ohp[sub] == maximum_filter(Ohp[sub], size=(5, 1))) & (Sref > thr) & msk
        p0 = np.round(prior_ifu / DX_STEP) * DX_STEP
        deltas = p0 + DX_STEP * np.arange(-int(round(DX_HALF / DX_STEP)), int(round(DX_HALF / DX_STEP)) + 1)
        res = []
        for d in deltas:
            Shp = hp(ndshift(Sy, (0, d), order=3, mode='nearest') if abs(d) > 1e-6 else Sy)
            with np.errstate(all='ignore'):
                q = Ohp[sub] / Shp
            qin = np.nanmedian(q[pk & inner])
            e = []
            for p in (pos[sub], posr[sub]):
                for j in (0, 1):
                    sel = pk & (p == j)
                    e.append(np.nanmedian(q[sel]) / qin - 1 if sel.sum() > 30 else np.nan)
            res.append(e)
        A = np.array(res)

        def opt(c):
            v = A[:, c]
            if not np.all(np.isfinite(v)):
                return np.nan
            c2 = np.sum(v ** 2, 1)
            j = int(np.argmin(c2))
            if j == 0 or j == len(c2) - 1:          # minimum at the border of the search window
                return np.nan
            y0, y1, y2 = c2[j - 1:j + 2]
            den = y0 - 2 * y1 + y2
            return deltas[j] + (0.5 * DX_STEP * (y0 - y2) / den if den > 0 else 0.)

        dL, dR = opt([0, 1]), opt([2, 3])
        v = [d for d in (dL, dR) if np.isfinite(d)]
        dx = np.nan
        if len(v) == 2 and abs(dL - dR) <= DX_LR:
            dx = 0.5 * (dL + dR)
        elif len(v) == 1:
            dx = v[0]
        if np.isfinite(dx) and abs(dx - prior_ifu) <= DX_FLAT:
            return (float(dx), 'MEAS', float(dL), float(dR))
        else:
            return (float(prior_ifu), 'FLAT', float(dL), float(dR))


def align_sky_data(S0, det, dx, angle, cal):
    """Slitlet-end alignment of one raw SKY detector image S0 (float array) to the trace positions of an
    OBJECT frame taken at calibration bin angle; dx = {ifu: shift along x}. Returns (S', stats)."""
    Srep = cal.repair(S0, det)
    dig = cal.ifu_digit(det)
    D = np.zeros_like(Srep)
    for k in range(1, 9):
        cols = np.nonzero((dig == k).sum(0) > 0)[0]
        if not len(cols):
            continue
        c0, c1 = max(cols.min() - 2, 0), min(cols.max() + 3, Srep.shape[1])
        b0, b1 = max(c0 - 32, 0), min(c1 + 32, Srep.shape[1])        # margin: the spline prefilter decays as 0.27^n
        blk = Srep[:, b0:b1]
        D[:, c0:c1] = (ndshift(blk, (0, dx[(det - 1) * 8 + k]), order=3, mode='nearest', prefilter=True)
                       - blk)[:, c0 - b0:c1 - b0]
    g = cal.geometry(det, angle)
    al = cal.along(det, angle)
    out = S0.copy()
    width = g['x1'] - g['x0'] + 1
    ok = (width >= 12) & (width <= 16)
    sum_corr, sum_d = 0., 0.
    for sl in np.unique(g['sl']):
        m = (g['sl'] == sl) & ok
        if m.sum() < 200:
            continue
        ys, x0, x1 = g['y'][m], g['x0'][m], g['x1'][m]
        for base, sgn in ((x0, 1), (x1, -1)):
            spec = np.median(np.stack([Srep[ys, base + sgn * k] for k in SPEC_PIX]), 0)
            dspec = np.gradient(spec)
            d2spec = np.gradient(dspec)
            yv = np.concatenate([ys for j in EDGE_PIX])
            xv = np.concatenate([base + sgn * j for j in EDGE_PIX])
            rv = np.concatenate([np.arange(len(ys)) for j in EDGE_PIX])
            b = np.floor(al[yv, xv] / BIN_MAS).astype(int)
            A = np.c_[np.ones(len(yv)), spec[rv], dspec[rv], d2spec[rv]]
            d = D[yv, xv]
            corr = np.zeros(len(yv))
            for bb in np.unique(b):
                sel = np.nonzero(b == bb)[0]
                if len(sel) < 30:
                    continue
                use = np.isfinite(d[sel])
                for _ in range(3):
                    c = np.linalg.lstsq(A[sel][use], d[sel][use], rcond=None)[0]
                    r = d[sel] - A[sel] @ c
                    s = 1.4826 * np.nanmedian(np.abs(r[use]))
                    use = np.isfinite(d[sel]) & (np.abs(r) < 4 * s)
                corr[sel] = A[sel, 1:] @ c[1:]
            out[yv, xv] = S0[yv, xv] + corr
            sum_corr += np.sum(np.abs(corr))
            sum_d += np.nansum(np.abs(d))
    return out, dict(changed=int(np.sum(out != S0)), corr_over_d=sum_corr / sum_d if sum_d > 0 else np.nan)


def _write_aligned_copy(sky, obj, dst, cal, dx_mode, logger=None):
    """Aligned copy of sky for obj with obj's NAANGLE; returns {ifu: (dx, src, dx_flat)}."""
    a_obj = float(fits.getheader(obj)[ROT])
    angle = _angle_bin(a_obj)
    flat = cal.flat_offsets(obj, sky)
    if dx_mode == 'measure':
        meas = cal.measure_offsets(obj, sky, flat)
        dx = dict((i, meas[i][0]) for i in meas)
        src = dict((i, meas[i][1]) for i in meas)
    else:
        dx, src = dict(flat), dict((i, 'FLAT') for i in flat)
    msg = []
    with fits.open(sky) as H:
        h0 = H[0].header
        h0[ORIG] = (float(h0[ROT]), 'true OCS ROT NAANGLE of this SKY frame')
        h0[ROT] = a_obj
        for det in (1, 2, 3):
            new, st = align_sky_data(np.asarray(H[det].data, float), det, dx, angle, cal)
            H[det].data = new.astype(np.float32)
            msg.append('det{0} {1:.1f} % px, |corr|/|D| {2:.2f}'.format(det, 100. * st['changed'] / new.size,
                                                                        st['corr_over_d']))
        H.writeto(dst, overwrite=True)
    if logger is not None:
        nm = sum(1 for i in src if src[i] == 'MEAS')
        logger.info('-- sky_align: {0} for {1}: calibrations at {2:.2f} deg (bin {3:.0f}), dx det medians {4} px '
                    '({5}/24 IFUs measured, rest flat model); {6}'.format(
                        os.path.basename(sky), os.path.basename(obj), a_obj, angle,
                        ' '.join('{0:+.2f}'.format(np.median([dx[i] for i in range(d * 8 + 1, d * 8 + 9)]))
                                 for d in range(3)), nm, '; '.join(msg)))
    return dict((i, (dx[i], src[i], flat[i])) for i in range(1, 25))


def obj_sky_pairs(frames, obsband=None):
    """[(OBJECT frame, SKY frame)]: every OBJECT frame with its nearest same-DIT SKY frame, as sky_angle_plan and
    match_obj_sky pair them (sky_angle_plan also lists, for a SKY frame no OBJECT frame uses, the nearest OBJECT
    frame, for the angle only: that is not a pair). OBJECT frames without a same-DIT SKY frame are left out."""
    inf = [_info(f) for f in frames]
    if obsband:
        inf = [i for i in inf if i["band"] == obsband]
    skies = [i for i in inf if i["sky"]]
    pairs = []
    for o in (i for i in inf if not i["sky"]):
        cand = [s for s in skies if s["dit"] == o["dit"]]
        if cand:
            pairs.append((o["path"], min(cand, key=lambda s: abs(s["t"] - o["t"]))["path"]))
    return sorted(pairs)


def align_skies(frames, cal_lines, caldir, outputdir, subdir, obsband, routine_args, dx_mode='measure',
                esorex_opts='', env=None, logger=None):
    """Write sky_align/SCI_RECONSTRUCTED_<OBJECT frame> (the aligned SKY cube of every OBJECT frame).

    frames       : the original science frames of this OB and band (not angle-edited SKY copies)
    cal_lines    : the calibration lines of the reconstruction SOF ('<file> <TAG>')
    routine_args : the kmos_sci_red arguments of the main reconstruction
    Returns the number of cubes written, or -1 if kmos_sci_red failed.
    """
    dx_mode = str(dx_mode).strip().lower()
    if dx_mode not in ('measure', 'flat'):
        raise ValueError('sky_align_dx must be measure or flat, got {0!r}'.format(dx_mode))
    al_dir = os.path.abspath(os.path.join(outputdir, subdir, 'sky_align'))     # the run below changes the cwd
    pairs = obj_sky_pairs(frames, obsband)

    #cubes left over from an earlier run would be picked up by subtract_sky
    for o, _ in pairs:
        old = os.path.join(al_dir, 'SCI_RECONSTRUCTED_{0}'.format(os.path.basename(o)))
        if os.path.isfile(old):
            os.remove(old)
    if logger is not None:
        logger.info('-- sky_align: {0} OBJECT frames, SKY copies aligned at the slitlet ends (dx: {1})'.format(
            len(pairs), dx_mode))
    if not pairs:
        return 0

    #own working directory: copies named after their OBJECT frame, so the cube names follow the OBJECT frames
    work = os.path.join(al_dir, 'work_{0}'.format(id_gen()))
    os.makedirs(work)
    cal = AlignCalib(caldir, obsband)
    info = {}
    try:
        for o, s in pairs:
            info[o] = (s, _write_aligned_copy(s, o, os.path.join(work, os.path.basename(o)), cal, dx_mode, logger))
    finally:
        cal.close()
    sof = os.path.join(work, 'sky_align.sof')
    with open(sof, 'wt') as fh:
        for o, _ in pairs:
            fh.write('{0} SCIENCE\n'.format(os.path.join(work, os.path.basename(o))))
        for line in cal_lines:
            fname, tag = line.rsplit(None, 1)
            fh.write('{0} {1}\n'.format(os.path.abspath(fname), tag))

    opts = '{0} --output-dir={1} --log-dir={1}'.format(esorex_opts, work).strip()
    cwd = os.getcwd()
    os.chdir(work)
    try:
        ecode = run_spark('kmos_sci_red', args=routine_args, sof=sof, logger=logger, esorex_opts=opts, env=env)
    finally:
        os.chdir(cwd)
    if ecode != 0:
        if logger is not None:
            logger.error('-- sky_align: kmos_sci_red failed (exit {0})'.format(ecode))
        _clear(work)
        return -1

    nwritten = 0
    for o, (s, dxi) in sorted(info.items()):
        cube = os.path.join(work, 'SCI_RECONSTRUCTED_{0}'.format(os.path.basename(o)))
        if not os.path.isfile(cube):
            if logger is not None:
                logger.warning('-- sky_align: no cube for {0}, classic O - S'.format(os.path.basename(o)))
            continue
        patch_header(cube, HDR_KEYS, s)
        with fits.open(cube, mode='update') as fh:
            h = fh[0].header
            a_obj = float(h[ROT])
            h[ROT] = float(fits.getheader(s)[ROT])
            h[USED] = (round(a_obj, 3), 'rotator angle used to select calibrations')
            h.remove(ORIG, ignore_missing=True)
            h[ALIGN_SKY] = (os.path.basename(s), 'SKY frame aligned to OBJ')
            h[ALIGN_OBJ] = (os.path.basename(o), 'OBJECT frame this SKY cube is for')
            for i in range(1, 25):
                h[ALIGN_DX + str(i)] = (round(dxi[i][0], 4), '[px] S shift along det x, IFU {0}'.format(i))
            for i in range(1, 25):
                h[PFX + 'DXF' + str(i)] = (round(dxi[i][2], 4), '[px] flat-edge model value')
            for i in range(1, 25):
                h[PFX + 'SRC' + str(i)] = (dxi[i][1], 'MEAS: from OH lines, FLAT: flat model')
            h['HISTORY'] = 'pyspark sky_align: slitlet ends aligned to {0}, calibrations at {1:.3f} deg'.format(
                os.path.basename(o), a_obj)
        os.replace(cube, os.path.join(al_dir, os.path.basename(cube)))
        nwritten += 1
    _clear(work)
    return nwritten


def _clear(work):
    """Remove the working directory (aligned raw copies, logs and unused products)."""
    for f in glob.glob(os.path.join(work, '*')):
        os.remove(f)
    os.rmdir(work)
