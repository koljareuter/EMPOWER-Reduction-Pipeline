"""Hybrid sky subtraction for OBs whose SKY frames were taken at another rotator angle (docs/hybrid-sky.md).

pyspark subtracts the sky in cube space: O and S are reconstructed separately, each with its own calibrations
and its own per-frame OH wavelength correction. When the SKY frame's rotator angle differs from the OBJECT's
(37-90 deg for cosmos_0.6_0.8_h21.4/h21.7, cosmos_target_legac, goods_0.6_0.8_h21.6_lir), the same detector
pixels (warm/hot/RTS pixels) land in different voxels of O and S and do not cancel: 3-5 % outlier voxels and a
pattern shared by all O - S frames of a target, which does not average down in the combination. These lists
are 2-3x noisier per sqrt(t) than the matched ones.

The DRP's own detector-level subtraction (kmos_sci_red -no_subtract=FALSE) maps S with the OBJECT's
calibrations and lambda correction, so detector features cancel, but it leaves the OH-line misalignment between
the two frames (flexure) that the separate reconstruction corrects. The hybrid keeps both:
    R_ds = detector-level O - S (DRP)
    D    = (O - S) - R_ds = recon_O(S) - recon_S(S)       (linear reconstruction: S frame only, no target signal)
    R_hyb = R_ds + model(D),   model per slitlet: in OH-line planes the slitlet median of D, elsewhere a fit of
            the slitlet-median of D with (a0 + a1 x) g + (b0 + b1 x) S + (c0 + c1 x) g',  g = dS/dlambda,
            S = IFU median of the SKY cube, x = (lambda - 0.93)/0.14, all high-passed (101 planes)
Every term of the model comes from the SKY frame; R_hyb then goes through ZAP exactly like O - S.

Validated on IZ (2026-10-23): 6-frame Hbeta-window stacks noise x0.44-0.67 on two mismatched OBs, matched OB
unchanged (so use it only for mismatched OBs); injected detector lines keep their flux exactly when the
object's lambda correction is frozen (906 compact + 241 extended/edge sources, ratio 1.00000). Not for H (the
IZ model makes H 1.3-3.3x worse). YJ (2026-10-27) needs oh_model='pos3' and free_model='ifumed': its
detector-level cube leaves about 10x larger OH residuals than in IZ.
"""
from __future__ import print_function

import numpy as np
import bottleneck as bn

from ..utils.obsutils import find_lines

__all__ = ["hybrid_ifu", "highpass"]


def highpass(spec, width=101):
    """spec minus its centred running median (NaN-aware, at least width/4 valid values)."""
    s = np.asarray(spec, float)[:, None, None]
    nz, half = s.shape[0], width // 2
    pad = np.concatenate([np.repeat(s[:1], half, 0) * np.nan, s, np.repeat(s[-1:], half, 0) * np.nan], 0)
    med = bn.move_median(pad, width, axis=0, min_count=width // 4)[width - 1:width - 1 + nz]
    return (s - med)[:, 0, 0]


def _runmed(a, width=101):
    """centred running median along axis 0 (NaN-aware, at least width/4 valid values, NaN padding)."""
    a = np.asarray(a, float)
    nz, half = a.shape[0], width // 2
    pad = np.concatenate([np.full((half,) + a.shape[1:], np.nan), a, np.full((half,) + a.shape[1:], np.nan)], 0)
    return bn.move_median(pad, width, axis=0, min_count=width // 4)[width - 1:width - 1 + nz]


def _shift(mask, k):
    """mask shifted by k planes without wrap-around (k > 0: towards higher plane numbers)."""
    out = np.zeros_like(mask)
    if k > 0:
        out[k:] = mask[:-k]
    elif k < 0:
        out[:k] = mask[-k:]
    else:
        out[:] = mask
    return out


def _along(sub, oh_model):
    """Model of D in the OH-core planes along one slitlet: sub is (nz, L) with L spaxels along the slitlet.
    'runN': centred running median over N spaxels (window shrinks at the slitlet ends); 'lin': linear fit along
    the slitlet with one 3-sigma clipping pass; 'full': D itself."""
    if oh_model == "full":
        return np.nan_to_num(sub)
    nz, L = sub.shape
    if oh_model.startswith("run"):
        n = int(oh_model[3:])
        half = n // 2
        pad = np.concatenate([np.full((nz, half), np.nan), sub, np.full((nz, half), np.nan)], 1)
        return np.nan_to_num(bn.move_median(pad, 2 * half + 1, axis=1, min_count=1)[:, 2 * half:2 * half + L])
    if oh_model == "lin":
        t = np.arange(L) - (L - 1) / 2.
        w = np.isfinite(sub).astype(float)
        y = np.nan_to_num(sub)
        for _ in range(2):
            sw, st, stt = w.sum(1), (w * t).sum(1), (w * t * t).sum(1)
            sy, sty = (w * y).sum(1), (w * t * y).sum(1)
            det = sw * stt - st * st
            ok = det > 0
            a = np.where(ok, (stt * sy - st * sty) / np.where(ok, det, 1), np.where(sw > 0, sy / np.maximum(sw, 1), 0))
            b = np.where(ok, (sw * sty - st * sy) / np.where(ok, det, 1), 0)
            fit = a[:, None] + b[:, None] * t[None]
            res = np.where(w > 0, y - fit, np.nan)
            sig = 1.4826 * np.nanmedian(np.abs(res), 1)
            w = (np.isfinite(sub) & ~(np.abs(y - fit) > 3 * sig[:, None])).astype(float)
        return fit
    raise ValueError("unknown oh_model %r" % oh_model)


def hybrid_ifu(ocube, scube, rds, ifu, lam, wing=0, oh_model="median", free_model="tmpl", edge=0):
    """Hybrid sky-subtracted cube of one IFU.

    ocube, scube : un-subtracted OBJECT and nearest SKY cubes (pyspark sky/ reconstructions)
    rds          : the DRP detector-level sky-subtracted cube of the same OBJECT frame and IFU
    ifu          : IFU number (slitlets run along y for IFUs 1-16, along x for 17-24)
    lam          : wavelength of every plane (micron)
    wing         : 0 = as validated in October 2026.  W >= 2: in the OH-line wings, planes 2..W from a find_lines
                   core, use the (high-passed) IFU-wide median of D instead of the per-slitlet template fit.  The
                   misalignment residual there is common to all slitlets and too weak for a per-slitlet estimate;
                   with the template fit it survives as a pattern shared by the IFUs of a detector
                   (docs/hybrid-sky.md, 'OH wings'; W = 3 adopted for IZ).
    oh_model     : model of D in the OH-core planes (find_lines core +-1). 'median' = slitlet median (as validated
                   for IZ); 'run3'/'run5' = running median along the slitlet, 'lin' = linear fit along the slitlet,
                   'full' = D itself (= classic O - S there). In YJ the detector-level cube leaves OH residuals as
                   large as the sky whose size changes towards the slitlet ends (docs/hybrid-sky.md, 'YJ').
                   'run3b<T>': run3 only in OH-core planes where the high-passed SKY spectrum exceeds T times its
                   rms away from lines (bright lines), the slitlet median in the others.
                   'pos'/'pos<N>': slitlet median plus the position profile along the slitlet (median over the
                   IFU's slitlets of D minus the slitlet median; 'pos<N>' only at the N end positions); gate as
                   above with 'pos<N>b<T>'.
    free_model   : 'tmpl' = per-slitlet template fit of the high-passed slitlet median of D outside the OH cores and
                   the high-passed IFU median in the wings (as validated for IZ); 'ifumed' = the exact per-plane IFU
                   median of D there plus the template fit of the slitlet deviations from it.
    edge         : 0 = off.  N > 0: add the slitlet-end level of D at the N end positions of every slitlet in all planes
                   where no 'pos' profile is applied: per plane the median over the IFU's slitlets of D minus the
                   slitlet median, smoothed along wavelength (101-plane running median without the OH-core planes).
                   Without it the model is flat along the slitlet there and the result keeps the slitlet-end level
                   of the detector-level cube (hot edge rows in the combined cubes, docs/hybrid-sky.md 'Slitlet ends').
    Returns a float64 cube; NaN where rds is NaN.
    """
    oc = np.asarray(ocube, float)
    sc = np.asarray(scube, float)
    rds = np.asarray(rds, float)
    D = (oc - sc) - rds
    nz, ny, nx = D.shape
    x = (np.asarray(lam, float) - 0.93) / 0.14      # only scales the linear-term coefficients: band-independent model
    smed = np.nanmedian(sc.reshape(nz, -1), axis=1)
    smed = np.where(np.isfinite(smed), smed, 0.0)
    g = np.gradient(smed)
    g2 = np.gradient(g)
    T = np.column_stack([highpass(g), highpass(g) * x, highpass(smed), highpass(smed) * x,
                         highpass(g2), highpass(g2) * x])
    T = np.where(np.isfinite(T), T, 0.0)
    sky = np.nanmedian(sc[:, 1:13, 1:13].reshape(nz, -1), axis=1)
    core = find_lines(sky).astype(bool)
    oh = core | np.roll(core, 1) | np.roll(core, -1)
    wings, ifu_med = None, None
    if wing >= 2:
        cw = core & np.isfinite(sky)          # find_lines can flag a NaN band-end tail after an open rising edge
        grown = cw.copy()
        for k in range(1, int(wing) + 1):
            grown |= _shift(cw, k) | _shift(cw, -k)
        wings = grown & ~oh
        ifu_med = np.nan_to_num(highpass(np.nanmedian(D.reshape(nz, -1), axis=1)))
    ifu_full = None
    if free_model == "ifumed":
        # exact per-plane IFU median of D (no high-pass: with the dense OH residuals of a badly aligned detector-level
        # cube the 101-plane running median of D is not smooth and leaks into the result)
        ifu_full = np.nan_to_num(np.nanmedian(D.reshape(nz, -1), axis=1))
    elif free_model != "tmpl":
        raise ValueError("unknown free_model %r" % free_model)
    along = oh_model
    sel = oh
    if oh_model.startswith(("run", "pos")) and "b" in oh_model:
        # gated: along the slitlet only on bright OH lines (high-passed SKY spectrum > T x its rms away from lines)
        along, thr = oh_model.split("b")
        thr = float(thr or 30)
        grown6 = core.copy()
        for k in range(1, 7):
            grown6 |= _shift(core, k) | _shift(core, -k)
        shp = highpass(np.where(np.isfinite(sky), sky, np.nan))
        ref = shp[~grown6 & np.isfinite(shp)]
        rms = 1.4826 * np.nanmedian(np.abs(ref - np.nanmedian(ref))) if ref.size > 50 else np.inf
        sel = oh & (np.nan_to_num(shp) > thr * rms)
    if ifu <= 16:
        groups = [(slice(None), slice(i, i + 1)) for i in range(nx)]
    else:
        groups = [(slice(i, i + 1), slice(None)) for i in range(ny)]
    prof = None
    if along.startswith("pos") and sel.any():
        # 'pos[N]': the deviation of D from the slitlet median as a function of the position along the slitlet, the
        # same for all slitlets of the IFU (reconstruction edge effect at the slitlet ends): median over the slitlets,
        # so single-spaxel features of D (SKY-frame detector artefacts) are not put back; N > 0: end positions only
        nend = int(along[3:]) if len(along) > 3 else 0
        res = []
        for gy, gx in groups:
            sub = D[:, gy, gx].reshape(nz, -1)
            if np.isfinite(sub).sum() >= nz:
                res.append(sub[sel] - np.nanmedian(sub[sel], axis=1)[:, None])
        if len(res) >= 3:
            prof = np.nan_to_num(np.nanmedian(np.array(res), axis=0))
            if nend > 0:
                prof[:, nend:prof.shape[1] - nend] = 0.0
    eprof, epl = None, None
    if int(edge) > 0:
        res = []
        for gy, gx in groups:
            sub = D[:, gy, gx].reshape(nz, -1)
            if np.isfinite(sub).sum() >= nz:
                res.append(sub - np.nanmedian(sub, axis=1)[:, None])
        if len(res) >= 3:
            pr = np.nanmedian(np.array(res), axis=0)
            pr[oh] = np.nan                               # smooth level: OH-core planes left out of the running median
            sm = _runmed(pr)
            zz = np.arange(nz)
            for j in range(sm.shape[1]):                 # dense OH regions (< width/4 free planes): interpolate
                okj = np.isfinite(sm[:, j])
                if 0 < okj.sum() < nz:
                    sm[~okj, j] = np.interp(zz[~okj], zz[okj], sm[okj, j])
            eprof = np.nan_to_num(sm)
            eprof[:, int(edge):eprof.shape[1] - int(edge)] = 0.0
            epl = np.ones(nz, bool)
            if oh_model != "median" and not (along.startswith("pos") and prof is None):
                epl = ~sel                                # planes with a 'pos'/along-slitlet model keep it
    model = np.zeros_like(D)
    for gy, gx in groups:
        sub = D[:, gy, gx].reshape(nz, -1)
        if np.isfinite(sub).sum() < nz:
            continue
        if ifu_full is None:
            dmed = highpass(np.nanmedian(sub, axis=1))
        else:
            dmed = highpass(np.nanmedian(sub, axis=1) - ifu_full)
        ok = np.isfinite(dmed)
        if ok.sum() < 200:
            continue
        coef = np.linalg.lstsq(T[ok], dmed[ok], rcond=None)[0]
        if ifu_full is None:
            m = np.where(oh, np.nan_to_num(np.nanmedian(sub, axis=1)), T @ coef)
            if wings is not None:
                m = np.where(wings, ifu_med, m)
        else:
            m = np.where(oh, np.nan_to_num(np.nanmedian(sub, axis=1)), ifu_full + T @ coef)
            if wings is not None:
                m = np.where(wings, ifu_full, m)
        if oh_model == "median":
            model[:, gy, gx] = m[:, None, None]
        else:
            # per spaxel along the slitlet in the selected OH-core planes, the slitlet value elsewhere
            m2 = np.repeat(m[:, None], sub.shape[1], axis=1)
            if along.startswith("pos"):
                if prof is not None:
                    m2[sel] = m[sel][:, None] + prof
            elif sel.any():
                m2[sel] = _along(sub[sel], along)
            model[:, gy, gx] = m2.reshape(model[:, gy, gx].shape)
        if eprof is not None and eprof.shape[1] == sub.shape[1]:
            m3 = model[:, gy, gx].reshape(nz, -1)
            m3[epl] += eprof[epl]
            model[:, gy, gx] = m3.reshape(model[:, gy, gx].shape)
    hyb = rds + model
    hyb[~np.isfinite(rds)] = np.nan
    return hyb
