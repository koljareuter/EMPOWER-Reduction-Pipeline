"""Where does the target signal of the hybrid (VAR, e.g. PROCDIR_RULEALL) differ from the classic run (REF, e.g.
PROCDIR_CLEAN)? Per IFU of one OB, frames stacked in IFU coordinates, at three stages:
  raw    VAR: classic O - S = sky/O - sky/S   vs   R_ds = skysub_det/O (DRP detector-level O - S, before the model)
         (both from VAR's own sky/ and skysub_det/; tests the assumption that R_ds holds the same target signal)
  s      science/*_s.fits      (after the sky subtraction incl. hybrid model and ZAP): VAR / REF
  final  science/*_s_i_bkg2d_rv.fits                                              : VAR / REF
Target signal = sum over the OH-free planes of (aperture r = 2.8 spx at the continuum peak  -  N_ap x per-plane
median of the spaxels > 4.5 spx from the peak, outer ring excluded), i.e. with the local background removed, so a
change of the background level does not count as a change of the target. 'bg' is that background level per
spaxel and plane (VAR - REF, in units of the per-spaxel noise of one plane of the stack), 'line' the same ratio
for the OH-free planes where the high-passed aperture spectrum (mean of both) is > 5 sigma.
Only IFUs whose REF target signal is > 10 sigma are listed (stars first: highest S/N).

    python scripts/diag_signal.py --ref /local_storage/KOLJA/PROCDIR_CLEAN --var /local_storage/KOLJA/PROCDIR_RULEALL \
        --ob KMOS.2026-01-14T05:23:13.608 --band IZ
"""
import argparse
import glob
import os
import warnings

import numpy as np
from astropy.io import fits
from scipy.ndimage import median_filter

warnings.filterwarnings('ignore')
p = argparse.ArgumentParser()
p.add_argument('--ref', required=True)
p.add_argument('--var', required=True)
p.add_argument('--ob', required=True)
p.add_argument('--band', default='IZ')
a = p.parse_args()
LAM = {'IZ': [(0.80, 0.92), (0.99, 1.05)], 'YJ': [(1.02, 1.10), (1.16, 1.33)], 'H': [(1.50, 1.75)]}[a.band]
SKY_KEY = 'HIERARCH ESO PYSPARK SKYSUB DET SKY'


def rmad(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return 1.4826 * np.median(np.abs(x - np.median(x))) if x.size else np.nan


def data(f, ifu):
    with fits.open(f) as h:
        if ifu >= len(h) or h[ifu].data is None:
            return None
        return h[ifu].data.astype(float)


def lam_of(f, ifu):
    hd = fits.getheader(f, ifu)
    return hd['CRVAL3'] + hd['CDELT3'] * (np.arange(hd['NAXIS3']) + 1 - hd['CRPIX3'])


def stack(files, ifu):
    cs = [c for c in (data(f, ifu) for f in files) if c is not None]
    return np.nanmean(cs, axis=0) if cs else None


sci = lambda pd, suf: sorted(glob.glob(os.path.join(pd, a.ob, 'science', 'SCI_RECONSTRUCTED_*' + suf)))
fin_r, fin_v = sci(a.ref, '_s_i_bkg2d_rv.fits'), sci(a.var, '_s_i_bkg2d_rv.fits')
s_r, s_v = sci(a.ref, '_s.fits'), sci(a.var, '_s.fits')
names = sorted(set(map(os.path.basename, fin_r)) & set(map(os.path.basename, fin_v)))
fin_r = [os.path.join(a.ref, a.ob, 'science', n) for n in names]
fin_v = [os.path.join(a.var, a.ob, 'science', n) for n in names]
stems = [n.replace('_s_i_bkg2d_rv.fits', '') for n in names]
s_r = [os.path.join(a.ref, a.ob, 'science', s + '_s.fits') for s in stems]
s_v = [os.path.join(a.var, a.ob, 'science', s + '_s.fits') for s in stems]
raw_o, raw_s, raw_d = [], [], []
for s in stems:
    det = os.path.join(a.var, a.ob, 'skysub_det', s + '.fits')
    if os.path.isfile(det):
        sky = fits.getheader(det).get(SKY_KEY)
        raw_d.append(det)
        raw_o.append(os.path.join(a.var, a.ob, 'sky', s + '.fits'))
        sf = os.path.join(a.var, a.ob, 'sky', 'SCI_RECONSTRUCTED_' + sky)
        if not os.path.isfile(sf):                      # copies on Windows: ':' -> '-' in the file names
            sf = os.path.join(a.var, a.ob, 'sky', 'SCI_RECONSTRUCTED_' + sky.replace(':', '-'))
        raw_s.append(sf)
print('%s: %d frames; raw stage from %d detector-level cubes of VAR' % (a.ob, len(names), len(raw_d)))
h0 = fits.getheader(fin_r[0])

rows = []
for ifu in range(1, 25):
    fr, fv = stack(fin_r, ifu), stack(fin_v, ifu)
    if fr is None or fv is None or not os.path.isfile(raw_o[0] if raw_o else ''):
        continue
    so = data(raw_o[0], ifu)
    if so is None:
        continue
    sky = np.nanmedian(so.reshape(so.shape[0], -1), axis=1)
    lam_raw = lam_of(raw_o[0], ifu)
    lam_fin = lam_of(fin_r[0], ifu)
    ny, nx = fr.shape[1:]
    yy, xx = np.mgrid[:ny, :nx]

    def free_mask(lam):
        sk = np.interp(lam, lam_raw, sky)
        ok = np.zeros(len(lam), bool)
        for lo, hi in LAM:
            ok |= (lam > lo) & (lam < hi)
        run = median_filter(np.nan_to_num(sk), 101)
        return ok & np.isfinite(sk) & (sk < 1.1 * run)

    fm_fin, fm_raw = free_mask(lam_fin), free_mask(lam_raw)
    img = np.nanmedian(fr[fm_fin], axis=0)
    sub = np.where((yy > 1) & (yy < ny - 2) & (xx > 1) & (xx < nx - 2) & np.isfinite(img), img, -np.inf)
    py, px = np.unravel_index(np.argmax(sub), sub.shape)
    rr = np.hypot(yy - py, xx - px)
    ap = rr <= 2.8
    bgm = (rr > 4.5) & (yy > 0) & (yy < ny - 1) & (xx > 0) & (xx < nx - 1)

    def target(c, fm):
        bg = np.nanmedian(c[:, bgm], axis=1)
        spec = np.nansum(c[:, ap], axis=1) - ap.sum() * bg
        return spec, bg

    def line_ratio(cv, cr, fm):
        hv = cv - median_filter(np.nan_to_num(cv), 31)
        hr = cr - median_filter(np.nan_to_num(cr), 31)
        m = 0.5 * (hv + hr)
        st = fm & (m > 5 * rmad(m[fm]))
        return np.sum(hv[st]) / np.sum(hr[st]) if st.sum() >= 3 else np.nan

    tr, bgr = target(fr, fm_fin)
    tv, bgv = target(fv, fm_fin)
    noise = rmad((fr - median_filter(np.nan_to_num(fr), (31, 1, 1)))[fm_fin][:, bgm])
    err = rmad(tr[fm_fin] - median_filter(np.nan_to_num(tr), 31)[fm_fin]) * np.sqrt(fm_fin.sum())
    sn = np.nansum(tr[fm_fin]) / err
    if not sn > 10:
        continue
    row = dict(ifu=ifu, arm=str(h0.get('HIERARCH ESO OCS ARM%d NAME' % ifu, '?')), sn=sn, peak=(py, px),
               fin=np.nansum(tv[fm_fin]) / np.nansum(tr[fm_fin]), line_fin=line_ratio(tv, tr, fm_fin),
               bg_fin=np.nanmean(bgv[fm_fin] - bgr[fm_fin]) / noise)
    sr, sv = stack(s_r, ifu), stack(s_v, ifu)
    t1, b1 = target(sr, fm_raw)
    t2, b2 = target(sv, fm_raw)
    row['s'] = np.nansum(t2[fm_raw]) / np.nansum(t1[fm_raw])
    row['bg_s'] = np.nanmean(b2[fm_raw] - b1[fm_raw]) / noise
    co = stack(raw_o, ifu) - stack(raw_s, ifu)
    rd = stack(raw_d, ifu)
    t1, b1 = target(co, fm_raw)
    t2, b2 = target(rd, fm_raw)
    row['raw'] = np.nansum(t2[fm_raw]) / np.nansum(t1[fm_raw])
    row['bg_raw'] = np.nanmean(b2[fm_raw] - b1[fm_raw]) / noise
    rows.append(row)

print('target signal VAR/REF (raw: R_ds / classic O-S of VAR); bg: background VAR-REF in per-spaxel sigma')
print('%4s %-14s %7s %7s %7s %7s %7s %8s %7s %7s %8s' % ('IFU', 'arm', 'S/N', 'peak', 'raw', 's', 'final', 'line',
                                                      'bg_raw', 'bg_s', 'bg_fin'))
for r in sorted(rows, key=lambda r: -r['sn']):
    print('%4d %-14s %7.0f %7s %7.3f %7.3f %7.3f %8.3f %7.2f %7.2f %8.2f' % (
        r['ifu'], r['arm'][:14], r['sn'], '%d,%d' % r['peak'], r['raw'], r['s'], r['fin'], r['line_fin'],
        r['bg_raw'], r['bg_s'], r['bg_fin']))
if rows:
    print('median %26s %7.3f %7.3f %7.3f %8.3f %7.2f %7.2f %8.2f' % ('', *[np.nanmedian([r[k] for r in rows]) for k in
                                                                          ('raw', 's', 'fin', 'line_fin', 'bg_raw',
                                                                           'bg_s', 'bg_fin')]))
