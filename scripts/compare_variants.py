"""Compare the final science frames (science/*_s_i_bkg2d_rv.fits) of test variants with a reference run, per OB.

Per OB and IFU the OBJECT frames are averaged in IFU coordinates (this is where sky-subtraction residuals add up;
the dither offsets are small), continuum removed along wavelength (31-plane running median), and measured in
the planes between OH lines and on OH lines (from the median sky spectrum of the reference sky/ cube):
  spx    per-spaxel noise (interior spaxels)
  box    3x3-aperture noise (four boxes around the IFU centre, where the targets are; robust over planes)
  cm     per-plane IFU-wide offset / spx (0.1 ~ noise only for ~100 spaxels)
  kept   signal kept in the central r = 2.8 spx aperture: slope of variant vs reference over the planes
         where the reference aperture spectrum is > 4 sigma (emission lines, continuum features); 1 = unchanged
Printed: median over the IFUs with data of spx, box (variant / reference), cm, kept.

    python scripts/compare_variants.py --band IZ --ref /local_storage/KOLJA/PROCDIR_CLEAN \
        --var all=/local_storage/KOLJA/PROCDIR_RULEALL --var all+bkgp=/local_storage/KOLJA/PROCDIR_RULEALL_BKGP \
        --obs KMOS.2026-01-13T02:54:58.606 KMOS.2026-01-14T05:23:13.608
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
p.add_argument('--var', action='append', default=[], help='name=procdir')
p.add_argument('--obs', nargs='+', required=True)
p.add_argument('--band', default='IZ')
p.add_argument('--suffix', default='_s_i_bkg2d_rv.fits')
a = p.parse_args()
runs = [('ref', a.ref)] + [tuple(v.split('=', 1)) for v in a.var]
LAM = {'IZ': [(0.80, 0.92), (0.99, 1.05)], 'YJ': [(1.02, 1.10), (1.16, 1.33)], 'H': [(1.50, 1.75)]}[a.band]


def rmad(x, axis=None):
    m = np.nanmedian(x, axis=axis, keepdims=True)
    return 1.4826 * np.nanmedian(np.abs(x - m), axis=axis)


def frames(procdir, ob):
    out = {}
    for f in glob.glob(os.path.join(procdir, ob, 'science', 'SCI_RECONSTRUCTED_*' + a.suffix)):
        h = fits.getheader(f)
        if h.get('HIERARCH ESO INS FILT1 ID', '').strip() == a.band:
            out[os.path.basename(f)] = f
    return out


def stack(files, ifu):
    cubes = []
    for f in files:
        with fits.open(f) as h:
            if ifu < len(h) and h[ifu].data is not None:
                cubes.append(h[ifu].data.astype(float))
                hd = h[ifu].header
    if not cubes:
        return None, None
    lam = hd['CRVAL3'] + hd['CDELT3'] * (np.arange(cubes[0].shape[0]) + 1 - hd['CRPIX3'])
    return np.nanmean(cubes, axis=0), lam


for ob in a.obs:
    fr = {name: frames(pd, ob) for name, pd in runs}
    common = sorted(set.intersection(*[set(v) for v in fr.values()]))
    hyb = {name: sum(1 for n in common if fits.getheader(fr[name][n]).get('HIERARCH ESO PYSPARK SKY HYBRID', False))
           for name, _ in runs}
    bkgp = {name: sum(1 for n in common if fits.getheader(fr[name][n]).get('HIERARCH ESO PYSPARK BKG PLANE', False))
            for name, _ in runs}
    print('\n%s: %d OBJECT frames in all runs; hybrid frames %s; bkg_plane frames %s' % (
        ob, len(common), ', '.join('%s %d' % kv for kv in hyb.items()), ', '.join('%s %d' % kv for kv in bkgp.items())))
    if not common:
        continue
    skyf = os.path.join(a.ref, ob, 'sky', common[0].replace(a.suffix, '.fits'))
    rows = {name: [] for name, _ in runs}
    for ifu in range(1, 25):
        st = {name: stack([fr[name][n] for n in common], ifu) for name, _ in runs}
        if any(v[0] is None for v in st.values()):
            continue
        d0, lam = st['ref']
        if os.path.isfile(skyf):
            with fits.open(skyf) as h:
                sc = h[ifu].data
            sky = np.nanmedian(sc.reshape(sc.shape[0], -1), axis=1) if sc is not None else None
        else:
            sky = None
        if sky is None:
            continue
        ok = np.zeros(len(lam), bool)
        for lo, hi in LAM:
            ok |= (lam > lo) & (lam < hi)
        ok &= np.isfinite(sky)
        run = median_filter(np.nan_to_num(sky), 101)
        oh, free = ok & (sky > 1.5 * run), ok & (sky < 1.1 * run)
        ny, nx = d0.shape[1:]
        yy, xx = np.mgrid[:ny, :nx]
        ap = np.hypot(yy - (ny - 1) / 2., xx - (nx - 1) / 2.) <= 2.8
        hp = {}
        for name, (d, _) in st.items():
            h = d - median_filter(np.nan_to_num(d), size=(31, 1, 1))
            h[~np.isfinite(d)] = np.nan
            hp[name] = h
        a0 = np.nansum(hp['ref'][:, ap], axis=1)
        sig0 = rmad(a0[free])
        strong = ok & (a0 > 4 * sig0)
        for name in hp:
            h = hp[name]
            inner = h[:, 2:-2, 2:-2]
            flat = inner.reshape(len(lam), -1)
            spx = rmad(flat, axis=1)
            cm = np.nanmedian(flat, axis=1)
            # four 3x3 boxes around the IFU centre, inside the object mask of bkg_plane (r = 4 spx), so that they are
            # not part of its estimate (a box inside the estimate would get ~5 % lower noise by arithmetic alone)
            c0y, c0x = ny // 2, nx // 2
            box = np.array([np.nansum(h[:, i:i + 3, j:j + 3], axis=(1, 2))
                            for i in (c0y - 3, c0y) for j in (c0x - 3, c0x)])
            r = {}
            for m, mn in [(free, 'fr'), (oh, 'oh')]:
                s = np.nanmedian(spx[m])
                r['spx_' + mn] = s
                r['box_' + mn] = rmad(box[:, m].ravel())
                r['cm_' + mn] = rmad(cm[m]) / s
            av = np.nansum(h[:, ap], axis=1)
            r['kept'] = np.sum(av[strong] * a0[strong]) / np.sum(a0[strong] ** 2) if strong.sum() >= 5 else np.nan
            r['nstrong'] = int(strong.sum())
            rows[name].append(r)
    if not rows['ref']:
        print('  no IFUs with data')
        continue
    print('  %-12s %5s %9s %9s %9s %9s %7s %7s %7s' % ('run', 'IFUs', 'spx free', 'spx OH', 'box free', 'box OH',
                                                     'cm free', 'cm OH', 'kept'))
    for name, _ in runs:
        rr, r0 = rows[name], rows['ref']
        rel = lambda k: np.nanmedian([x[k] / y[k] for x, y in zip(rr, r0)])
        print('  %-12s %5d %9.3f %9.3f %9.3f %9.3f %7.3f %7.3f %7.3f' % (
            name, len(rr), rel('spx_fr'), rel('spx_oh'), rel('box_fr'), rel('box_oh'),
            np.nanmedian([x['cm_fr'] for x in rr]), np.nanmedian([x['cm_oh'] for x in rr]),
            np.nanmedian([x['kept'] for x in rr])))
print('\nspx/box: variant / reference (median over IFUs); cm: absolute; kept: signal slope vs reference.')
