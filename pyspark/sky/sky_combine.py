"""Multi-sky support for the ZAP sky subtraction (sky_subtraction.zap_sky_v2).

The classic pyspark scheme subtracts, for every OBJECT frame, the single
nearest-in-time SKY frame with the same DIT (match_obj_sky).  The helpers here
let zap_sky_v2 subtract a weighted combination of several SKY frames instead:

    match_obj_skies   which SKY frames (same DIT) and which weights, per OBJECT
    combine_sky_cubes NaN-aware weighted combination of one IFU's sky cubes
    scale_sky         per-OH-band rescaling of the combined sky to the OBJECT
    provenance_cards  header cards recording what was subtracted

[cubes] options (all optional; the defaults reproduce the classic behaviour,
because subtract_sky then never calls anything in this module):

    sky_combine = nearest | mean | interp
        nearest  single nearest SKY (classic; with sky_scale != none the
                 nearest SKY is rescaled)
        mean     equal-weight mean of the sky_frames nearest SKY frames
        interp   linear interpolation in time between the SKY frames just
                 before and just after the OBJECT; nearest SKY only (weight 1)
                 for OBJECT frames outside the SKY time range (no extrapolation)
    sky_frames  = N      max number of SKY frames for 'mean' (default: all)
    sky_max_dt  = sec    ignore SKY frames further than this from the OBJECT
                         (default: no limit; the nearest one is always kept)
    sky_nan     = renorm | strict
        renorm   a voxel NaN in some SKY frames is combined from the finite
                 ones (weights renormalised); NaN only if NaN in all of them
        strict   NaN if NaN in any used SKY frame
    sky_scale   = none | bands
        none     the combined sky is subtracted as it is
        bands    its OH lines are rescaled to each OBJECT frame per OH
                 vibrational sub-band of the DRP sky tweak (scale_sky);
                 needed for 'mean': OH lines vary by ~10 % in 10 min
"""
from __future__ import print_function

import os
import numpy as np

__all__ = ["SKY_COMBINE_MODES", "SKY_SCALE_MODES", "parse_multisky_options",
           "match_obj_skies", "combine_sky_cubes", "scale_sky", "background_spectra",
           "band_scales", "scale_curve", "sky_continuum", "provenance_cards"]

SKY_COMBINE_MODES = ('nearest', 'mean', 'interp')
SKY_SCALE_MODES = ('none', 'bands')
SKY_NAN_MODES = ('renorm', 'strict')


def _none_or(value, conv):
    """[cubes] values arrive as strings from set_config ('None', '' -> None)."""
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if value.lower() in ('', 'none', 'all'):
            return None
    return conv(value)


def parse_multisky_options(sky_combine='nearest', sky_frames=None,
                           sky_max_dt=None, sky_scale='none', sky_nan='renorm'):
    """Normalise and validate the multi-sky [cubes] options.

    Returns (sky_combine, sky_frames, sky_max_dt, sky_scale, sky_nan) and a flag
    `classic` that is True when the classic single-nearest-sky path must be used
    unchanged.
    """
    sky_combine = str(sky_combine).strip().lower()
    sky_scale = str(sky_scale).strip().lower()
    sky_nan = str(sky_nan).strip().lower()
    if sky_combine not in SKY_COMBINE_MODES:
        raise ValueError('sky_combine must be one of {0}, got {1!r}'.format(SKY_COMBINE_MODES, sky_combine))
    if sky_scale not in SKY_SCALE_MODES:
        raise ValueError('sky_scale must be one of {0}, got {1!r}'.format(SKY_SCALE_MODES, sky_scale))
    if sky_nan not in SKY_NAN_MODES:
        raise ValueError('sky_nan must be one of {0}, got {1!r}'.format(SKY_NAN_MODES, sky_nan))
    sky_frames = _none_or(sky_frames, int)
    sky_max_dt = _none_or(sky_max_dt, float)
    if sky_frames is not None and sky_frames < 1:
        raise ValueError('sky_frames must be >= 1')
    classic = (sky_combine == 'nearest' and sky_scale == 'none')
    return sky_combine, sky_frames, sky_max_dt, sky_scale, sky_nan, classic


def _sky_weights(t_obj, t_sky, sky_combine, sky_frames=None, sky_max_dt=None):
    """Indices into t_sky and weights for one OBJECT frame.

    t_sky are the times of the SKY frames with the OBJECT's DIT, in the order
    get_obs_exposures returned them (ties are broken by that order, exactly like
    np.argmin in match_obj_sky).
    """
    t_sky = np.asarray(t_sky, dtype=float)
    dt = t_sky - float(t_obj)
    adt = np.fabs(dt)
    inearest = int(np.argmin(adt))           # same choice as match_obj_sky
    if sky_combine == 'nearest':
        return [inearest], [1.0]

    usable = np.ones(len(t_sky), dtype=bool)
    if sky_max_dt is not None:
        usable &= adt <= sky_max_dt
    usable[inearest] = True                  # never end up with no sky

    if sky_combine == 'mean':
        order = [i for i in np.argsort(adt, kind='stable') if usable[i]]
        if sky_frames is not None:
            order = order[:sky_frames]
        return [int(i) for i in order], [1.0 / len(order)] * len(order)

    if sky_combine == 'interp':
        before = [i for i in range(len(t_sky)) if usable[i] and dt[i] <= 0]
        after = [i for i in range(len(t_sky)) if usable[i] and dt[i] > 0]
        if not before or not after:          # outside the SKY time range
            return [inearest], [1.0]
        ib = max(before, key=lambda i: (dt[i], -i))   # latest SKY before
        ia = min(after, key=lambda i: (dt[i], i))     # earliest SKY after
        span = dt[ia] - dt[ib]
        wb = dt[ia] / span
        return [int(ib), int(ia)], [float(wb), float(1.0 - wb)]

    raise ValueError('unknown sky_combine {0!r}'.format(sky_combine))


def match_obj_skies(exposures, sky_combine='mean', sky_frames=None,
                    sky_max_dt=None, logger=None):
    """Like match_obj_sky, but returns for every OBJECT frame a list of
    (sky_file, weight, dt_seconds) tuples instead of a single sky file:

        [(obj_file, [(sky_file, w, dt), ...], dit), ...]

    Only SKY frames with the OBJECT's DIT are considered (as match_obj_sky).
    """
    time_dict = dict([(efile, etime) for efile, _, etime, _ in exposures])
    object_exposures = [(efile, dit) for efile, etype, _, dit in exposures if etype == 'OBJ']
    sky_exposures = [(efile, dit) for efile, etype, _, dit in exposures if etype == 'SKY']

    # sky/ cubes written by kmos_sci_red without pyspark's reconstruct()
    # carry the DATE-OBS of the first frame of the run (reconstruct() repairs it
    # with patch_header); time weights would then be meaningless.
    all_times = [time_dict[e] for e, _ in object_exposures + sky_exposures]
    if len(set(all_times)) == 1 and len(all_times) > 1:
        raise ValueError('all exposures have the same DATE-OBS; sky/ cubes were not '
                         'produced by pyspark reconstruct() (patch_header) - cannot '
                         'weight SKY frames in time')

    out = []
    for obj, odit in object_exposures:
        matched = [efile for efile, dit in sky_exposures if dit == odit]
        if not matched:
            raise ValueError('no SKY frame with DIT={0} for {1}'.format(odit, os.path.basename(obj)))
        t_sky = [time_dict[efile] for efile in matched]
        idx, wts = _sky_weights(time_dict[obj], t_sky, sky_combine,
                                sky_frames=sky_frames, sky_max_dt=sky_max_dt)
        skies = [(matched[i], w, t_sky[i] - time_dict[obj]) for i, w in zip(idx, wts)]
        out.append((obj, skies, odit))
        if logger is not None:
            logger.debug('-- {0}: {1}'.format(os.path.basename(obj), ', '.join(
                '{0} w={1:.3f} dt={2:+.0f}s'.format(os.path.basename(s), w, d) for s, w, d in skies)))
    return out


def combine_sky_cubes(cubes, weights, sky_nan='renorm'):
    """Weighted combination of one IFU's sky cubes (list of (nz, ny, nx) arrays).

    A single cube with weight 1 is returned unchanged, so the multi-sky path with
    one SKY frame gives bit-identical sky-subtracted data to the classic path.
    The result is float32 like the input cubes, so the ZAP step runs with the
    same precision as in the classic path.
    """
    if len(cubes) == 1 and float(weights[0]) == 1.0:
        return cubes[0]
    w = np.asarray(weights, dtype=np.float64)
    stack = np.array(cubes, dtype=np.float64)              # (nsky, nz, ny, nx)
    if sky_nan == 'strict':
        comb = np.tensordot(w, stack, axes=1) / w.sum()
    else:
        finite = np.isfinite(stack)
        wfin = np.where(finite, w[:, None, None, None], 0.0)
        num = np.sum(np.where(finite, stack, 0.0) * wfin, axis=0)
        den = np.sum(wfin, axis=0)
        with np.errstate(invalid='ignore', divide='ignore'):
            comb = np.where(den > 0, num / den, np.nan)
    return comb.astype(np.float32)


# OH sub-band boundaries of the KMOS DRP sky tweak (kmos_priv_sky_tweak.c:931-955), micron.
# Sub-band k (1-based, DRP numbering) is [SUBBANDS[k-1], SUBBANDS[k]).
SUBBANDS = (0.780, 0.824, 0.873, 0.926, 0.964, 1.014, 1.067, 1.125, 1.196, 1.252, 1.289,
            1.400, 1.472, 1.5543, 1.6356, 1.7253, 1.840, 1.9570, 2.095, 2.30, 2.460)
SCALE_MIN_FRAC = 0.3        # background spaxels: faintest 30 % of the object cube (DRP min_frac)
SCALE_MIN_PIX = 20          # sub-bands with <= 20 valid planes are not fitted (as the DRP)
SCALE_MAX_ERR = 0.05        # a sub-band scale with a larger formal error is not used
SCALE_RANGE = (0.5, 2.0)    # ... nor one outside this range


def _wavelengths(header, nz):
    """Wavelength (micron) of every plane from the IFU header (CRVAL3/CDELT3 or CD3_3/CRPIX3)."""
    cdelt = header.get('CDELT3', header.get('CD3_3'))
    return (np.arange(nz) + 1 - header['CRPIX3']) * cdelt + header['CRVAL3']


def background_spectra(ocube, scube, min_frac=SCALE_MIN_FRAC):
    """Median spectra of the object and the sky cube over the background spaxels.

    Background spaxels are the faintest min_frac of the object cube (median along
    wavelength), as kmo_lcorr_create_object_mask in the DRP. A plane is NaN when
    fewer than half of those spaxels are finite in both cubes.
    """
    med = np.nanmedian(ocube, axis=0)
    finite = np.isfinite(med)
    if finite.sum() < 10:
        return None, None, np.zeros(med.shape, bool)
    thr = np.sort(med[finite])[int(finite.sum() * min_frac)]
    mask = finite & (med <= thr)
    o = ocube[:, mask].astype(np.float64)
    s = scube[:, mask].astype(np.float64)
    good = np.isfinite(o) & np.isfinite(s)
    with np.errstate(invalid='ignore'):
        o_bg = np.nanmedian(np.where(good, o, np.nan), axis=1)
        s_bg = np.nanmedian(np.where(good, s, np.nan), axis=1)
    few = good.sum(axis=1) < 0.5 * mask.sum()
    o_bg[few] = np.nan
    s_bg[few] = np.nan
    return o_bg, s_bg, mask


def _line_pixels(s):
    """DRP line definition within one sub-band: sky > median + stdev, widened by +-2 pixels."""
    line = s > np.median(s) + np.std(s, ddof=1)
    return np.convolve(line.astype(float), np.ones(5), mode='same') > 0


def band_scales(lam, o_bg, s_bg):
    """Scale of the sky's OH lines to the object's, per DRP sub-band.

    Within each sub-band: line pixels as in the DRP; continuum of object and sky
    interpolated linearly from the continuum pixels; scale = least-squares fit
    sum(a*b)/sum(b*b) of the continuum-subtracted object (a) to the sky (b), one
    3-sigma clip and refit. A scale is valid only if its formal error is <=
    SCALE_MAX_ERR and it lies within SCALE_RANGE (this replaces the DRP's 2-sigma
    rejection across sub-bands, which also discards genuinely different bands).

    Returns (fits, line_mask): fits {k: dict(scale, err, nline, valid)}, line_mask a
    boolean array over all planes (union of the sub-band line masks).
    """
    ok = np.isfinite(o_bg) & np.isfinite(s_bg)
    line_all = np.zeros(lam.shape, bool)
    fits_ = {}
    for k in range(1, len(SUBBANDS)):
        sel = np.flatnonzero(ok & (lam >= SUBBANDS[k - 1]) & (lam < SUBBANDS[k]))
        if sel.size <= SCALE_MIN_PIX:
            continue
        w, o, s = lam[sel], o_bg[sel], s_bg[sel]
        line = _line_pixels(s)
        line_all[sel[line]] = True
        cont = np.flatnonzero(~line)
        if cont.size < 3:
            continue
        idx = np.arange(cont[0], cont[-1] + 1)
        L = idx[line[idx]]
        if L.size < 5:
            fits_[k] = dict(scale=np.nan, err=np.nan, nline=int(L.size), valid=False)
            continue
        a = o[L] - np.interp(w[L], w[cont], o[cont])
        b = s[L] - np.interp(w[L], w[cont], s[cont])
        keep = np.ones(L.size, bool)
        for _ in range(2):
            bb = np.sum(b[keep] ** 2)
            if bb <= 0:
                break
            sc = np.sum(a[keep] * b[keep]) / bb
            r = a - sc * b
            sd = np.std(r[keep], ddof=1)
            new = np.abs(r - np.median(r[keep])) <= 3 * sd
            if new.sum() < 3 or np.array_equal(new, keep):
                break
            keep = new
        bb = np.sum(b[keep] ** 2)
        if bb <= 0 or keep.sum() < 3:
            fits_[k] = dict(scale=np.nan, err=np.nan, nline=int(L.size), valid=False)
            continue
        sc = np.sum(a[keep] * b[keep]) / bb
        err = np.std(a[keep] - sc * b[keep], ddof=1) / np.sqrt(bb)
        valid = bool(np.isfinite(sc) and np.isfinite(err) and err <= SCALE_MAX_ERR
                     and SCALE_RANGE[0] <= sc <= SCALE_RANGE[1])
        fits_[k] = dict(scale=float(sc), err=float(err), nline=int(keep.sum()), valid=valid)
    return fits_, line_all


def scale_curve(lam, fits_):
    """s(lambda): the sub-band scale inside every valid sub-band, linear across
    sub-bands without a valid scale, constant beyond the outermost valid ones
    (as the DRP). All ones if no sub-band is valid."""
    x, y = [], []
    for k, f in sorted(fits_.items()):
        if f['valid']:
            inband = (lam >= SUBBANDS[k - 1]) & (lam < SUBBANDS[k])
            x.append(lam[inband])
            y.append(np.full(inband.sum(), f['scale']))
    if not x:
        return np.ones(lam.shape)
    x, y = np.concatenate(x), np.concatenate(y)
    order = np.argsort(x)
    return np.interp(lam, x[order], y[order])


def sky_continuum(lam, s_bg, line_mask, width=31):
    """Smooth continuum of the sky background spectrum: OH lines (and NaN planes)
    bridged by linear interpolation, then a running median of `width` planes."""
    from scipy.ndimage import median_filter
    use = np.isfinite(s_bg) & ~line_mask
    if use.sum() < 10:
        return None
    c = np.interp(lam, lam[use], s_bg[use])
    return median_filter(c, size=width, mode='nearest')


def scale_sky(ocube, scube, header, sky_scale='none', logger=None):
    """Rescale the OH lines of the (combined) sky cube to the OBJECT frame.

    sky_scale = 'none'  : the sky cube is returned unchanged.
    sky_scale = 'bands' : per DRP sky-tweak sub-band (Davies 2007; kmos_priv_sky_tweak),
        fitted on the background spaxels of this IFU of this OBJECT frame:
            sky' = S + (s(lambda) - 1) * (S - C(lambda))
        where C is the continuum of the sky's background spectrum. Only the OH
        line part of the sky is rescaled; unlike the DRP no continuum is fitted
        to and subtracted from the object, and band-edge planes are rescaled
        too instead of being left unsubtracted.

    Returns (scaled_sky, cards): cards is a list of (keyword, value, comment) for
    the IFU extension header (the fitted scales).
    """
    if sky_scale == 'none':
        return scube, []
    if sky_scale != 'bands':
        raise ValueError('unknown sky_scale {0!r}'.format(sky_scale))
    nz = scube.shape[0]
    lam = _wavelengths(header, nz)
    o_bg, s_bg, mask = background_spectra(ocube, scube)
    if o_bg is None:
        if logger is not None:
            logger.warning('--- sky scaling: too few valid spaxels, sky not rescaled')
        return scube, [('HIERARCH ESO PYSPARK SKY SCALE NB', 0, 'valid OH sub-band scales')]
    fits_, line_mask = band_scales(lam, o_bg, s_bg)
    valid = {k: f for k, f in fits_.items() if f['valid']}
    cont = sky_continuum(lam, s_bg, line_mask)
    cards = [('HIERARCH ESO PYSPARK SKY SCALE NB', len(valid), 'valid OH sub-band scales'),
             ('HIERARCH ESO PYSPARK SKY SCALE NBG', int(mask.sum()), 'background spaxels of the fit')]
    for k, f in sorted(valid.items()):
        cards.append(('HIERARCH ESO PYSPARK SKY SCALE B{0}'.format(k), round(f['scale'], 5),
                      '{0:.4f}-{1:.4f} um +-{2:.4f}'.format(SUBBANDS[k - 1], SUBBANDS[k], f['err'])))
    if logger is not None:
        logger.info('--- sky scales per OH sub-band: ' + ', '.join(
            '{0}:{1:.3f}{2}'.format(k, f['scale'], '' if f['valid'] else '(rej)')
            for k, f in sorted(fits_.items())))
    if not valid or cont is None:
        if logger is not None:
            logger.warning('--- sky scaling: no valid sub-band scale, sky not rescaled')
        return scube, cards
    s_lam = scale_curve(lam, fits_)
    S = scube.astype(np.float64)
    scaled = S + (s_lam - 1.0)[:, None, None] * (S - cont[:, None, None])
    return scaled.astype(np.float32), cards


def _frame_id(path):
    """'.../SCI_RECONSTRUCTED_dc_KMOS.2026-02-12T06:08:34.653.fits' -> 'KMOS.2026-02-12T06:08:34.653'"""
    name = os.path.basename(path)
    if name.endswith('.fits'):
        name = name[:-5]
    return name.split('_')[-1]


def provenance_cards(skies, sky_combine, sky_scale, sky_nan):
    """Primary-header cards for the *_s.fits file of one OBJECT frame."""
    w = np.array([s[1] for s in skies], dtype=float)
    cards = [('HIERARCH ESO PYSPARK SKY COMBINE', sky_combine, 'sky frame combination'),
             ('HIERARCH ESO PYSPARK SKY SCALE', sky_scale, 'sky rescaling to object'),
             ('HIERARCH ESO PYSPARK SKY NAN', sky_nan, 'NaN policy of the combination'),
             ('HIERARCH ESO PYSPARK SKY NFRAMES', len(skies), 'number of SKY frames used'),
             ('HIERARCH ESO PYSPARK SKY VARFAC', float(np.sum(w ** 2) / np.sum(w) ** 2),
              'sky variance rel. to one SKY frame')]
    for i, (path, wi, dt) in enumerate(skies, 1):
        cards += [('HIERARCH ESO PYSPARK SKY S{0} ID'.format(i), _frame_id(path), 'SKY frame'),
                  ('HIERARCH ESO PYSPARK SKY S{0} W'.format(i), round(float(wi), 6), 'weight'),
                  ('HIERARCH ESO PYSPARK SKY S{0} DT'.format(i), round(float(dt), 1), '[s] t_sky - t_obj')]
    return cards
