"""DRP detector-level sky subtraction for the hybrid sky subtraction (docs/hybrid-sky.md).

For every OBJECT frame whose nearest SKY frame (the one subtract_sky uses: same DIT, closest DATE-OBS) was
taken at a rotator angle more than hybrid_min_dangle away, kmos_sci_red is run with -no_subtract=FALSE
-sky_tweak=FALSE: the SKY frame is subtracted on the detector and reconstructed with the OBJECT's
calibrations and OH wavelength correction. The cubes go to <OB>/skysub_det/ and are combined with the
classic O - S by subtract_sky(hybrid_sky=True). The DRP's own object/sky pairing (obj_sky_table.txt) is
checked against the pyspark pairing; frames where they differ are not used (classic O - S instead).
"""
from __future__ import print_function
import glob
import os
import time

from astropy.io import fits

from ..utils.fileutils import patch_header, dateobsfmt
from ..utils.obsutils import run_spark
from ..utils.setuputils import id_gen

__all__ = ["skysub_det", "pair_frames", "read_obj_sky_table", "SKY_KEY", "DANGLE_KEY"]

ROT_KEY = 'HIERARCH ESO OCS ROT NAANGLE'
SKY_KEY = 'HIERARCH ESO PYSPARK SKYSUB DET SKY'
DANGLE_KEY = 'HIERARCH ESO PYSPARK SKYSUB DET DANGLE'
HDR_KEYS = ['DATE', 'DATE-OBS', 'OBJECT', 'RA', 'DEC', 'MJD-OBS', 'UTC', 'LST']


def _is_sky(hdr):
    return any(str(hdr.get('HIERARCH ESO OCS ARM{0} NAME'.format(i), '')).endswith('_SKY') for i in (1, 2, 3))


def pair_frames(frames):
    """Nearest SKY frame of every OBJECT frame, as match_obj_sky does it (same DIT, closest DATE-OBS).

    Returns ({object frame: (sky frame, |rotator angle difference| in deg)}, [all SKY frames]); OBJECT
    frames without a SKY frame of the same DIT are left out.
    """
    info = {}
    for f in frames:
        h = fits.getheader(f)
        info[f] = (_is_sky(h), time.mktime(time.strptime(h['DATE-OBS'], dateobsfmt)),
                   float(h['HIERARCH ESO DET SEQ1 DIT']), float(h[ROT_KEY]))
    skies = [f for f in frames if info[f][0]]
    pairs = {}
    for o in frames:
        if info[o][0]:
            continue
        cand = [s for s in skies if info[s][2] == info[o][2]]
        if not cand:
            continue
        s = min(cand, key=lambda f: abs(info[f][1] - info[o][1]))
        pairs[o] = (s, abs((info[s][3] - info[o][3] + 540.) % 360. - 180.))
    return pairs, skies


def read_obj_sky_table(path):
    """{object frame basename: set of SKY frame basenames over its IFUs} from the DRP's obj_sky_table.txt."""
    index, out, cur = {}, {}, None
    with open(path) as fh:
        for line in fh:
            s = line.split()
            if len(s) == 3 and s[0] == '#' and s[1].endswith(':'):
                index[s[1][:-1]] = os.path.basename(s[2])
            elif len(s) >= 4 and s[0] == 'frame' and s[1] == '#':
                cur = os.path.basename(s[3])
            elif cur is not None and line.strip().startswith('sky in #:'):
                ids = line.split(':', 1)[1].split()
                out[cur] = set(index.get(i, '?' + i) for i in ids if i.isdigit())
                cur = None
    return out


def skysub_det(frames, cal_lines, outputdir, subdir, min_dangle=20., esorex_opts='', env=None, logger=None):
    """Write skysub_det/SCI_RECONSTRUCTED_<frame> for the OBJECT frames with a rotator mismatch.

    frames    : the original science frames of this OB and band (not angle-edited SKY copies)
    cal_lines : the calibration lines of the reconstruction SOF ('<file> <TAG>')
    Returns the number of cubes written, or -1 if kmos_sci_red failed.
    """
    min_dangle = float(min_dangle)
    det_dir = os.path.abspath(os.path.join(outputdir, subdir, 'skysub_det'))     # the run below changes the cwd
    pairs, skies = pair_frames(frames)
    use = dict((o, p) for o, p in pairs.items() if p[1] > min_dangle)

    #cubes left over from an earlier run with other frames/settings would be picked up by subtract_sky
    for o in pairs:
        old = os.path.join(det_dir, 'SCI_RECONSTRUCTED_{0}'.format(os.path.basename(o)))
        if os.path.isfile(old):
            os.remove(old)
    if logger is not None:
        logger.info('-- Hybrid sky: {0} of {1} OBJECT frames have their SKY frame > {2:g} deg away'.format(
            len(use), len(pairs), min_dangle))
    if not use:
        return 0

    #own working directory: kmos_sci_red writes its products and obj_sky_table.txt into the cwd
    work = os.path.join(det_dir, 'work_{0}'.format(id_gen()))
    os.makedirs(work)
    sof = os.path.join(work, 'skysub_det.sof')
    with open(sof, 'wt') as fh:
        for f in sorted(use) + skies:     # all SKY frames, so that the DRP can pick the same nearest one
            fh.write('{0} SCIENCE\n'.format(os.path.abspath(f)))
        for line in cal_lines:
            fname, tag = line.rsplit(None, 1)
            fh.write('{0} {1}\n'.format(os.path.abspath(fname), tag))

    args = ('-background=FALSE -tbsub=TRUE -no_subtract=FALSE -sky_tweak=FALSE -lcmethod=SLICES_MEDIAN '
            '-skip_sky_oh_align=TRUE -no_combine=TRUE')
    opts = '{0} --output-dir={1} --log-dir={1}'.format(esorex_opts, work).strip()
    cwd = os.getcwd()
    os.chdir(work)
    try:
        ecode = run_spark('kmos_sci_red', args=args, sof=sof, logger=logger, esorex_opts=opts, env=env)
    finally:
        os.chdir(cwd)
    if ecode != 0:
        if logger is not None:
            logger.warning('-- Hybrid sky: kmos_sci_red failed (exit {0}), classic O - S for this OB'.format(ecode))
        _clear(work)
        return -1

    tables = glob.glob(os.path.join(work, 'obj_sky_table*.txt'))
    drp_pairs = read_obj_sky_table(tables[0]) if tables else {}
    nwritten = 0
    for o, (s, dangle) in sorted(use.items()):
        cube = os.path.join(work, 'SCI_RECONSTRUCTED_{0}'.format(os.path.basename(o)))
        drp_sky = drp_pairs.get(os.path.basename(o))
        if not os.path.isfile(cube) or drp_sky != {os.path.basename(s)}:
            if logger is not None:
                logger.warning('-- Hybrid sky: {0}: DRP sky {1} != pyspark sky {2} (or no cube), classic O - S'.format(
                    os.path.basename(o), sorted(drp_sky) if drp_sky else None, os.path.basename(s)))
            continue
        patch_header(cube, HDR_KEYS, o)
        with fits.open(cube, mode='update') as fh:
            fh[0].header[SKY_KEY] = (os.path.basename(s), 'SKY frame subtracted on the detector')
            fh[0].header[DANGLE_KEY] = (round(dangle, 2), '[deg] rotator angle SKY - OBJECT')
        os.replace(cube, os.path.join(det_dir, os.path.basename(cube)))
        nwritten += 1
        if logger is not None:
            logger.info('--- {0} - {1} (rotator {2:.1f} deg apart) -> skysub_det/'.format(
                os.path.basename(o), os.path.basename(s), dangle))

    _clear(work)
    return nwritten


def _clear(work):
    """Remove the working directory of the detector-level run (its remaining products and logs are not used)."""
    for f in glob.glob(os.path.join(work, '*')):
        os.remove(f)
    os.rmdir(work)
