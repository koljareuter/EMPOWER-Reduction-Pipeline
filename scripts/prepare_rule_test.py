"""Set up a test from the reconstruction on, on top of a finished run (e.g. PROCDIR_CLEAN), for options that act in
the reconstruction (hybrid_rule) or later (bkg_plane).

Creates DEST/<OB>/ with
  calib/                 link to SRC/<OB>/calib (read only)
  dcr_KMOS*.fits         copies of the detector-corrected SCIENCE frames of the band (kmos_sci_red reads them;
                         copies, so that nothing can touch the frames of SRC)
Then reduce_kmos.py with all detector and calibration steps off and reconstruct_cubes = True runs the
reconstruction (with skysub_det), sky subtraction and the later steps into DEST/<OB>/. Nothing in SRC is changed.

    python scripts/prepare_rule_test.py /local_storage/KOLJA/PROCDIR_CLEAN /local_storage/KOLJA/PROCDIR_RULEALL \
           KMOS.2026-01-13T02:54:58.606 KMOS.2026-01-14T05:23:13.608 --band IZ --prefix dcr
"""
import argparse
import glob
import os
import shutil

from astropy.io import fits

p = argparse.ArgumentParser()
p.add_argument('src')
p.add_argument('dest')
p.add_argument('obs', nargs='+')
p.add_argument('--band', default='IZ')
p.add_argument('--prefix', default='dcr')
a = p.parse_args()

if os.path.abspath(a.src) == os.path.abspath(a.dest):
    raise SystemExit('DEST must differ from SRC')
for ob in a.obs:
    s, d = os.path.join(a.src, ob), os.path.join(a.dest, ob)
    if not os.path.isdir(os.path.join(s, 'calib')):
        raise SystemExit('{0}: no calib/'.format(s))
    if os.path.exists(d):
        raise SystemExit('{0} exists already; remove it first'.format(d))
    frames = []
    for f in sorted(glob.glob(os.path.join(s, '{0}_KMOS*.fits'.format(a.prefix)))):
        h = fits.getheader(f)
        if h.get('HIERARCH ESO DPR CATG') == 'SCIENCE' and h.get('HIERARCH ESO INS FILT1 ID', '').strip() == a.band:
            frames.append(f)
    if not frames:
        raise SystemExit('{0}: no {1}_KMOS*.fits SCIENCE frames in {2} (deleted after the run?)'.format(
            ob, a.prefix, a.band))
    os.makedirs(d)
    os.symlink(os.path.abspath(os.path.join(s, 'calib')), os.path.join(d, 'calib'))
    for f in frames:
        shutil.copy2(f, d)
    print('{0}: calib/ linked, {1} {2} SCIENCE frames copied'.format(ob, len(frames), a.band))
