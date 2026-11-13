"""Set up a run from the reconstruction on, on top of a finished run (e.g. PROCDIR_CLEAN), for options that act in
the reconstruction (hybrid_rule) or later (bkg_plane).

Creates DEST/<OB>/ with
  calib/                 link to SRC/<OB>/calib (read only)
  dcr_KMOS*.fits         the detector-corrected SCIENCE frames of the band: copies (default), or with --link
                         symbolic links (kmos_sci_red and pyspark only read them; no step with the detector and
                         calibration steps off writes to them)
Then reduce_kmos.py with all detector and calibration steps off and reconstruct_cubes = True runs the
reconstruction (with skysub_det), sky subtraction and the later steps into DEST/<OB>/. Nothing in SRC is changed.

    python scripts/prepare_rule_test.py /local_storage/KOLJA/PROCDIR_CLEAN /local_storage/KOLJA/PROCDIR_RULEALL \
           KMOS.2026-01-13T02:54:58.606 KMOS.2026-01-14T05:23:13.608 --band IZ --prefix dcr
    python scripts/prepare_rule_test.py /local_storage/KOLJA/PROCDIR_CLEAN /local_storage/KOLJA/PROCDIR_IZALL \
           --cfg ~/empower_clean_cfg/empower_rerun_IZ_all.cfg --band IZ --prefix dcr --link
"""
import argparse
import configparser
import glob
import os
import re
import shutil

from astropy.io import fits

p = argparse.ArgumentParser()
p.add_argument('src')
p.add_argument('dest')
p.add_argument('obs', nargs='*', help='OB folders (or --cfg)')
p.add_argument('--cfg', help='take the OB folders from [pathing] subdirs of this config')
p.add_argument('--band', default='IZ')
p.add_argument('--prefix', default='dcr')
p.add_argument('--link', action='store_true', help='link the frames instead of copying them')
a = p.parse_args()

obs = list(a.obs)
if a.cfg:
    cp = configparser.ConfigParser(interpolation=None)
    cp.read(os.path.expanduser(a.cfg))
    obs += [s for s in re.split(r'[,\s]+', cp.get('pathing', 'subdirs').strip()) if s]
if not obs:
    raise SystemExit('no OB folders given (arguments or --cfg)')
if os.path.abspath(a.src) == os.path.abspath(a.dest):
    raise SystemExit('DEST must differ from SRC')

# check everything first, so that nothing is half set up
plan = []
for ob in obs:
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
    plan.append((ob, s, d, frames))

for ob, s, d, frames in plan:
    os.makedirs(d)
    os.symlink(os.path.realpath(os.path.join(s, 'calib')), os.path.join(d, 'calib'))
    for f in frames:
        if a.link:
            os.symlink(os.path.realpath(f), os.path.join(d, os.path.basename(f)))
        else:
            shutil.copy2(f, d)
    print('{0}: calib/ linked, {1} {2} SCIENCE frames {3}'.format(ob, len(frames), a.band,
                                                                  'linked' if a.link else 'copied'))
print('{0} OBs prepared in {1}'.format(len(plan), a.dest))
