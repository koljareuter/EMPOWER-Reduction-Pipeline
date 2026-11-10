"""Set up a sky-subtraction-only test of one OB on top of a finished hybrid run (e.g. PROCDIR_HYB).

Creates DEST/<OB>/ with
  calib/        link to SRC/<OB>/calib (read only)
  sky/          links to the reconstructed cubes SRC/<OB>/sky/SCI_RECONSTRUCTED_*.fits (read only)
  skysub_det/   copies of SRC/<OB>/skysub_det/SCI_RECONSTRUCTED_*.fits with the calibration bins of OBJECT and
                SKY frame added (ESO PYSPARK SKYSUB DET OCAL/SCAL). The cubes of the old > 20 deg rule do not
                record them, and subtract_sky of this repository falls back to the classic O - S without them.
Then reduce_kmos.py with reconstruct_cubes = False, all detector and calibration steps off, runs only the sky
subtraction and the later steps into DEST/<OB>/science/. Nothing in SRC is changed.

    python scripts/prepare_hybrid_test.py /local_storage/KOLJA/PROCDIR_HYB /local_storage/KOLJA/PROCDIR_OHHP_OFF \
           KMOS.2026-02-10T02:57:00.698 --band IZ
"""
import argparse
import glob
import os
import shutil

import numpy as np
from astropy.io import fits

ROT_KEY = 'HIERARCH ESO OCS ROT NAANGLE'
SKY_KEY = 'HIERARCH ESO PYSPARK SKYSUB DET SKY'
OCAL_KEY = 'HIERARCH ESO PYSPARK SKYSUB DET OCAL'
SCAL_KEY = 'HIERARCH ESO PYSPARK SKYSUB DET SCAL'


def cal_angles(calib, band):
    """Calibration rotator angles of the OB's MASTER_FLAT (as skysub_det.cal_angles); 0..300 deg if none."""
    for f in sorted(glob.glob(os.path.join(calib, 'MASTER_FLAT_{0}*.fits'.format(band)))):
        with fits.open(f) as fh:
            rot = sorted(set(h.header['HIERARCH ESO PRO ROT NAANGLE'] for h in fh[1:]
                             if 'HIERARCH ESO PRO ROT NAANGLE' in h.header))
        if rot:
            return np.array(rot, float)
    return np.arange(0., 360., 60.)


def cal_angle(rot, cal_rot):
    """Nearest calibration angle, wrapped (as obsutils.match_obs_rotator)."""
    return int(cal_rot[np.argmin(np.abs((cal_rot - rot + 540.) % 360 - 180))])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('src', help='finished hybrid run, e.g. /local_storage/KOLJA/PROCDIR_HYB')
    ap.add_argument('dest', help='new test procdir')
    ap.add_argument('ob', nargs='+', help='OB folder name(s)')
    ap.add_argument('--band', default='IZ', help='band of the MASTER_FLAT used for the calibration angles')
    args = ap.parse_args()

    for ob in args.ob:
        src, dest = os.path.join(args.src, ob), os.path.join(args.dest, ob)
        if os.path.isdir(os.path.join(dest, 'science')) and os.listdir(os.path.join(dest, 'science')):
            raise SystemExit('%s/science is not empty: use a fresh dest' % dest)
        os.makedirs(os.path.join(dest, 'sky'), exist_ok=True)
        os.makedirs(os.path.join(dest, 'skysub_det'), exist_ok=True)
        if not os.path.lexists(os.path.join(dest, 'calib')):
            os.symlink(os.path.realpath(os.path.join(src, 'calib')), os.path.join(dest, 'calib'))
        cubes = sorted(glob.glob(os.path.join(src, 'sky', 'SCI_RECONSTRUCTED_*.fits')))
        for f in cubes:
            link = os.path.join(dest, 'sky', os.path.basename(f))
            if not os.path.lexists(link):
                os.symlink(os.path.realpath(f), link)
        cal_rot = cal_angles(os.path.join(src, 'calib'), args.band)
        ncopy = 0
        for f in sorted(glob.glob(os.path.join(src, 'skysub_det', 'SCI_RECONSTRUCTED_*.fits'))):
            out = os.path.join(dest, 'skysub_det', os.path.basename(f))
            shutil.copyfile(os.path.realpath(f), out)       # new mtime: newer than the sky/ cubes, as required
            with fits.open(out, mode='update') as fh:
                h = fh[0].header
                orot = fits.getheader(os.path.join(src, 'sky', os.path.basename(f)), 0)[ROT_KEY]
                srot = fits.getheader(os.path.join(src, 'sky', 'SCI_RECONSTRUCTED_' + h[SKY_KEY]), 0)[ROT_KEY]
                h[OCAL_KEY] = (cal_angle(orot, cal_rot), '[deg] cal. angle of OBJECT')
                h[SCAL_KEY] = (cal_angle(srot, cal_rot), '[deg] cal. angle of SKY')
                print('%s  O %6.1f -> %3d  S %6.1f -> %3d' % (os.path.basename(f), orot, h[OCAL_KEY],
                                                             srot, h[SCAL_KEY]))
            ncopy += 1
        print('%s: %d sky/ links, %d skysub_det/ copies, calibration angles %s' % (
            ob, len(cubes), ncopy, ', '.join('%g' % a for a in cal_rot)))


if __name__ == '__main__':
    main()
