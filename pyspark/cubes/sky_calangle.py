"""[cubes] sky_cal_angle = own | object : calibration angle of the SKY frames (docs/noise-improvements-2026-10.md).

kmos_sci_red picks MASTER_FLAT, LCAL and (interpolated) XCAL/YCAL by the rotator angle in each frame's own
header ('ESO OCS ROT NAANGLE', kmo_functions.c:2051), from six calibration angles 0..300 deg. pyspark
reconstructs OBJECT and SKY frames separately and subtracts them in cube space, so whenever the SKY frame lies
in another 60-deg calibration bin than its OBJECT frames (41 % of IZ, 81 % of H, 84 % of YJ object time), the
two are reconstructed with different flats and slitlet geometry, and O - S picks up the difference. The
DRP's own sky subtraction on the detector uses the object's calibrations for both frames.

'object': every SKY frame is reconstructed from a copy whose NAANGLE is the circular mean of the angles of the
OBJECT frames it is subtracted from (pyspark's match_obj_sky rule: nearest DATE-OBS among SKY frames with the
same DIT); afterwards the true angle is written back into the cube header and the angle used is recorded.
Measured: 5x5 aperture noise -2.6..-3.3 % per frame on the worst IZ OB, -4..-8.5 % per frame and -5..-11 % in
6-frame stacks on an H OB. 'own' (default) is the current behaviour.
"""
from __future__ import print_function

import os
import shutil
import time

import numpy as np
from astropy.io import fits

__all__ = ["sky_angle_plan", "write_sky_copies", "restore_sky_header"]

ROT = "HIERARCH ESO OCS ROT NAANGLE"
ORIG = "HIERARCH ESO PYSPARK SKY ROT ORIG"
USED = "HIERARCH ESO PYSPARK SKY CALANGLE"


def _info(path):
    h = fits.getheader(path)
    sky = any(str(h.get("HIERARCH ESO OCS ARM%d NAME" % i, "")).endswith("_SKY") for i in (1, 2, 3))
    t = time.mktime(time.strptime(h["DATE-OBS"][:19], "%Y-%m-%dT%H:%M:%S"))
    return dict(path=path, sky=sky, t=t, dit=float(h["HIERARCH ESO DET SEQ1 DIT"]), rot=float(h[ROT]),
                band=str(h.get("HIERARCH ESO INS GRAT1 ID", "")).strip())


def _circ_mean(deg):
    a = np.radians(np.asarray(deg, float))
    return float(np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean())))


def sky_angle_plan(frames, obsband=None):
    """{sky frame: (true angle, angle to use, [object frames it serves])} for the frames of one OB."""
    inf = [_info(f) for f in frames]
    if obsband:
        inf = [i for i in inf if i["band"] == obsband]
    objs = [i for i in inf if not i["sky"]]
    skies = [i for i in inf if i["sky"]]
    served = dict((s["path"], []) for s in skies)
    for o in objs:
        cand = [s for s in skies if s["dit"] == o["dit"]]
        if cand:
            served[min(cand, key=lambda s: abs(s["t"] - o["t"]))["path"]].append(o)
    plan = {}
    for s in skies:
        use = served[s["path"]]
        if not use and objs:                         # a sky no object uses: angle of the nearest object
            use = [min(objs, key=lambda o: abs(o["t"] - s["t"]))]
        ang = _circ_mean([o["rot"] for o in use]) if use else s["rot"]
        plan[s["path"]] = (s["rot"], ang, [o["path"] for o in use])
    return plan


def write_sky_copies(frames, workdir, obsband=None, logger=None):
    """Copies of the SKY frames with the planned NAANGLE in workdir (same file names, so the kmos_sci_red
    output names do not change). Returns (frame list for the SOF, plan)."""
    plan = sky_angle_plan(frames, obsband)
    if not os.path.isdir(workdir):
        os.makedirs(workdir)
    out = []
    for f in frames:
        if f not in plan:
            out.append(f)
            continue
        rot, ang, objs = plan[f]
        dst = os.path.join(workdir, os.path.basename(f))
        shutil.copyfile(f, dst)
        with fits.open(dst, mode="update") as h:
            h[0].header[ORIG] = (rot, "true OCS ROT NAANGLE of this SKY frame")
            h[0].header[ROT] = ang
        if logger is not None:
            logger.info("-- sky_cal_angle=object: {0} NAANGLE {1:.2f} -> {2:.2f} ({3} object frames)".format(
                os.path.basename(f), rot, ang, len(objs)))
        out.append(dst)
    return out, plan


def restore_sky_header(cube, true_angle, used_angle):
    """Write the true rotator angle back into a reconstructed SKY cube and record the angle used."""
    with fits.open(cube, mode="update") as h:
        h[0].header[ROT] = true_angle
        h[0].header[USED] = (round(float(used_angle), 3), "rotator angle used to select calibrations")
        h[0].header.remove(ORIG, ignore_missing=True)
        h[0].header["HISTORY"] = "pyspark sky_cal_angle=object: calibrations at {0:.3f} deg (true {1:.3f})".format(
            used_angle, true_angle)
