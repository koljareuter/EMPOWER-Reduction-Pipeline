#!/usr/bin/env python
"""Pick clean kmos cubes_combination jobs out of the EDPS job store.

EDPS keeps every job in <base_dir>/KMOS/cubes_combination/<uuid>/.  When EDPS was
pointed at all of PROCDIR, it also combined sky/ cubes and intermediate stages
(_s, _s_i, _s_i_bkg2d).  This script reports, per job, the input stage(s), band,
target list and date, and with --copy collects the products of the jobs whose
inputs are ALL *_s_i_bkg2d_rv.fits (one band) into OUT/<job uuid>/, with the
combined cubes named DATACUBE_COMBINED_<target list>_<target id>.fits.

Inputs are read from any text file in the job folder (sof, json, log ...) and
from symlinks in it.

Usage:
    python scripts/edps_pick_jobs.py --base <EDPS base_dir>/KMOS/cubes_combination --since 2026-11-09   # report only
    python scripts/edps_pick_jobs.py --base ... --band IZ --since 2026-11-09 --copy /path/to/res_IZ
"""
import argparse
import glob
import os
import re
import shutil
from datetime import datetime

from astropy.io import fits

STAGE_RE = re.compile(r"SCI_RECONSTRUCTED_[\w.:-]*?KMOS\.\d{4}-\d\d-\d\dT[\d:.-]+?\d{3}(_[a-z0-9_]+)?\.fits")


def job_inputs(job):
    names = set()
    for f in glob.glob(os.path.join(job, "**", "*"), recursive=True):
        if os.path.islink(f):
            names.add(os.path.basename(os.path.realpath(f)))
            continue
        if not os.path.isfile(f) or f.endswith(".fits") or os.path.getsize(f) > 50e6:
            continue
        try:
            txt = open(f, errors="ignore").read()
        except OSError:
            continue
        names.update(m.group(0) for m in STAGE_RE.finditer(txt))
    return names


def stage(name):
    m = STAGE_RE.search(name)
    return (m.group(1) or "(none)") if m else "?"


def products(job):
    out = []
    for f in sorted(glob.glob(os.path.join(job, "*.fits"))):
        try:
            h = fits.getheader(f, 0)
        except Exception:
            continue
        out.append((f, h))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=os.path.expanduser("~/EDPS_data/KMOS/cubes_combination"))
    ap.add_argument("--since", default="2026-10-08", help="only jobs modified on/after this date")
    ap.add_argument("--band", help="only jobs of this band (e.g. IZ)")
    ap.add_argument("--copy", metavar="OUT", help="copy products of clean jobs to OUT/<uuid>/")
    a = ap.parse_args()

    since = datetime.fromisoformat(a.since).timestamp()
    jobs = [j for j in sorted(glob.glob(os.path.join(a.base, "*"))) if os.path.isdir(j)
            and os.path.getmtime(j) >= since]
    print("%d jobs since %s in %s" % (len(jobs), a.since, a.base))
    print("%-9s %-16s %-4s %-28s %5s %6s  %s" % ("job", "modified", "band", "target list", "n_in", "n_fits", "input stages"))
    n_clean = n_copied = 0
    first = True
    for job in jobs:
        prods = products(job)
        hdr = prods[0][1] if prods else {}
        band = str(hdr.get("HIERARCH ESO INS GRAT1 ID", "?")).strip()
        targ = str(hdr.get("HIERARCH ESO OBS TARG NAME", "?")).strip()
        if a.band and band != a.band:
            continue
        ins = job_inputs(job)
        stages = sorted({stage(n) for n in ins})
        clean = stages == ["_s_i_bkg2d_rv"]
        n_clean += clean
        print("%-9s %-16s %-4s %-28s %5d %6d  %s%s" % (
            os.path.basename(job)[:8], datetime.fromtimestamp(os.path.getmtime(job)).strftime("%Y-%m-%d %H:%M"),
            band, targ[:28], len(ins), len(prods), ",".join(stages) or "NO INPUT LIST FOUND",
            "   <- clean" if clean else ""))
        if first:                                    # show what a job folder holds, once
            print("    files in this job:", ", ".join(sorted(os.listdir(job))[:12]), "...")
            first = False
        if a.copy and clean:
            dst = os.path.join(a.copy, os.path.basename(job))
            os.makedirs(dst, exist_ok=True)
            for f, h in prods:
                name = os.path.basename(f)
                catg = str(h.get("HIERARCH ESO PRO CATG", ""))
                suf = str(h.get("HIERARCH ESO PRO REFLEX SUFFIX", "")).strip()
                t = str(h.get("HIERARCH ESO OBS TARG NAME", "")).strip()
                m = re.match(r"COMBINE_SCI_RECONSTRUCTED_(.+)\.fits$", name)
                if m and t:                         # EDPS-internal name → name used in the archive export
                    name = "DATACUBE_COMBINED_%s_%s.fits" % (t, m.group(1))
                elif "COMBINED_CUBE" in catg and suf and not name.startswith("DATACUBE_COMBINED_"):
                    name = "DATACUBE_COMBINED_%s_%s.fits" % (t, suf)
                shutil.copy2(f, os.path.join(dst, name))
            n_copied += 1
    print("\n%d clean jobs (all inputs *_s_i_bkg2d_rv)" % n_clean)
    if a.copy:
        print("%d jobs copied to %s" % (n_copied, a.copy))


if __name__ == "__main__":
    main()
