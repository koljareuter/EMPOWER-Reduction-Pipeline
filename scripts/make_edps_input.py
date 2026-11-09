#!/usr/bin/env python
"""Build a clean EDPS input folder for one band: symlinks to the final cubes only.

EDPS must never be pointed at PROCDIR itself — it then also combines the sky/ cubes
(no sky subtraction) and every intermediate stage (_s, _s_i, _s_i_bkg2d, _rv_tell).
This script links exactly the PROCDIR/<OB>/science/SCI_RECONSTRUCTED_*_s_i_bkg2d_rv.fits
of one band into OUT:

  * adds links that are missing, keeps existing ones
  * removes links in OUT that are dangling or point to another band / stage
  * reports the number of files per OB and the file dates (re-run check:
    --min-date flags files older than that date, e.g. not yet re-reduced)

Usage:
    python scripts/make_edps_input.py IZ --procdir /path/to/kmos_proc --out /path/to/edps_in_IZ
    python scripts/make_edps_input.py YJ --procdir /path/to/kmos_proc --out /path/to/edps_in_YJ --min-date 2026-11-09
    python scripts/make_edps_input.py YJ --procdir ... --out ... --expected expected_frames.csv --search /path/to/runs
        (compare with the frames of a previous combination; missing final cubes
         are searched for under --search)
Exit code 1 if files older than --min-date were linked.
"""
import argparse
import collections
import csv
import glob
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import date

from astropy.io import fits

FINAL = "_s_i_bkg2d_rv.fits"


def band_of(path):
    try:
        return (fits.getheader(path, 0).get("HIERARCH ESO INS GRAT1 ID") or "").strip()
    except Exception:
        return ""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("band", help="IZ, YJ, H, HK or K")
    ap.add_argument("--procdir", required=True, help="procdir of the reduction (holds the <OB>/science/ folders)")
    ap.add_argument("--out", required=True, help="EDPS input folder for this band (created if missing)")
    ap.add_argument("--min-date", help="flag files modified before this date (YYYY-MM-DD)")
    ap.add_argument("--expected", help="CSV band,target_list,frame of a previous combination")
    ap.add_argument("--search", help="where to look for missing final cubes (with --expected; default: --procdir)")
    a = ap.parse_args()
    a.search = a.search or a.procdir
    out = a.out
    os.makedirs(out, exist_ok=True)

    wanted = {}
    for f in sorted(glob.glob(os.path.join(a.procdir, "KMOS.*", "science", "SCI_RECONSTRUCTED_*" + FINAL))):
        if band_of(f) == a.band:
            wanted[os.path.basename(f)] = os.path.realpath(f)
    if not wanted:
        sys.exit("no %s *%s found under %s" % (a.band, FINAL, a.procdir))

    removed = 0
    for name in os.listdir(out):
        p = os.path.join(out, name)
        if not os.path.islink(p):
            print("WARNING: not a link, left untouched: %s" % p)
            continue
        if name not in wanted or os.path.realpath(p) != wanted[name] or not os.path.exists(p):
            os.remove(p)                       # removes the link only, never the target
            removed += 1
    added = 0
    for name, target in wanted.items():
        dst = os.path.join(out, name)
        if not os.path.lexists(dst):
            os.symlink(target, dst)
            added += 1

    dates = collections.Counter()
    per_ob = collections.Counter()
    old = []
    for name, target in wanted.items():
        d = date.fromtimestamp(os.path.getmtime(target)).isoformat()
        dates[d] += 1
        per_ob[target.split(os.sep)[-3]] += 1
        if a.min_date and d < a.min_date:
            old.append(target.split(os.sep)[-3])

    print("%s: %d final cubes linked in %s  (%d added, %d stale links removed)"
          % (a.band, len(wanted), out, added, removed))
    print("  OB folders: %d   files per OB: %s" % (len(per_ob), dict(sorted(collections.Counter(per_ob.values()).items()))))
    print("  file dates: %s" % dict(sorted(dates.items())))
    missing = check_expected(a, wanted) if a.expected else 0
    if old:
        print("  OLDER THAN %s (%d files) in: %s" % (a.min_date, len(old), ", ".join(sorted(set(old)))))
    if old or missing:
        sys.exit(1)


def check_expected(a, wanted):
    """Compare with the frames of a previous combination (CSV: band,target_list,frame,
    e.g. scripts/expected_frames_2026-10-02.csv from the PROVn keywords of the
    2026-10-02 combined cubes) and look for missing final cubes anywhere under --search."""
    exp = defaultdict(set)
    with open(a.expected, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["band"] == a.band:
                exp[r["target_list"]].add(r["frame"])
    have = set()
    for name in wanted:
        # time separators may be ':' (server), '-' or U+F03A (local Windows copies)
        m = re.search(r"KMOS\.(\d{4}-\d\d-\d\d)T(\d\d)[-:](\d\d)[-:](\d\d\.\d+)_s_i_bkg2d_rv\.fits$", name)
        if m:
            have.add("KMOS.%sT%s:%s:%s" % m.groups())
    print("\n  vs. %s:" % os.path.basename(a.expected))
    miss_all = []
    for tl in sorted(exp):
        miss = sorted(exp[tl] - have)
        miss_all += miss
        print("    %-30s expected %4d  found %4d  missing %4d" % (tl, len(exp[tl]), len(exp[tl]) - len(miss), len(miss)))
    extra = have - set().union(*exp.values()) if exp else have
    if extra:
        print("    not in the previous combination: %d (e.g. new OBs): %s" % (len(extra), ", ".join(sorted(extra)[:5])))
    if not miss_all:
        return 0
    print("\n  missing frames by night: %s" % dict(sorted(Counter(f[5:15] for f in miss_all).items())))
    print("  searching %s for them (final cubes only) ..." % a.search)
    found = defaultdict(list)
    names = {"SCI_RECONSTRUCTED_%s_%s%s" % (p, f, FINAL): f for f in miss_all for p in ("dc", "dcr")}
    for root, dirs, files in os.walk(a.search):
        dirs[:] = [d for d in dirs if not d.startswith("edps_in_")]
        for fn in files:
            if fn in names:
                found[os.path.dirname(os.path.dirname(os.path.join(root, fn)))].append(names[fn])
    for d, fs in sorted(found.items()):
        print("    found %3d of them in %s" % (len(fs), d))
    n_found = sum(len(v) for v in found.values())
    if n_found < len(miss_all):
        rest = sorted(set(miss_all) - {f for v in found.values() for f in v})
        print("    NOT FOUND anywhere (%d): %s%s" % (len(rest), ", ".join(rest[:8]), " ..." if len(rest) > 8 else ""))
    return len(miss_all)


if __name__ == "__main__":
    main()
