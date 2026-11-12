#!/usr/bin/env python
"""Batch reduction of archival KMOS OBs (one folder per OB, e.g. KMOS.2014-04-07T00:19:19.885/) with reduce_kmos.py.

Three steps, all resumable:

  prepare  Stage every OB folder of SRC into STAGE/<OB>/ as plain KMOS*.fits (symlinks for .fits, decompressed
           copies for .fits.Z/.gz/.bz2/.fz), read the headers and write STAGE/inventory.json: per OB and band the
           number of science frames and of flats/arcs/darks/standards, plus the problems that stop a band from being
           reduced (no flat, no arc, no dark). Prints a time estimate. Nothing in SRC is changed.

      python scripts/archive_batch.py prepare /path/ARCHIVE/COSMOS_FIELD_FULL_RESULTS_old /path/cosmos_stage -j 8

  run      Reduce every OB/band of the inventory: one reduce_kmos.py call per OB and band, from its own working
           directory WORK/<OB>/<band>/ (esorex writes .sof files there), with a config built from TEMPLATE
           (the EMPOWER settings, examples/empower.cfg with real paths). -j OBs run in parallel; the bands of one OB
           run one after the other. Finished OB/bands (status.json says ok) are skipped on a rerun.
           --clean removes the large intermediate files of an OB once all its bands are ok.

      python scripts/archive_batch.py run /path/cosmos_stage /path/cosmos_proc /path/cosmos_work \
             --template my_empower.cfg -j 6 --clean

  status   Progress, measured minutes per OB/band, a fit  t = a + b * n_science_frames  and the ETA of the rest.

      python scripts/archive_batch.py status /path/cosmos_stage /path/cosmos_work -j 6

Start `run` inside tmux/screen (or with nohup): it takes days.
"""
import argparse
import configparser
import glob
import json
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from astropy.io import fits

HERE = os.path.dirname(os.path.abspath(__file__))
REDUCE = os.path.join(HERE, 'reduce_kmos.py')
BANDS = ('IZ', 'YJ', 'H', 'K', 'HK')
FINAL = '_s_i_bkg2d_rv.fits'
STD_TPL = ('KMOS_spec_cal_stdstar', 'KMOS_spec_cal_stdstarscipatt')
DECOMP = {'.Z': ['gzip', '-dc'], '.gz': ['gzip', '-dc'], '.bz2': ['bzip2', '-dc']}

# default guesses for the estimate before anything was measured (minutes); `status` replaces them by measured values
T_CAL = 20.0      # calibrations of one OB and band: dark, flat + illum at 6 rotator angles, arc, standard
T_FRAME = 4.0     # one science frame: detector + row common-mode, kmos_sci_red (twice if hybrid), ZAP, illum/bkg/rv


# --------------------------------------------------------------------------------------------- prepare
def stage_file(src, dst):
    """Symlink a .fits, decompress anything else to dst (atomic via .part)."""
    if os.path.exists(dst):
        return
    if src.endswith('.fits'):
        os.symlink(os.path.realpath(src), dst)
        return
    tmp = dst + '.part'
    if src.endswith('.fz'):
        if os.path.exists(tmp):
            os.remove(tmp)
        subprocess.run(['funpack', '-O', tmp, src], check=True)
    else:
        cmd = DECOMP[os.path.splitext(src)[1]]
        with open(tmp, 'wb') as fo:
            subprocess.run(cmd + [src], stdout=fo, check=True)
    os.replace(tmp, dst)


def raw_files(obdir):
    """All raw KMOS frames below an OB folder (any depth), as {staged name: source path}."""
    out = {}
    for root, _, files in os.walk(obdir):
        for f in files:
            if not f.startswith('KMOS'):          # M.KMOS... (archive products), README, xml, ... are ignored
                continue
            for ext in ('.fits', '.fits.Z', '.fits.gz', '.fits.bz2', '.fits.fz'):
                if f.endswith(ext):
                    out.setdefault(f[:-len(ext)] + '.fits', os.path.join(root, f))
    return out


def classify(h):
    tpl = str(h.get('ESO TPL ID', '')).strip()
    typ = str(h.get('ESO DPR TYPE', '')).strip()
    band = str(h.get('ESO INS FILT1 ID', '')).strip().upper()
    if tpl in STD_TPL:
        kind = 'std'
    elif typ == 'OBJECT,SKY':                     # the selection of fileutils.get_sof_list(filetype='science')
        kind = 'science'
    elif typ.startswith('DARK'):
        kind = 'dark'
    elif tpl == 'KMOS_spec_cal_calunitflat' and typ != 'FLAT,OFF':
        kind = 'flat'
    elif tpl == 'KMOS_spec_cal_wave' and typ != 'WAVE,OFF':
        kind = 'arc'
    else:
        kind = 'other'
    return kind, band


def inventory_ob(src, stage, ob):
    files = raw_files(os.path.join(src, ob))
    dest = os.path.join(stage, ob)
    os.makedirs(dest, exist_ok=True)
    rec = {'ob': ob, 'nraw': len(files), 'bands': {}, 'dark_dits': [], 'targets': [], 'progs': [], 'problems': []}
    count = {b: Counter() for b in BANDS}
    targets, progs = Counter(), Counter()
    for name, path in sorted(files.items()):
        out = os.path.join(dest, name)
        try:
            stage_file(path, out)
            h = fits.getheader(out, 0)
        except Exception as e:                    # corrupt download, missing funpack, ...
            rec['problems'].append('%s: %s' % (name, e))
            continue
        kind, band = classify(h)
        if kind == 'dark':
            rec['dark_dits'].append(round(float(h.get('ESO DET SEQ1 DIT', 0))))
        elif band in count and kind != 'other':
            count[band][kind] += 1
            if kind == 'science':
                targets[str(h.get('ESO OBS TARG NAME', ''))] += 1
                progs[str(h.get('ESO OBS PROG ID', ''))] += 1
    rec['dark_dits'] = sorted(set(rec['dark_dits']))
    rec['targets'], rec['progs'] = sorted(targets), sorted(progs)
    for b in BANDS:
        c = count[b]
        if not c['science']:
            continue
        miss = [k for k in ('flat', 'arc') if not c[k]] + ([] if rec['dark_dits'] else ['dark'])
        rec['bands'][b] = {'science': c['science'], 'flat': c['flat'], 'arc': c['arc'], 'std': c['std'],
                           'missing': miss, 'reduce': not miss}
        if miss:
            rec['problems'].append('%s: no %s, band skipped' % (b, '/'.join(miss)))
        elif not c['std']:
            rec['problems'].append('%s: no standard star, run with make_standard = False' % b)
    if not files:
        rec['problems'].append('no raw KMOS*.fits frames in this folder')
    elif not rec['bands']:
        rec['problems'].append('no OBJECT,SKY science frames')
    return rec


def estimate(inv, jobs, t_cal=T_CAL, t_frame=T_FRAME, skip=()):
    serial = 0.0
    for r in inv:
        for b, c in r['bands'].items():
            if c['reduce'] and (r['ob'], b) not in skip:
                serial += t_cal + t_frame * c['science']
    return serial / 60.0, serial / 60.0 / max(jobs, 1)


def cmd_prepare(a):
    obs = sorted(d for d in os.listdir(a.src) if os.path.isdir(os.path.join(a.src, d)) and d.startswith('KMOS'))
    a.stage = os.path.abspath(a.stage)
    os.makedirs(a.stage, exist_ok=True)
    print('%d OB folders in %s' % (len(obs), a.src))
    inv = []
    with ThreadPoolExecutor(a.jobs) as ex:
        futs = {ex.submit(inventory_ob, a.src, a.stage, ob): ob for ob in obs}
        for i, f in enumerate(as_completed(futs), 1):
            inv.append(f.result())
            if i % 50 == 0 or i == len(obs):
                print('  %d/%d inventoried' % (i, len(obs)), flush=True)
    inv.sort(key=lambda r: r['ob'])
    with open(os.path.join(a.stage, 'inventory.json'), 'w') as fo:
        json.dump(inv, fo, indent=1)

    nb, nf, nskip = Counter(), Counter(), Counter()
    for r in inv:
        for b, c in r['bands'].items():
            (nb if c['reduce'] else nskip)[b] += 1
            if c['reduce']:
                nf[b] += c['science']
    print('\nband   OBs to reduce   science frames   OBs skipped (missing calibs)')
    for b in BANDS:
        if nb[b] or nskip[b]:
            print('%-5s  %13d   %14d   %d' % (b, nb[b], nf[b], nskip[b]))
    progs = Counter(p for r in inv for p in r['progs'])
    print('\nprogrammes (OBs):', ', '.join('%s (%d)' % kv for kv in progs.most_common()))
    bad = [r for r in inv if r['problems']]
    print('%d OBs with problems, see "problems" in %s' % (len(bad), os.path.join(a.stage, 'inventory.json')))
    for r in bad[:15]:
        print('  %s: %s' % (r['ob'], '; '.join(r['problems'][:3])))
    serial, wall = estimate(inv, a.run_jobs)
    print('\nEstimate (guess: %.0f min calibrations per OB/band + %.0f min per science frame):' % (T_CAL, T_FRAME))
    print('  %.0f h serial = %.1f days;  with %d parallel OBs ~%.0f h = %.1f days' % (
        serial, serial / 24, a.run_jobs, wall, wall / 24))
    print('  Run a few OBs first, then `status` gives the estimate from the measured times.')
    if nb['K'] or nb['HK']:
        print('\nNote: K/HK use the bundled ZAPv2 basis and classic O-S (hybrid only for IZ,YJ,H), and '
              'telluric_static/ has no K files for telluric_combined.py.')


# --------------------------------------------------------------------------------------------- run
def make_cfg(template, stage, procdir, ob, band, has_std, path):
    cp = configparser.ConfigParser()
    cp.optionxform = str
    if not cp.read(template):
        raise SystemExit('cannot read template %s' % template)
    cp['instrument']['obsband'] = band
    cp['pathing']['rawdir'] = stage
    cp['pathing']['procdir'] = procdir
    cp['pathing']['subdirs'] = ob
    if not has_std:
        cp['calibration']['make_standard'] = 'False'
    with open(path, 'w') as fo:
        cp.write(fo)


def finals(procdir, ob, band):
    n = 0
    for f in glob.glob(os.path.join(procdir, ob, 'science', 'SCI_RECONSTRUCTED_*' + FINAL)):
        try:
            if str(fits.getheader(f, 0).get('ESO INS FILT1 ID', '')).strip().upper() == band:
                n += 1
        except Exception:
            pass
    return n


def clean_ob(procdir, ob):
    """Remove the large intermediates; keeps calib/, sky/ (to rerun the sky subtraction) and the final cubes."""
    base = os.path.join(procdir, ob)
    pats = ['dc_KMOS*.fits', 'dcr_KMOS*.fits', 'science/*_s.fits', 'science/*_s_i.fits', 'science/*_s_i_bkg2d.fits']
    freed = 0
    for p in pats:
        for f in glob.glob(os.path.join(base, p)):
            freed += os.path.getsize(f)
            os.remove(f)
    return freed


def read_status(work, ob, band):
    try:
        with open(os.path.join(work, ob, band, 'status.json')) as fi:
            return json.load(fi)
    except (OSError, ValueError):
        return None


LOCK = threading.Lock()


def run_ob(r, a, cfg_threads):
    out = []
    for band, c in r['bands'].items():
        if not c['reduce']:
            continue
        st = read_status(a.work, r['ob'], band)
        if st and st['status'] == 'ok' and not a.redo:
            out.append(st)
            continue
        wdir = os.path.join(a.work, r['ob'], band)
        os.makedirs(wdir, exist_ok=True)
        for f in glob.glob(os.path.join(wdir, '*.sof')):
            os.remove(f)
        cfg = os.path.join(wdir, 'reduce.cfg')
        make_cfg(a.template, a.stage, a.procdir, r['ob'], band, c['std'] > 0, cfg)
        env = dict(os.environ, OMP_NUM_THREADS=str(cfg_threads))
        t0 = time.time()
        with open(os.path.join(wdir, 'reduce.out'), 'w') as fo:
            rc = subprocess.run([sys.executable, REDUCE, '-p', cfg], cwd=wdir, stdout=fo,
                                stderr=subprocess.STDOUT, env=env).returncode
        dt = (time.time() - t0) / 60.0
        nfin = finals(a.procdir, r['ob'], band)
        # reduce_kmos.py logs a failed step and returns 0, so success is judged from the final cubes
        st = {'ob': r['ob'], 'band': band, 'minutes': round(dt, 2), 'nscience': c['science'], 'nfinal': nfin,
              'returncode': rc, 'status': 'ok' if (rc == 0 and nfin > 0) else 'failed',
              'finished': time.strftime('%Y-%m-%d %H:%M:%S')}
        with open(os.path.join(wdir, 'status.json'), 'w') as fo:
            json.dump(st, fo)
        with LOCK, open(os.path.join(a.work, 'runlog.jsonl'), 'a') as fo:
            fo.write(json.dumps(st) + '\n')
        out.append(st)
    if a.clean and out and all(s['status'] == 'ok' for s in out):
        freed = clean_ob(a.procdir, r['ob'])
        if freed:
            print('    %s: cleaned %.1f GB' % (r['ob'], freed / 1e9), flush=True)
    return out


def load_inventory(stage):
    with open(os.path.join(stage, 'inventory.json')) as fi:
        return json.load(fi)


def cmd_run(a):
    with open(a.template) as fi:
        if '/path/to' in fi.read():
            raise SystemExit('%s still contains /path/to placeholders (svd_path_*, oh_spec_iz, recipe_dir)' % a.template)
    a.stage, a.procdir, a.work, a.template = map(os.path.abspath, (a.stage, a.procdir, a.work, a.template))
    inv = load_inventory(a.stage)
    if a.only:
        want = set(a.only)
        inv = [r for r in inv if r['ob'] in want]
    if a.bands:
        bs = set(a.bands.upper().split(','))
        inv = [dict(r, bands={b: c for b, c in r['bands'].items() if b in bs}) for r in inv]
    inv = [r for r in inv if any(c['reduce'] for c in r['bands'].values())]
    if a.limit:
        inv = inv[:a.limit]
    os.makedirs(a.work, exist_ok=True)
    os.makedirs(a.procdir, exist_ok=True)
    threads = a.threads or max(1, (os.cpu_count() or 1) // a.jobs)
    print('%d OBs, %d in parallel, OMP_NUM_THREADS=%d' % (len(inv), a.jobs, threads), flush=True)
    t0, done, durations, nfail = time.time(), 0, [], 0
    with ThreadPoolExecutor(a.jobs) as ex:
        futs = {ex.submit(run_ob, r, a, threads): r['ob'] for r in inv}
        for f in as_completed(futs):
            done += 1
            try:
                res = f.result()
            except Exception as e:
                print('%s: exception %s' % (futs[f], e), flush=True)
                nfail += 1
                continue
            for s in res:
                nfail += s['status'] != 'ok'
            mins = sum(s['minutes'] for s in res)
            durations.append(mins)
            el = (time.time() - t0) / 3600.0
            eta = el / done * (len(inv) - done)
            print('[%d/%d] %s %s  %.0f min  | elapsed %.1f h, ETA %.1f h, failed OB/bands %d' % (
                done, len(inv), futs[f], ' '.join('%s:%s(%d/%d)' % (s['band'], s['status'], s['nfinal'],
                                                                     s['nscience']) for s in res),
                mins, el, eta, nfail), flush=True)
    print('finished; failed OB/bands: %d (see %s)' % (nfail, os.path.join(a.work, 'runlog.jsonl')))


# --------------------------------------------------------------------------------------------- status
def cmd_status(a):
    inv = load_inventory(a.stage)
    ok, failed, todo = [], [], set()
    for r in inv:
        for b, c in r['bands'].items():
            if not c['reduce']:
                continue
            st = read_status(a.work, r['ob'], b)
            if st is None:
                todo.add((r['ob'], b))
            elif st['status'] == 'ok':
                ok.append(st)
            else:
                failed.append(st)
                todo.add((r['ob'], b))
    print('OB/bands: %d ok, %d failed, %d not done yet' % (len(ok), len(failed), len(todo) - len(failed)))
    t_cal, t_frame = T_CAL, T_FRAME
    if ok:
        m = [s['minutes'] for s in ok]
        print('measured minutes per OB/band: median %.0f, min %.0f, max %.0f' % (
            statistics.median(m), min(m), max(m)))
        n = [s['nscience'] for s in ok]
        if len(ok) >= 5 and len(set(n)) > 1:
            mn, mm = statistics.mean(n), statistics.mean(m)
            b = sum((x - mn) * (y - mm) for x, y in zip(n, m)) / sum((x - mn) ** 2 for x in n)
            t_frame, t_cal = max(b, 0.1), max(mm - b * mn, 0.0)
            print('fit: %.1f min per OB/band + %.1f min per science frame' % (t_cal, t_frame))
        else:
            t_cal, t_frame = 0.0, statistics.median(s['minutes'] / max(s['nscience'], 1) for s in ok)
            print('%.1f min per science frame (median; fit needs >= 5 OB/bands of different size)' % t_frame)
    done = {(s['ob'], s['band']) for s in ok}
    serial, wall = estimate(inv, a.jobs, t_cal, t_frame, skip=done)
    print('remaining: %.0f h serial; with %d parallel OBs ~%.0f h = %.1f days' % (serial, a.jobs, wall, wall / 24))
    if failed:
        print('failed (see WORK/<OB>/<band>/reduce.out and the pyspark log in PROCDIR/<OB>/):')
        for s in failed[:30]:
            print('  %s %s rc=%s finals=%d/%d' % (s['ob'], s['band'], s['returncode'], s['nfinal'], s['nscience']))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('prepare', help='stage + decompress + inventory + estimate')
    p.add_argument('src', help='archive folder with one KMOS.<date> folder per OB')
    p.add_argument('stage', help='new folder for the staged raw frames (becomes rawdir)')
    p.add_argument('-j', '--jobs', type=int, default=8, help='parallel staging threads')
    p.add_argument('--run-jobs', type=int, default=6, help='parallel OBs assumed for the estimate')
    p = sub.add_parser('run', help='reduce all OB/bands of the inventory')
    p.add_argument('stage'); p.add_argument('procdir'); p.add_argument('work')
    p.add_argument('--template', required=True, help='config with the EMPOWER settings and real paths')
    p.add_argument('-j', '--jobs', type=int, default=6, help='OBs reduced in parallel')
    p.add_argument('--threads', type=int, help='OMP_NUM_THREADS per job (default: cpus / jobs)')
    p.add_argument('--bands', help='only these bands, e.g. YJ,H')
    p.add_argument('--only', nargs='+', help='only these OB folders')
    p.add_argument('--limit', type=int, help='only the first N OBs (timing test)')
    p.add_argument('--redo', action='store_true', help='rerun OB/bands already ok')
    p.add_argument('--clean', action='store_true', help='delete dc_/dcr_ frames and intermediate stages of ok OBs')
    p = sub.add_parser('status', help='progress and ETA from measured times')
    p.add_argument('stage'); p.add_argument('work')
    p.add_argument('-j', '--jobs', type=int, default=6)
    a = ap.parse_args()
    {'prepare': cmd_prepare, 'run': cmd_run, 'status': cmd_status}[a.cmd](a)


if __name__ == '__main__':
    main()
