from __future__ import print_function
import numpy as np
import shutil
import astropy.io.fits as fits
import os, sys
import subprocess
import time

__all__ = ["clean_move", "get_obs_extensions", "fetch_resource_path", 
           "get_sof_list", "duplicate", "get_obs_exposures", "patch_header",
           "update_header"]

dateobsfmt = "%Y-%m-%dT%H:%M:%S.%f"

def clean_move(src, dst):
    """Move a file, deleting the target if it already exists. If target is a directory, 
    preserve the original file name, otherwise move change name.

    Parameters
    ----------
    src: string
        Name of the source file.

    dst: string
        Name of the destination file or directory.

    """
    if os.path.isdir(dst): #assume copying same file to different directory
        if os.path.exists(os.path.join(dst,src)):
            os.remove(os.path.join(dst,src))
    else:
        if os.path.exists(dst):
            os.remove(dst)
    shutil.move(src, dst)
    return


def clean_copy(src, dst):
    """copy a file, deleting the target if it already exists. If target is a directory, 
    preserve the original file name, otherwise move change name.

    Parameters
    ----------
    src: string
        Name of the source file.

    dst: string
        Name of the destination file or directory.

    """
    if os.path.isdir(dst): #assume copying same file to different directory
        if os.path.exists(os.path.join(dst,src)):
            os.remove(os.path.join(dst,src))
    else:
        if os.path.exists(dst):
            os.remove(dst)
    shutil.copy(src, dst)
    return


def duplicate(full_path, outputdir=None, prefix=None, suffix=None):
    """ Copy a file from fullpath to outputdir, prepending/appending
    prefix or suffix if supplied.
    """
    splitfile = full_path.split('/')
    fpath, fname = '/'.join(splitfile[:-1]), splitfile[-1]
    ffile, fext = os.path.splitext(fname)

    if fpath == outputdir and prefix is None and suffix is None:
        raise ValueError("Change output directory or add prefix/suffix")
    
    if suffix is not None:
        soutname = '{0}_{1}{2}'.format(ffile, suffix, fext)
    else:
        soutname = fname

    if prefix is not None:
        outname = '{0}_{1}'.format(prefix, soutname)
    else:
        outname = soutname

    outpath = os.path.join(outputdir, outname)
    clean_copy(full_path, outpath)
    
    return outpath


def fetch_resource_path(resource):
    """ Fetch the path for a static resource included with the package.
    takes as input the relative path (usually data/XXX.fits) and returns 
    the full install path as a string.
    """
    d = os.path.dirname(sys.modules['pyspark'].__file__)
    return os.path.join(d, resource)


def get_sof_list(filetype=None, subdir=None, inputdir=None, obsband=None, 
                 darkexp=None, prefix=None, rotangle=None, logger=None, **kwargs):
    """ Return a list of frames matching the requested filetype and band
    in the given directory.

    Parameters
    ----------
    filetype: string
        File type to search for, should be one of arc, flat, dark, 
        standard, or science.

    input_dir: string
        Full path to the directory to search.

    obs_band: string
        Observing band to check for. Should be one of IZ, YJ, H, HK, K.

    exptime: int
        Exposure time constraint (only used for darks).  If no dark is 
        found with the given exposure time, this defaults to a 60s dark.
    """
    if subdir is not None:
        fpath = os.path.join(inputdir, subdir)
    else:
        fpath = inputdir

    if prefix is None:
        filepath = 'KMOS'
    else:
        filepath = '{0}_KMOS'.format(prefix)
   
    
#    dfits_string = 'dfits '+filepath+'*{0}*.fits | fitsort tpl.id dpr.type ins.filt1.id det.seq1.dit ocs.rot.naangle date-obs obs.start obs.targ.name | tail -n +2 '
    dfits_string = 'dfits '+filepath+'*.fits | fitsort tpl.id dpr.type ins.filt1.id det.seq1.dit ocs.rot.naangle date-obs obs.start obs.targ.name | grep {0}  '

    sof_out = []
   
    # --- Flat field frames --- 
    if filetype.lower() == 'flat':
        block_init = []
        flat_exp = []
        with cd(fpath):
            #p = subprocess.Popen(dfits_string.format('CAL'), stdout=subprocess.PIPE, bufsize=1, shell=True)
            p = subprocess.Popen(dfits_string.format('FLAT'), stdout=subprocess.PIPE, bufsize=-1, shell=True)
            for line in iter(p.stdout.readline, b''):  
                sline = line.decode('utf-8').strip().split(None)
                if sline[1] == 'KMOS_spec_cal_calunitflat' and sline[3].upper() in [obsband.upper(), 'BLOCK']:
                    if rotangle is None:
                        if sline[2] == 'FLAT,OFF':
                            block_init.append((os.path.join(fpath, sline[0]), sline[2], float(sline[4])))
                        else:
                            sof_out.append((os.path.join(fpath, sline[0]), sline[2]))
                            flat_exp.append(float(sline[4]))
                    else:
                        if sline[2] == 'FLAT,OFF':
                            block_init.append((os.path.join(fpath, sline[0]), sline[2], float(sline[4])))
                        else:
                            rangle = float(sline[5])
                            if rangle < 0: rangle += 360.
                            if float(rotangle) == rangle:
                                sof_out.append((os.path.join(fpath, sline[0]), sline[2]))
                                flat_exp.append(float(sline[4]))
            p.wait()
        
        # pass back over the flats to match appropriate block frames
        flat_exp = np.unique(flat_exp)
        if len(flat_exp) > 1:
            return 1
        for dframe, dtype, dexp in block_init:
            if dexp == flat_exp:
                sof_out.append((dframe, dtype,))

    # --- Arc line frames --- 
    elif filetype.lower() == 'arc':
        block_init = []
        arc_exp = []
        with cd(fpath):
#            p = subprocess.Popen(dfits_string.format('CAL'), stdout=subprocess.PIPE, bufsize=1, shell=True)
            p = subprocess.Popen(dfits_string.format('WAVE'), stdout=subprocess.PIPE, bufsize=-1, shell=True)
            for line in iter(p.stdout.readline, b''):
               sline = line.decode('utf-8').strip().split(None)
               if sline[1] == 'KMOS_spec_cal_wave' and sline[3].upper() in [obsband.upper(), 'BLOCK']:
                    if rotangle is None:
                        if sline[2] == 'WAVE,OFF':
                            block_init.append((os.path.join(fpath, sline[0]), sline[2], float(sline[4])))
                        else:
                            sof_out.append((os.path.join(fpath, sline[0]), sline[2]))
                            arc_exp.append(float(sline[4]))
                    else:
                        if sline[2] == 'WAVE,OFF':
                            block_init.append((os.path.join(fpath, sline[0]), sline[2], float(sline[4])))
                        else:
                            rangle = float(sline[5])
                            if rangle < 0: rangle += 360.
                            if float(rotangle) == rangle:
                                sof_out.append((os.path.join(fpath, sline[0]), sline[2]))
                                arc_exp.append(float(sline[4]))
            p.wait()

        # pass back over the arcs to match appropriate block frames
        arc_exp = np.unique(arc_exp)
        if len(arc_exp) > 1:
            return 1

        for dframe, dtype, dexp in block_init:
            if dexp == arc_exp:
                sof_out.append((dframe, dtype,))

    # --- Darks frames --- 
    elif filetype.lower() == 'dark':
        # track darks with two different exposure times
        exp_dark = []
        standard_dark = []

        # first pass: identify darks and group by exposure time
        store_darks = {}
        with cd(fpath):
#            p = subprocess.Popen(dfits_string.format('DARK'), stdout=subprocess.PIPE, bufsize=1, shell=True)
            p = subprocess.Popen(dfits_string.format('DARK'), stdout=subprocess.PIPE, bufsize=-1, shell=True)
            for line in iter(p.stdout.readline, b''):
                sline = line.decode('utf-8').strip().split(None)
                store_darks.setdefault(int(float(sline[4])), []).append((os.path.join(fpath, sline[0]),None))
            p.wait()

        dark_times = list(store_darks.keys())
        if darkexp is not None and int(darkexp) in dark_times:
            if logger is not None:
                logger.debug("- Found darks with {0}s".format(darkexp))
            sof_out = store_darks[int(darkexp)]
        elif 60 in dark_times:
            if logger is not None:
                logger.debug("- {0}s darks not found, falling back to 60s".format(darkexp))
            sof_out = store_darks[60]
        else:
            # neither the provided time or 60s is in the dark folder
            if len(dark_times) == 1: #use the only time that's there
                if logger is not None:
                    logger.debug("- Only one set of darks found, using {0}s".format(dark_times[0]))
                sof_out = store_darks[dark_times[0]]
            else: # otherwise use the shortest time there
                if logger is not None:
                    logger.debug("- Multiple (non-specified) darks found, using {0}s".format(min(dark_times)))
                sof_out = store_darks[min(dark_times)]

    # --- Standard star exposures --- 
    elif filetype.lower() == 'standard':
        with cd(fpath):
#            p = subprocess.Popen(dfits_string.format('CAL'), stdout=subprocess.PIPE, bufsize=1, shell=True)
            p = subprocess.Popen(dfits_string.format('CAL'), stdout=subprocess.PIPE, bufsize=-1, shell=True)
            for line in iter(p.stdout.readline, b''):
                sline = line.decode('utf-8').strip().split(None)
                if obsband is not None:
                    if sline[1] in ['KMOS_spec_cal_stdstar','KMOS_spec_cal_stdstarscipatt'] and sline[3] == obsband.upper():
                        sof_out.append((os.path.join(fpath, sline[0]), sline[4], sline[7], sline[8], float(sline[5])))
                else:
                    if sline[1] in ['KMOS_spec_cal_stdstar','KMOS_spec_cal_stdstarscipatt']:
                        sof_out.append((os.path.join(fpath, sline[0]), sline[4], sline[7], sline[8], float(sline[5])))

            p.wait()

    # --- Science exposures --- 
    elif filetype.lower() == 'science':
        #print('OK') 
        #with open("/home/lcoccato/pyspark/pyspark/scripts/tmp.txt", "a+") as f:
        #f.write('speriam bene')
            
         with cd(fpath):
#            p = subprocess.Popen(dfits_string.format('OBJECT,SKY'), stdout=subprocess.PIPE, bufsize=1, shell=True)
            p = subprocess.Popen(dfits_string.format('OBJECT,SKY'), stdout=subprocess.PIPE, bufsize=-1, shell=True)
            for line in iter(p.stdout.readline, b''):
                sline = line.decode('utf-8').strip().split(None)
                if sline[2] == 'OBJECT,SKY' and sline[3] == obsband.upper():
                    sof_out.append((os.path.join(fpath, sline[0]), sline[6]))
            p.wait()

    return sof_out
 

def get_obs_exposures(inputdir=None, subdir=None, obsband=None, prefix=None,
		      suffix=None, **kwargs):
    """ Get list of reconstructed frames (assumes file naming convention from
    kmo_reconstruct, i.e. sci_reconstructed_***.fits.
    """
    if subdir is not None:
        fpath = os.path.join(inputdir, subdir)
    else:
        fpath = inputdir

    #keep track of the expected number of name elements to identify correct frames
    #print(prefix, suffix)
    if prefix is None:
        expected_split = 2
        filepath = 'SCI_RECONSTRUCTED_KMOS'
        #print('1 expected: ',str(expected_split))
    else:
        expected_split = 3
        filepath = 'SCI_RECONSTRUCTED{0}_KMOS'.format('_{0}'.format(prefix))
        expected_split += len(prefix.split('_'))
        #print('2 expected: ',str(expected_split))

    if suffix is None:
        dfits_string = 'dfits '+filepath+'*.fits | fitsort date-obs ins.grat1.id det.seq1.dit ocs.arm1.name ocs.arm2.name ocs.arm3.name | tail -n +2'
        #print('3 expected: ',str(expected_split))
    else:
        dfits_string = 'dfits '+filepath+'*{0}.fits | fitsort date-obs ins.grat1.id det.seq1.dit ocs.arm1.name ocs.arm2.name ocs.arm3.name | tail -n +2'.format('_{0}'.format(suffix))
        #dfits_string = 'dfits '+filepath+'*.fits | fitsort date-obs ins.grat1.id det.seq1.dit ocs.arm1.name ocs.arm2.name ocs.arm3.name | tail -n +2'#.format('_{0}'.format(suffix))
        expected_split += len(suffix.split('_'))
        #print('4 expected: ',str(expected_split))

    # run the query, sort things into object and sky
    # we actually temporarily change directories here
    # to deal with the length limit imposed by dfits
    exposures = []
    with cd(fpath):
        #p = subprocess.Popen(dfits_string, stdout=subprocess.PIPE, bufsize=1, shell=True)
        p = subprocess.Popen(dfits_string, stdout=subprocess.PIPE, bufsize=-1, shell=True)
        for line in iter(p.stdout.readline, b''):
            sline = line.decode('utf-8').strip().split(None)
            #print(os.path.split(sline[0])[1].split('_'), len(os.path.split(sline[0])[1].split('_')), expected_split)
            if len(os.path.split(sline[0])[1].split('_')) == expected_split:
                if all([True if not ent.endswith('_SKY') else False
                        for ent in sline[-3:]]) and sline[2] == obsband:
                    exposures.append((os.path.join(fpath, sline[0]), 'OBJ', time.mktime(time.strptime(sline[1], dateobsfmt)), float(sline[3])))
                elif sline[2] == obsband:
                    exposures.append((os.path.join(fpath, sline[0]), 'SKY', time.mktime(time.strptime(sline[1], dateobsfmt)), float(sline[3])))
        p.wait()

    return exposures


def patch_header(frame, header_keys, reference_frame, ext=0):
    """ Given a reference frame and list of header keywords, copy header 
    values into frame. 
    """
    tfile = fits.open(frame, mode='update')
    ref_header = fits.getheader(reference_frame, ext=ext)
    for hkey in header_keys:
        hval = ref_header[hkey]
        hcom = ref_header.comments[hkey]

        tfile[ext].header[hkey] = (hval, hcom)
    tfile.flush()
    return 0


def update_header(frame, header_dict, ext=0):
    """ Given a reference frame and list of header keywords, copy header 
    values into frame. 
    """
    tfile = fits.open(frame, mode='update')
    for hkey in header_dict.keys():
        tfile[ext].header[hkey] = header_dict[hkey]
    tfile.flush()
    return 0



def verify_structure(outputdir=None, subdir=None):
    """ Routine which checks if the necessary folder structure exists
    if not, put it in place.
    """
    checkdirs = ['sky', 'science', 'calib', 'psf', 'sof', 'std']
    basedir = os.path.join(outputdir, subdir)

    if not os.path.exists(basedir):
        os.makedirs(basedir)
    for subdir in checkdirs:
        if not os.path.exists(os.path.join(basedir, subdir)):
            os.makedirs(os.path.join(basedir, subdir))

    return


class cd:
    """ A context manager class that can be used to temporarily change
    directories.
    """
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)
    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)
    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)


