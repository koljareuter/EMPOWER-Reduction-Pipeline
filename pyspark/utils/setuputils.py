from __future__ import print_function
import logging
import datetime
import sys 
import os
import numpy as np
import getpass
import re
import random
import string
#import json

try:
    import configparser
except(ImportError):
    import ConfigParser as configparser

__all__ = ["set_logging", "set_config", "get_date"]


def get_date():
    time = datetime.datetime.now().strftime('%Y-%d-%m_%Hh%Mm%S')
    return time


def id_gen(size=10, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


def set_config(configfile=None, logger=None):
    
    """Build parameter dictionaries from input configfile.
    """

    booldir = {'True':True, 'False':False}

    input_params = configparser.ConfigParser()
    input_params.read(configfile)

    #write details of the configuration file to log file, if given
    if logger is not None:
        logger.debug('# --- configuration settings --- #'.format(configfile))
        for sect in input_params.sections():
            logger.debug('[{0}]'.format(sect))
            for name, val in input_params.items(sect):
                logger.debug(' {0} = {1}'.format(name, val))
        logger.debug('# --- configuration settings --- #'.format(configfile))
        logger.debug(' '.format(configfile))

    #Do some additional checking of input bands and directories
    if not input_params.has_option('instrument', 'obsband'):
        raise configparser.NoOptionError("obsband", "instrument")
    else:
        bands = input_params.get('instrument','obsband').strip().split(',')
        if not all([band in ['IZ', 'YJ', 'H', 'K', 'HK'] for band in bands]):
            raise ValueError("Please check that input bands are one of: IZ, YJ, H, K, HK")

    if not input_params.has_option('pathing', 'rawdir') or \
            not input_params.has_option('pathing', 'procdir'):
                raise ValueError("Please make sure that both rawdir and procdir are specified in the configuration file.")

    if not input_params.has_option('pathing', 'subdirs'):
        raise ValueError("Please specify subdirectories for processing")
    else:
        folders = re.split(',|\n', input_params.get('pathing', 'subdirs').strip())

    #read in star prefixes?
    if input_params.has_option('cubes', 'star_prefix'):
        star_names = re.split(',|\n', input_params.get('cubes', 'star_prefix').strip())
    else:
        star_names = None
    
    #A shitty use of configparser: we're building our own dictionary
    #for driving most of the reduction procedures.
    run_params = {'obsband': bands,
                  'rawdir': input_params.get('pathing', 'rawdir'),
                  'procdir': input_params.get('pathing', 'procdir'),
                  'subdirs': folders,
                  }
    
    if input_params.has_section('detector'):
        ddict = {'inputdir': input_params.get('pathing','rawdir'),
                'outputdir': input_params.get('pathing', 'procdir')}
        
        #ddict.update(dict((key, booldir[val]) if val in ['True','False'] 
        #    else (key, val) for (key, val) in input_params['detector'].items()))
        ddict.update(dict((opt, booldir[input_params.get('detector',opt)]) 
                     if input_params.get('detector',opt) in ['True','False'] 
                     else (opt, input_params.get('detector', opt)) 
                     for opt in input_params.options('detector')))
        run_params['detector'] = ddict
    
    if input_params.has_section('calibration'):
        cdict = {'inputdir': input_params.get('pathing','rawdir'),
                'outputdir': input_params.get('pathing','procdir')}
        
        cdict.update(dict((opt, booldir[input_params.get('calibration',opt)]) 
                     if input_params.get('calibration',opt) in ['True','False'] 
                     else (opt, input_params.get('calibration', opt)) 
                     for opt in input_params.options('calibration')))
        run_params['calibration'] = cdict

    if input_params.has_section('cubes'):
        cdict = {'inputdir': input_params.get('pathing','procdir'),
                'outputdir': input_params.get('pathing','procdir'),
                'rawdir': input_params.get('pathing','rawdir'),
                'star_prefix': star_names,
                }
        
        #include prefix value if provided
        if input_params.has_option('detector', 'prefix'):
            cdict['prefix'] = input_params.get('detector', 'prefix')

        copts = input_params.options('cubes')
        for opt in copts:
            if opt != 'star_prefix':
                if input_params.get('cubes', opt) in ['True', 'False']:
                    cdict[opt] = booldir[input_params.get('cubes',opt)]
                else:
                    cdict[opt] = input_params.get('cubes',opt)
        run_params['cubes'] = cdict

    
    if input_params.has_section('combine'):
        combdict = {'inputdir': input_params.get('pathing','procdir'),
                'outputdir': input_params.get('pathing','procdir'),
                }

        #include prefix value if provided
        if input_params.has_option('detector', 'prefix'):
            combdict['prefix'] = input_params.get('detector', 'prefix')


        combdict.update(dict((opt, booldir[input_params.get('combine',opt)]) 
                     if input_params.get('combine',opt) in ['True','False'] 
                     else (opt, input_params.get('combine', opt)) 
                     for opt in input_params.options('combine')))
        run_params['combine'] = combdict

   

    return run_params


def set_logging(logbase='pyspark', outputdir=os.getcwd(), log_level=3, master_log=None, **kwargs):
    """Configure logging,
    """
    logLevels = {0:logging.CRITICAL, 1:logging.ERROR, 2:logging.WARNING, 3:logging.INFO, 4:logging.DEBUG}

    #use random ID generation to split multiple logs 
    logger = logging.getLogger(id_gen())
    logger.setLevel(logging.DEBUG)

    #concatenate time stamp onto output_file_base
    time = get_date()
    output_file = os.path.join(outputdir, '{0}_{1}_{2}.log'.format(getpass.getuser(), logbase, time))

    if master_log is not None:
        master_log.info('- Creating new log file {0}'.format('/'.join(output_file.split('/')[-2:])))
    
    #check if log file already exists (probably not!)
    if os.path.exists(output_file):
        os.remove(output_file)
        if master_log is not None:
            master_log.info('-- Removing existing file {0}'.format(output_file))

    fh = logging.FileHandler(output_file)
    fh.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logLevels[log_level])

    formatter = logging.Formatter('%(asctime)s [%(levelname)-8s]: %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger



