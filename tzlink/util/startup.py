#!/usr/bin/env python3
# coding: utf8

# Author: Lenz Furrer, 2018


'''
Run any startup script.
'''


import os
import logging
from importlib.machinery import SourceFileLoader


def run_scripts(conf):
    '''
    Find startup script paths and execute all.
    '''
    paths = conf.general.startup_scripts
    if not paths:
        return
    if isinstance(paths, str):
        paths = [paths]
    for i, path in enumerate(paths):
        if not os.path.exists(path):
            logging.warning('ignoring unfound startup script: %s', path)
            continue
        run(path, 'startup{}'.format(i))


def run(path, name='startup_script'):
    '''
    Execute a Python script by importing it.
    '''
    SourceFileLoader(name, path).load_module()
