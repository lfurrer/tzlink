#!/usr/bin/env python3
# coding: utf8

# Author: Lenz Furrer, 2018


'''
Entry point function for running the ranking CNN.
'''


import sys

from ..util.util import get_config
from ..util.record import Recorder


def run(config, record=False, **kwargs):
    '''
    Run the CNN (incl. preprocessing).
    '''
    conf = get_config(config)
    recorder = Recorder(conf)

    from . import cnn
    cnn.run(conf, test=[sys.stdout, recorder.results], **kwargs)
    if record:
        recorder.dump()
