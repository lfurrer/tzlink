#!/usr/bin/env python3
# coding: utf8

# Author: Lenz Furrer, 2018


'''
Entry point function for running the ranking CNN.
'''


from ..conf.config import Config


def run(config, **kwargs):
    '''
    Run the CNN (incl. preprocessing).
    '''
    if not isinstance(config, Config):
        if config is None:
            config = []
        elif isinstance(config, str):
            config = [config]
        config = Config(*config)

    from . import cnn
    cnn.run(config, **kwargs)
