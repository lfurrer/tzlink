#!/usr/bin/env python3
# coding: utf8

# Author: Lenz Furrer, 2018


'''
Top-level functions for running the ranking CNN.
'''


# Make it possible to run `python3 -m tzlink.rank`.
# In order for the startup script to work properly,
# tzlink.launch.launch must be called before keras/
# tensorflow are imported (which happens in tzlink.
# rank.cnn).  Therefore, some package imports are
# inside the top-level functions.


from .predictions import handle_predictions


def run_training(*args, **kwargs):
    '''Train a model.'''
    from . import cnn
    cnn.run_training(*args, **kwargs)


def prediction_samples(conf):
    '''Get prediction samples.'''
    from ..preprocessing import samples
    return samples.Sampler(conf).prediction_samples()
