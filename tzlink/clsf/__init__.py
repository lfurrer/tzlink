#!/usr/bin/env python3
# coding: utf8

# Author: Lenz Furrer, 2018


'''
Top-level functions for running the classification CNN.
'''


from .predictions import Evaluator


def run_training(*args, **kwargs):
    '''Train a model.'''
    from . import classify
    classify.run_training(*args, **kwargs)


def prediction_samples(conf):
    '''Get prediction samples.'''
    from ..preprocessing import samples
    from .classify import getlabels, val_samples
    resources = samples.Sampler(conf)
    labelset = getlabels(conf, resources)
    return val_samples(conf, resources, labelset)
