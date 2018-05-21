#!/usr/bin/env python3
# coding: utf8

# Author: Lenz Furrer, 2018


'''
Create training samples from input documents.
'''


import logging
import multiprocessing as mp

import numpy as np

from .load import load_data, itermentions
from .vectorize import Vectorizer
from ..candidates.generate_candidates import candidate_generator


def samples(conf, voc_index, dataset, subset):
    '''
    Create vectorized samples with labels.

    Args:
        conf: a config.Config instance
        voc_index (dict): mapping of words to integers
        dataset (str): dataset identifier
        subset (str): train, dev, or test

    Returns:
        triple of np.array (2D): vocabulary vectors of
            question and answer side, and the labels
    '''
    corpus = load_data(conf, dataset, subset)
    dict_entries = load_data(conf, dataset, 'dict')
    logging.info('loading candidate generator...')
    cand_gen = candidate_generator(conf, dict_entries)
    logging.info('loading vectorizer...')
    vec = Vectorizer(conf, voc_index)
    q_vecs, a_vecs, labels = [], [], []
    logging.info('distributing load to %d workers...', conf.candidates.workers)
    with mp.Pool(conf.candidates.workers,
                 initializer=_set_global_instances,
                 initargs=[cand_gen, vec]) as p:
        for q, a, l in p.imap_unordered(_worker_task, itermentions(corpus)):
            q_vecs.extend(q)
            a_vecs.extend(a)
            labels.extend(l)
    logging.info('converting lists to 2D numpy arrays...')
    q, a, l = np.array(q_vecs), np.array(a_vecs), np.array(labels)
    logging.info('done loading')
    return q, a, l


# Global variables are necessary to allow Pool workers re-using the same
# instances across all tasks.
# https://stackoverflow.com/a/10118250
CAND_GEN = None
VECTORIZER = None

def _set_global_instances(cand_gen, vectorizer):
    global CAND_GEN
    global VECTORIZER
    CAND_GEN = cand_gen
    VECTORIZER = vectorizer


def _worker_task(item):
    mention, ref_ids = item
    q, a, labels = [], [], []
    vec_q = VECTORIZER.vectorize(mention)
    for candidate, label in CAND_GEN.samples(mention, ref_ids):
        vec_a = VECTORIZER.vectorize(candidate)
        q.append(vec_q)
        a.append(vec_a)
        labels.append((float(label),))
    return q, a, labels
