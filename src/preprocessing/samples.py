#!/usr/bin/env python3
# coding: utf8

# Author: Lenz Furrer, 2018


'''
Create training samples from input documents.
'''


import logging
import multiprocessing as mp
from collections import defaultdict

import numpy as np

from .load import load_data
from .vectorize import Vectorizer
from ..candidates.generate_candidates import candidate_generator


def training_samples(conf, voc_index, dataset, subset, oracle=True):
    '''
    Create vectorized samples with labels for training.

    Args:
        conf: a config.Config instance
        voc_index (dict): mapping of words to integers
        dataset (str): dataset identifier
        subset (str): train, dev, or test

    Returns:
        triple of np.array (2D): vocabulary vectors of
            question and answer side, and the labels
    '''
    vec, *_ = _samples(conf, voc_index, dataset, subset, oracle)
    return vec


def prediction_samples(conf, voc_index, dataset, subset, oracle=False):
    '''
    Create vectorized samples for prediction.

    Args:
        same as for training_samples()

    Returns:
        a triple <vectors, cand_ids, occurrences>:
            vectors: a triple of 2D arrays as returned by
                training_samples()
            cand_ids: a list of candidate IDs
                Has the same length as each of the sample
                vectors (x_q, x_a, y). Each element is a
                list of strings. Most of the time, the
                nested list has one member only, except for
                the cases where a candidate name maps to
                multiple concepts.
            occurrences: a list of occurrence-specific data
                for each mention. Each entry is a 7-tuple
                <docid, start, end, mention, refs, i, j>
                with i/j being the start/end indices wrt.
                to the rows of the sample vectors
    '''
    return _samples(conf, voc_index, dataset, subset, oracle)


def _itercandidates(conf, voc_index, dataset, subset, oracle):
    corpus = load_data(conf, dataset, subset)
    mentions = _deduplicated(corpus)
    dict_entries = load_data(conf, dataset, 'dict')
    logging.info('loading candidate generator...')
    cand_gen = candidate_generator(conf, dict_entries)
    logging.info('loading vectorizer...')
    vec = Vectorizer(conf, voc_index)
    logging.info('distributing load to %d workers...', conf.candidates.workers)
    with mp.Pool(conf.candidates.workers,
                 initializer=_set_global_instances,
                 initargs=[cand_gen, vec, oracle]) as p:
        vectorized = p.imap(_worker_task, mentions, chunksize=20)
        yield from zip(mentions.items(), vectorized)


def _samples(conf, *args):
    occurrences = []
    accumulators = [[], [], [], []]
    for item, vectors in _itercandidates(conf, *args):
        (mention, ref_ids), occs = item
        for occ in occs:
            offset, length = len(accumulators[0]), len(vectors[0])
            occurrences.append((*occ, mention, ref_ids, offset, offset+length))
            for accu, vec in zip(accumulators, vectors):
                accu.extend(vec)
    logging.info('converting lists to 2D numpy arrays...')
    cand_ids = accumulators.pop()
    x_q, x_a, y = (np.array(a) for a in accumulators)
    logging.info('done loading')
    return (x_q, x_a, y), cand_ids, occurrences


def _deduplicated(corpus):
    mentions = defaultdict(list)
    for doc in corpus:
        docid = doc['docid']
        for sec in doc['sections']:
            offset = sec['offset']
            for mention in sec['mentions']:
                key = mention['text'], mention['id']
                occ = docid, offset + mention['start'], offset + mention['end']
                mentions[key].append(occ)
    return mentions


# Global variables are necessary to allow Pool workers to re-use the same
# instances across all tasks.
# https://stackoverflow.com/a/10118250
CAND_GEN = None
VECTORIZER = None
ORACLE = None

def _set_global_instances(cand_gen, vectorizer, oracle):
    global CAND_GEN
    global VECTORIZER
    global ORACLE
    CAND_GEN = cand_gen
    VECTORIZER = vectorizer
    ORACLE = oracle


def _worker_task(item):
    mention, ref_ids = item
    q, a, labels, cand_ids = [], [], [], []
    vec_q = VECTORIZER.vectorize(mention)
    for candidate, label in CAND_GEN.samples(mention, ref_ids, ORACLE):
        vec_a = VECTORIZER.vectorize(candidate)
        q.append(vec_q)
        a.append(vec_a)
        labels.append((float(label),))
        cand_ids.append(CAND_GEN.ids(candidate))
    return q, a, labels, cand_ids
