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

from .word_embeddings import load as load_wemb
from .load import load_data, load_dict
from .vectorize import Vectorizer
from ..candidates.generate_candidates import candidate_generator


class Sampler:
    '''
    Central unit for creating training samples.

    Includes all preprocessing steps.
    Holds references to embedding vocab and matrix,
    terminology index, candidate generator, etc.
    '''
    def __init__(self, conf):
        self.conf = conf

        self.voc_index = None
        self.emb_matrix = None
        self.terminology = None
        self._pool = None
        self._load()

    def _load(self):
        logging.info('loading pretrained embeddings...')
        self.voc_index, self.emb_matrix = load_wemb(self.conf)
        logging.info('loading terminology...')
        self.terminology = load_dict(self.conf, self.conf.general.dataset)
        logging.info('loading candidate generator...')
        cand_gen = candidate_generator(self.conf, self.terminology)
        logging.info('loading vectorizer...')
        vectorizer = Vectorizer(self.conf, self.voc_index)
        logging.info('initializing multiprocessing pool')
        self._pool = mp.Pool(self.conf.candidates.workers,
                             initializer=_set_global_instances,
                             initargs=[cand_gen, vectorizer])

    def training_samples(self, subset='train', oracle=True):
        '''Default-value wrapper around self.samples().'''
        return self.samples(subset, oracle)

    def prediction_samples(self, subset='dev', oracle=False):
        '''Default-value wrapper around self.samples().'''
        return self.samples(subset, oracle)

    def samples(self, subset, oracle):
        '''
        Create vectorized samples with labels for training or prediction.
        '''
        corpus = load_data(self.conf, self.conf.general.dataset, subset)
        return self._samples(corpus, oracle)

    def _samples(self, corpus, oracle):
        occurrences = []
        accumulators = [[], [], [], []]
        for item, vectors in self._itercandidates(corpus, oracle):
            (mention, ref_ids), occs = item
            for occ in occs:
                offset, length = len(accumulators[0]), len(vectors[0])
                occurrences.append((*occ, mention, ref_ids, offset, offset+length))
                for accu, vec in zip(accumulators, vectors):
                    accu.extend(vec)
        data = DataSet(occurrences, *accumulators)
        logging.info('generated %d pair-wise samples', len(data.y))
        return data

    def _itercandidates(self, corpus, oracle):
        logging.info('loading corpus...')
        mentions = _deduplicated(corpus, self.terminology)
        logging.info('generating candidates with %d workers...',
                     self.conf.candidates.workers)
        items = [(key, oracle) for key in mentions]
        vectorized = self._pool.imap(_worker_task, items, chunksize=20)
        yield from zip(mentions.items(), vectorized)


class DataSet:
    '''
    Container for original and vectorized input/output data.

    Attributes:
        x_q, x_a, y: vocabulary vectors of question and
            answer side, and the labels (2D numpy arrays)
        x: the list [x_q, x_a]
        scores: store the predictions here
        ids: candidate IDs (list of list of str)
            Has the same length as each of the sample
            vectors (x_q, x_a, y). Each element is a
            list of strings. Most of the time, the
            nested list has one member only, except for
            the cases where a candidate name maps to
            multiple concepts.
        occs: a list of occurrence-specific data
            for each mention. Each entry is a 7-tuple
            <docid, start, end, mention, refs, i, j>
            with i/j being the start/end indices wrt.
            to the rows of the sample vectors
    '''
    def __init__(self, occs, x_q, x_a, y, ids):
        # Original data.
        self.occs = occs
        # Vectorized data.
        self.x_q = np.array(x_q)
        self.x_a = np.array(x_a)
        self.y = np.array(y)
        self.scores = None  # hook for predictions
        # A set of candidate IDs for each sample.
        self.ids = ids

    @property
    def x(self):
        '''List of input tensors.'''
        return [self.x_q, self.x_a]


def _deduplicated(corpus, terminology):
    mentions = defaultdict(list)
    for doc in corpus:
        docid = doc['docid']
        for sec in doc['sections']:
            offset = sec['offset']
            for mention in sec['mentions']:
                key = mention['text'], _canonical_ids(mention['id'], terminology)
                occ = docid, offset + mention['start'], offset + mention['end']
                mentions[key].append(occ)
    return mentions


def _canonical_ids(ids, terminology):
    canonical = tuple(
        frozenset().union(*(terminology.canonical_ids(alt) for alt in comp))
        for comp in ids
    )
    return canonical


# Global variables are necessary to allow Pool workers to re-use the same
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
    (mention, ref_ids), oracle = item
    q, a, labels, cand_ids = [], [], [], []
    vec_q = VECTORIZER.vectorize(mention)
    for candidate, label in CAND_GEN.samples(mention, ref_ids, oracle):
        vec_a = VECTORIZER.vectorize(candidate)
        q.append(vec_q)
        a.append(vec_a)
        labels.append((float(label),))
        cand_ids.append(CAND_GEN.terminology.ids([candidate]))
    return q, a, labels, cand_ids
