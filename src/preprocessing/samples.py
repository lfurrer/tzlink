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
        self.vectorizer = None
        self.cand_gen = None
        self._pool = None
        self._load()

    def _load(self):
        logging.info('loading pretrained embeddings...')
        self.voc_index, self.emb_matrix = load_wemb(self.conf)
        logging.info('loading terminology...')
        self.terminology = load_dict(self.conf, self.conf.general.dataset)
        logging.info('loading vectorizer...')
        self.vectorizer = Vectorizer(self.conf, self.voc_index)
        logging.info('loading candidate generator...')
        self.cand_gen = candidate_generator(self)

    @property
    def pool(self):
        '''Multiprocessing pool for parallel sample generation.'''
        if self._pool is None:
            logging.info('initializing multiprocessing pool')
            self._pool = mp.Pool(self.conf.candidates.workers,
                                 initializer=_set_global_instances,
                                 initargs=[self.cand_gen, self.vectorizer])
        return self._pool

    def training_samples(self):
        '''Default-value wrapper around self.samples().'''
        subset = self.conf.general.training_subset
        oracle = bool(self.conf.candidates.oracle in ('train', 'both'))
        return self.samples(subset, oracle)

    def prediction_samples(self):
        '''Default-value wrapper around self.samples().'''
        subset = self.conf.general.prediction_subset
        oracle = bool(self.conf.candidates.oracle in ('predict', 'both'))
        return self.samples(subset, oracle)

    def samples(self, subset, oracle):
        '''
        Create vectorized samples with labels for training or prediction.
        '''
        corpus = load_data(self.conf, self.conf.general.dataset, subset)
        return self._samples(corpus, oracle)

    def _samples(self, corpus, oracle):
        occurrences = []
        weights = []
        samples = []  # holds 5-tuples <x_q, x_a, scores, y, ids>
        for item, numbers in self._itercandidates(corpus, oracle):
            (mention, ref_ids), occs = item
            offset, length = len(samples), len(numbers)
            for occ in occs:
                occurrences.append((*occ, mention, ref_ids, offset, offset+length))
            samples.extend(numbers)
            weights.extend(len(occs) for _ in range(length))
        data = DataSet(occurrences, weights, *zip(*samples))
        logging.info('generated %d pair-wise samples (%d with duplicates)',
                     len(data.y), sum(data.weights))
        return data

    def _itercandidates(self, corpus, oracle):
        logging.info('loading corpus...')
        mentions = _deduplicated(corpus, self.terminology)
        workers = self.conf.candidates.workers
        logging.info('generating candidates with %d workers...', workers)
        if workers >= 1:
            # Group into chunks and flatten the result.
            chunks = [(chunk, oracle) for chunk in self._chunks(mentions, 100)]
            vectorized = self.pool.imap(_worker_task, chunks)
            vectorized = (elem for chunk in vectorized for elem in chunk)
        else:
            vectorized = _task(mentions, oracle, self.cand_gen, self.vectorizer)
        yield from zip(mentions.items(), vectorized)

    @staticmethod
    def _chunks(items, size):
        chunk = []
        for item in items:
            chunk.append(item)
            if len(chunk) == size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk


class DataSet:
    '''
    Container for original and vectorized input/output data.

    Attributes:
        x_q, x_a, y: vocabulary vectors of question and
            answer side, and the labels (2D numpy arrays)
        x: the list [x_q, x_a]
        weights: counts of repeated samples (1D numpy array)
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
    def __init__(self, occs, weights, x_q, x_a, x_scores, y, ids):
        # Original data.
        self.occs = occs
        # Vectorized data.
        self.x_q = np.array(x_q)
        self.x_a = np.array(x_a)
        self.x_scores = np.array(x_scores)
        self.y = np.array(y)
        self.weights = np.array(weights)  # repetition counts
        self.scores = None  # hook for predictions
        # A set of candidate IDs for each sample.
        self.ids = ids

    @property
    def x(self):
        '''List of input tensors.'''
        return [self.x_q, self.x_a, self.x_scores]


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


def _worker_task(arg):
    return list(_task(*arg, CAND_GEN, VECTORIZER))


def _task(items, oracle, cand_gen, vectorizer):
    for mention, samples in cand_gen.samples_many(items, oracle):
        vec_q = vectorizer.vectorize(mention)
        data = []
        for cand, score, label in samples:
            vec_a = vectorizer.vectorize(cand)
            data.append((
                vec_q,
                vec_a,
                score,
                (float(label),),
                cand_gen.terminology.ids([cand]),
            ))
        yield data
