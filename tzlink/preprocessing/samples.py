#!/usr/bin/env python3
# coding: utf8

# Author: Lenz Furrer, 2018


'''
Create training samples from input documents.
'''


import logging
import multiprocessing as mp
from collections import defaultdict, namedtuple

import numpy as np

from .load import load_data, load_dict
from .vectorize import load_wemb, Vectorizer
from .overlap import TokenOverlap
from ..candidates.generate_candidates import candidate_generator
from ..util.util import CacheDict


class Sampler:
    '''
    Central unit for creating training samples.

    Includes all preprocessing steps.
    Holds references to embedding vocab and matrix,
    terminology index, candidate generator, etc.
    '''
    def __init__(self, conf):
        self.conf = conf

        self.terminology = None
        self.emb = CacheDict(self._load_embeddings)
        self.vectorizers = None
        self.emb_matrices = None
        self.cand_gen = None
        self._pool = None
        self._load()

    def _load(self):
        logging.info('loading terminology...')
        self.terminology = load_dict(self.conf)
        self.vectorizers, self.emb_matrices = \
            zip(*(self.emb[emb] for emb in self.conf.rank.embeddings))
        logging.info('loading candidate generator...')
        self.cand_gen = candidate_generator(self)

    def _load_embeddings(self, which):
        econf = self.conf[which]
        logging.info('loading pretrained embeddings...')
        voc_index, emb_matrix = load_wemb(econf)
        logging.info('loading vectorizer...')
        vectorizer = Vectorizer(econf, voc_index)
        return EmbeddingInfo(vectorizer, emb_matrix)

    @property
    def pool(self):
        '''Multiprocessing pool for parallel sample generation.'''
        if self._pool is None:
            logging.info('initializing multiprocessing pool')
            self._pool = mp.Pool(self.conf.candidates.workers,
                                 initializer=_set_global_instances,
                                 initargs=[self.cand_gen, self.vectorizers])
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
        corpus = load_data(self.conf, subset, self.terminology)
        return self._samples(corpus, oracle)

    def _samples(self, corpus, oracle):
        ranges = []
        weights = []
        samples = []  # holds 7-tuples <x_q, x_a, score, overlap, y, cand, ids>
        for item, numbers in self._itercandidates(corpus, oracle):
            (mention, ref), occs = item
            offset, length = len(samples), len(numbers)
            ranges.append((offset, offset+length, mention, ref, occs))
            samples.extend(numbers)
            weights.extend(len(occs) for _ in range(length))
        data = DataSet(ranges, weights, *zip(*samples))
        logging.info('generated %d pair-wise samples (%d with duplicates)',
                     len(data.y), sum(data.weights))
        return data

    def _itercandidates(self, corpus, oracle):
        logging.info('loading corpus...')
        mentions = _deduplicated(corpus)
        workers = self.conf.candidates.workers
        logging.info('generating candidates with %d workers...', workers)
        if workers >= 1:
            # Group into chunks and flatten the result.
            chunks = [(chunk, oracle) for chunk in self._chunks(mentions, 100)]
            vectorized = self.pool.imap(_worker_task, chunks)
            vectorized = (elem for chunk in vectorized for elem in chunk)
        else:
            vectorized = _task(mentions, oracle, self.cand_gen, self.vectorizers)
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


EmbeddingInfo = namedtuple('EmbeddingInfo', 'vectorizer emb_matrix')


class DataSet:
    '''
    Container for original and vectorized input/output data.

    Attributes:
        x_q, x_a: vocabulary vectors of question and answer
            side (lists of 2D numpy arrays)
            If multiple embeddings are used, then each list
            contains multiple arrays.
        x_scores: confidence scores from candidate generation
            (1D numpy array)
        x: the list [x_q_1, ... x_q_n, x_a_1, ... x_a_n, x_scores]
        y: the labels (2D numpy array)
        weights: counts of repeated samples (1D numpy array)
        scores: store the predictions here
        candidates: candidate name of each sample (list of str)
            Holds the same information as x_a, but using a
            human-readable plain-text version.
        ids: candidate IDs (list of set of str)
            A set of IDs for each sample. Most of the time,
            the set has one member only, except for the
            cases where a candidate name maps to multiple
            concepts.
        mentions: a list of mention-specific data for each
            candidate set. Each entry is a 5-tuple
                <i, j, mention, refs, occs>
            where i/j are the start/end indices wrt. to
            the rows of the sample vectors and occs is a
            list of <docid, start, end> triples identifying
            the occurrences in the corpus.
    '''
    def __init__(self, mentions, weights,
                 x_q, x_a, x_scores, x_overlap, y,
                 cands, ids):
        # Original data.
        self.mentions = mentions
        # Vectorized data.
        self.x_q = [np.array(x) for x in zip(*x_q)]
        self.x_a = [np.array(x) for x in zip(*x_a)]
        self.x_scores = np.array(x_scores)
        self.x_overlap = np.array(x_overlap)
        self.y = np.array(y)
        self.weights = np.array(weights)  # repetition counts
        self.scores = None  # hook for predictions
        # A set of candidate IDs for each sample.
        self.candidates = cands
        self.ids = ids

    @property
    def x(self):
        '''List of input tensors.'''
        return [*self.x_q, *self.x_a, self.x_scores, self.x_overlap]


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
VECTORIZERS = None

def _set_global_instances(cand_gen, vectorizers):
    global CAND_GEN
    global VECTORIZERS
    CAND_GEN = cand_gen
    VECTORIZERS = vectorizers


def _worker_task(arg):
    return list(_task(*arg, CAND_GEN, VECTORIZERS))


def _task(items, oracle, cand_gen, vectorizers):
    overlap = TokenOverlap()
    for mention, samples in cand_gen.samples_many(items, oracle):
        vec_q = [v.vectorize(mention) for v in vectorizers]
        data = []
        for cand, score, label in samples:
            vec_a = [v.vectorize(cand) for v in vectorizers]
            data.append((
                vec_q,
                vec_a,
                score,
                overlap.overlap(mention, cand),
                (float(label),),
                cand,
                cand_gen.terminology.ids([cand]),
            ))
        yield data
