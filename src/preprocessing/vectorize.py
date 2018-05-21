#!/usr/bin/env python3
# coding: utf8

# Author: Lenz Furrer, 2018


'''
Convert text to integer vectors.
'''


import logging
import multiprocessing as mp
from string import punctuation

import numpy as np

from .load import load_data
from ..candidates.generate_candidates import candidate_generator


def load(conf, voc_index, dataset, subset):
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
        for q, a, l in p.imap_unordered(_worker_task, _itermentions(corpus)):
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


def _itermentions(corpus):
    for doc in corpus:
        for sec in doc['sections']:
            for mention in sec['mentions']:
                yield mention['text'], mention['id']


def _tokenize(text):
    return text.split()


class Vectorizer:
    '''
    Converter text -> vector.
    '''

    def __init__(self, conf, vocab):
        self.vocab = vocab
        self.length = conf.rank.sample_size  # max number of tokens per vector
        if conf.general.vectorizer_cache:  # trade speed for memory?
            self._cache = {}
        else:
            self.vectorize = self._vectorize  # hide the cache wrapper method

    def vectorize(self, text):
        '''
        Convert a piece of text to a fixed-length integer vector.
        '''
        try:
            vector = self._cache[text]
        except KeyError:
            vector = self._cache[text] = self._vectorize(text)
        return vector

    def _vectorize(self, text):
        vector = list(self._lookup(_tokenize(text)))
        # Pad or truncate the vector to the required size.
        if len(vector) < self.length:
            vector.extend(0 for _ in range(len(vector), self.length))
        elif len(vector) > self.length:
            vector[self.length:] = []
        return np.array(vector)

    def _lookup(self, tokens):
        for token in tokens:
            for variant in self._lookup_variants(token):
                try:
                    yield self.vocab[variant]
                except KeyError:
                    continue
                break
            # Skip tokens that cannot be found.

    @staticmethod
    def _lookup_variants(token):
        yield token
        yield token.lower()
        token = token.strip(punctuation)  # covers only ASCII punctuation
        yield token
        yield token.lower()
