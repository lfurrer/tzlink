#!/usr/bin/env python3
# coding: utf8

# Author: Lenz Furrer, 2018


from string import punctuation

import numpy as np

from .load import load_data
from ..rank.generate_candidates import CandidateGenerator


def load(conf, voc_index, dataset, subset):
    corpus = load_data(conf, dataset, subset)
    dict_entries = load_data(conf, dataset, 'dict')
    cand_gen = CandidateGenerator(dict_entries)
    vec = Vectorizer(conf, voc_index)
    q, a, labels = [], [], []
    for mention, ref_ids in _itermentions(corpus):
        vec_q = vec.vectorize(mention)
        for candidate, label in cand_gen.candidates(mention, ref_ids):
            vec_a = vec.vectorize(candidate)
            q.append(vec_q)
            a.append(vec_a)
            labels.append((float(label), float(not label)))  # 1-hot binary
    return np.array(q), np.array(a), np.array(labels)


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
        if conf.general.vectorizer_cache:
            self._cache = {}
        else:
            self.vectorize = self._vectorize

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
