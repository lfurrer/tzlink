#!/usr/bin/env python3
# coding: utf8

# Author: Lenz Furrer, 2018


'''
Convert text to integer vectors.
'''


from string import punctuation

import numpy as np


def _tokenize(text):
    return text.split()


class Vectorizer:
    '''
    Converter text -> vector.
    '''

    # Reserved rows in the embedding matrix.
    PAD = 0  # all zeros for padding
    UNK = 1  # random values for "the unknown word"

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
            vector.extend(self.PAD for _ in range(len(vector), self.length))
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
            else:
                # Handle tokens that couldn't be found.
                yield self.UNK

    @staticmethod
    def _lookup_variants(token):
        yield token
        yield token.lower()
        token = token.strip(punctuation)  # covers only ASCII punctuation
        yield token
        yield token.lower()
