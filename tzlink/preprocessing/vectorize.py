#!/usr/bin/env python3
# coding: utf8

# Author: Lenz Furrer, 2018


'''
Convert text to numerical vectors.
'''


import re
from string import punctuation

import numpy as np
from gensim.models.keyedvectors import KeyedVectors

from .tokenization import create_tokenizer
from ..util.util import identity, CacheDict


def load_wemb(econf):
    '''
    Load embeddings from disk and create a lookup table.
    '''
    fn = econf.embedding_fn
    if fn.endswith('.kv'):
        wv = KeyedVectors.load(fn, mmap='r')
    else:
        wv = KeyedVectors.load_word2vec_format(fn, binary=fn.endswith('.bin'))
    vocab, matrix = _adapt_mapping(econf, wv)

    # Add two rows in the beginning: one for padding and one for unknown words.
    dim = matrix.shape[1]
    dtype = matrix.dtype
    padding = np.zeros(dim, dtype)
    unknown = np.random.standard_normal(dim).astype(dtype)
    matrix = np.concatenate([[padding, unknown], matrix])
    lookup = {w: i for i, w in enumerate(vocab, 2)}
    return lookup, matrix


def _adapt_mapping(econf, wv):
    '''
    Modify the mapping of words to vectors.
    '''
    prep = get_preprocessing(econf)
    if prep is None:
        return wv.index2word, wv.syn0

    vocab = {}
    for word, entry in wv.vocab.items():
        modified = prep(word)
        if modified not in vocab or vocab[modified].count < entry.count:
            vocab[modified] = entry
    indices = [e.index for e in vocab.values()]
    return vocab.keys(), wv.syn0[indices]


def get_preprocessing(econf):
    '''
    Select and instantiate a preprocessor.
    '''
    name = econf.preprocess.lower()
    if name == 'none':
        return None
    if name == 'stem':
        from .stem import PorterStemmer
        stemmer = PorterStemmer()
        pattern = re.compile(r'\w+')
        def _stem(text):
            return pattern.sub(lambda m: stemmer.stem(m.group(0)), text)
        return _stem
    raise ValueError('unknown preprocessing: {}'.format(name))


def get_tokenizer(econf):
    '''
    Select and instantiate a tokenizer.
    '''
    name = econf.tokenizer.lower()
    model = getattr(econf, "tokenizer_model", None)
    return create_tokenizer(name, model)


class Vectorizer:
    '''
    Converter text -> vector of integers.
    '''

    # Reserved rows in the embedding matrix.
    PAD = 0  # all zeros for padding
    UNK = 1  # random values for "the unknown word"

    def __init__(self, econf, vocab, size_name):
        self.vocab = vocab
        self.length = econf[size_name]  # max number of tokens per vector
        self._preprocess = get_preprocessing(econf) or identity
        self._tokenize = get_tokenizer(econf)
        if econf.vectorizer_cache:  # trade memory for speed?
            self._cache = CacheDict(self._vectorize)
            self._call_vectorize = self._cache.__getitem__
        else:
            self._call_vectorize = self._vectorize

    def vectorize(self, text):
        '''
        Convert a piece of text to a fixed-length integer vector.
        '''
        return self._call_vectorize(text)

    def _vectorize(self, text):
        vector = self.indices(text)
        # Pad or truncate the vector to the required size.
        if len(vector) < self.length:
            vector.extend(self.PAD for _ in range(len(vector), self.length))
        elif len(vector) > self.length:
            vector[self.length:] = []
        return np.array(vector)

    def indices(self, text):
        '''
        Convert a piece of text to a variable-length list of int.
        '''
        return list(self._lookup(self._tokenize(self._preprocess(text))))

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
