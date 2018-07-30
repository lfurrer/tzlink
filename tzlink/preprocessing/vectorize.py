#!/usr/bin/env python3
# coding: utf8

# Author: Lenz Furrer, 2018


'''
Convert text to integer vectors.
'''


import re
from string import punctuation

import numpy as np


def get_tokenizer(econf):
    '''
    Select and instantiate a tokenizer.
    '''
    name = econf.tokenizer.lower()
    model = getattr(econf, "tokenizer_model", None)
    return _get_tokenizer(name, model)


def _get_tokenizer(name, model):
    if name == 'whitespace':
        # Simply split on whitespace.
        return str.split
    if name == 'charclass':
        # Tokenize on change of character class.
        pattern = re.compile(
            r'''\d+|            # match contiguous runs of digits
                [^\W\d_]+|      # or letters
                (?:[^\w\s]|_)+  # or other non-whitespace characters
                ''', re.VERBOSE)
        return pattern.findall
    if name == 'bpe':
        from subword_nmt.apply_bpe import BPE
        pretokenizer = _get_tokenizer('charclass', None)
        with open(model, encoding='utf8') as f:
            bpe = BPE(f)
        def _tokenize(text):
            pretok = ' '.join(pretokenizer(text))
            tokens = bpe.segment(pretok)
            return tokens.split()
        return _tokenize


class Vectorizer:
    '''
    Converter text -> vector.
    '''

    # Reserved rows in the embedding matrix.
    PAD = 0  # all zeros for padding
    UNK = 1  # random values for "the unknown word"

    def __init__(self, econf, vocab):
        self.vocab = vocab
        self.length = econf.sample_size  # max number of tokens per vector
        self._tokenize = get_tokenizer(econf)
        if econf.vectorizer_cache:  # trade memory for speed?
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
        return list(self._lookup(self._tokenize(text)))

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
