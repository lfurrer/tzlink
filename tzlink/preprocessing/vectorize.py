#!/usr/bin/env python3
# coding: utf8

# Author: Lenz Furrer, 2018


'''
Convert text to numerical vectors.
'''


import re
import os
import logging
from string import punctuation

import numpy as np
from gensim.models.keyedvectors import KeyedVectors

from .tokenization import create_tokenizer
from ..datasets.load import itertext_corpus, itertext_terminology
from ..util.util import identity, CacheDict, smart_open, ConfHash


def load_wemb(conf, econf):
    '''
    Load embeddings from disk and create a lookup table.
    '''
    # Check for a cached copy of the trimmed subset.
    path = _trimmed_fn(conf, econf)
    if os.path.exists(path):
        logging.info('load trimmed vectors from %s', path)
        with np.load(path) as f:
            vocab = f['vocab']
            matrix = f['matrix']
    else:
        logging.info('trim embedding table to actual vocabulary size')
        vocab, matrix = trim_wemb(conf, econf)
        logging.info('export trimmed vectors to %s', path)
        with smart_open(path, 'wb') as f:
            np.savez_compressed(f, vocab=vocab, matrix=matrix)
    lookup = {w: i for i, w in enumerate(vocab, 2)}  # +2 for PAD and UNK
    return lookup, matrix


def trim_wemb(conf, econf):
    '''
    Trim embeddings to the minimally required size and load them.
    '''
    # Load the embeddings from disk.
    fn = econf.embedding_fn
    if fn.endswith('.kv'):
        wv = KeyedVectors.load(fn, mmap='r')
    else:
        wv = KeyedVectors.load_word2vec_format(fn, binary=fn.endswith('.bin'))

    # Account for mapping changes due to preprocessing (eg. stemming).
    vocab = _adapt_mapping(econf, wv)
    vocab = {w: e.index for w, e in vocab.items()}

    # Reduce the matrix to the actual vocabulary of the dataset.
    used = _get_dataset_vocab(conf, econf, vocab)
    mapping = {old: new for new, old in enumerate(sorted(used))}  # preserve order
    ds_vocab = [None] * len(mapping)
    # Add two rows in the beginning: one for padding and one for unknown words.
    shape = len(mapping) + 2, wv.vectors.shape[1]
    matrix = np.zeros(shape, dtype=wv.vectors.dtype)
    matrix[1] = np.random.standard_normal(shape[1])  # unknown words
    for w, i in vocab.items():
        n = mapping.get(i)
        if n is not None:
            ds_vocab[n] = w
            matrix[n+2] = wv.vectors[i]

    return ds_vocab, matrix


def _adapt_mapping(econf, wv):
    '''
    Modify the mapping of words to vectors.
    '''
    prep = get_preprocessing(econf)
    if prep is None:
        return wv.vocab

    vocab = {}
    for word, entry in wv.vocab.items():
        modified = prep(word)
        if modified not in vocab or vocab[modified].count < entry.count:
            vocab[modified] = entry
    return vocab


def _get_dataset_vocab(conf, econf, emb_vocab):
    ds_vocab = set()
    vec = Vectorizer(econf, emb_vocab, 'sample_size')
    vec.PAD = vec.UNK = None  # unset these as this vocab starts from 0
    for text in _itertext(conf):
        ds_vocab.update(vec.indices(text))
    ds_vocab.discard(None)
    return ds_vocab


def _itertext(conf):
    '''Iterate over all corpus and terminology text.'''
    yield from itertext_corpus(conf, 'all')
    yield from itertext_terminology(conf)


def _trimmed_fn(conf, econf):
    '''
    Filename with a hash key of all relevant settings.
    '''
    h = ConfHash()

    dataset = conf.general.dataset
    h.add(dataset)
    h.add(conf[dataset].corpus_dir)
    h.add(conf[dataset].dict_fn)

    h.add(econf.embedding_fn)
    h.add(econf.preprocess.lower())
    h.add(econf.tokenizer.lower())
    h.add(getattr(econf, "tokenizer_model", None))

    fn = conf.general.vocab_cache.format(h.hexdigest())
    return fn



def get_preprocessing(econf):
    '''
    Select and instantiate a preprocessor.
    '''
    name = econf.preprocess.lower()
    if name == 'none':
        return None
    if name == 'single_ws':
        pattern = re.compile(r'\s+')
        def _single_ws(text):
            return pattern.sub(' ', text)
        return _single_ws
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
