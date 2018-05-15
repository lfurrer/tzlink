#!/usr/bin/env python3
# coding: utf8

# Author: Lenz Furrer, 2018


'''
Load word embeddings.
'''


from gensim.models.keyedvectors import KeyedVectors


def load(conf):
    '''
    Load embeddings from binary and create a lookup table.
    '''
    wv = KeyedVectors.load_word2vec_format(conf.rank.embedding_fn)
    lookup = {w: i for i, w in enumerate(wv.index2word)}
    return lookup, wv.syn0
