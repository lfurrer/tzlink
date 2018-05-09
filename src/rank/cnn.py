#!/usr/bin/env python3
# coding: utf8

# Author: Lenz Furrer, 2018


import os
import argparse

from keras.models import Model
from keras.layers import Input, Dense, Flatten, Concatenate, Dot
from keras.layers import Conv1D, GlobalMaxPooling1D, Embedding

from gensim.models.keyedvectors import KeyedVectors


HERE = os.path.dirname(__file__)

SAMPLE_SIZE = 100  # tokens per mention

EMBEDDING_FN = os.path.join(HERE, 'data', 'wvec_50.bin')
EMBEDDING_DIM = 50  # fallback if not using pretrained embeddings
EMBEDDING_VOC = 10000  # fallback

N_KERNELS = 50  # number of filters in the convolution
FILTER_WIDTH = 3
ACTIVATION = 'tanh'  # used in convolution and hidden layer


def main():
    '''
    Run as script.
    '''
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument()
    args = ap.parse_args()


def _run():
    emb_lookup, emb_matrix = _load_embeddings()
    model = _create_model(emb_matrix)


def _create_model(embeddings=None):
    inp_q, inp_a = (Input(shape=SAMPLE_SIZE) for _ in range(2))
    emb = _embedding_layer(embeddings)
    sem_q = _semantic_layers(emb(inp_q))
    sem_a = _semantic_layers(emb(inp_a))
    v_sem = Dot(-1)(sem_q, sem_a)
    join_layer = Concatenate()([sem_q, v_sem, sem_a])
    hidden_layer = Dense(units=1+2*N_KERNELS, activation=ACTIVATION)(join_layer)
    logistic_regression = Dense(units=2, activation='softmax')(hidden_layer)

    model = Model(inputs=(inp_q, inp_a), outputs=logistic_regression)
    return model


def _load_embeddings(fn=EMBEDDING_FN):
    wv = KeyedVectors.load_word2vec_format(fn)
    lookup = {w: i for i, w in enumerate(wv.index2word)}
    return lookup, wv.syn0


def _embedding_layer(matrix=None):
    if matrix is not None:
        vocab_size, embedding_dim = matrix.shape
        layer = Embedding(vocab_size,
                          embedding_dim,
                          weights=[matrix],
                          input_length=SAMPLE_SIZE,
                          trainable=False)
    else:
        layer = Embedding(EMBEDDING_VOC,
                          EMBEDDING_DIM,
                          input_length=SAMPLE_SIZE)
    return layer


def _semantic_layers(x):
    x = Conv1D(N_KERNELS,
               kernel_size=FILTER_WIDTH,
               activation=ACTIVATION,
              )(x)
    x = GlobalMaxPooling1D()(x)
    x = Flatten()(x)
    return x


if __name__ == '__main__':
    main()
