#!/usr/bin/env python3
# coding: utf8

# Author: Lenz Furrer, 2018


import argparse

from keras.models import Model
from keras.layers import Input, Dense, Concatenate, Dot
from keras.layers import Conv1D, GlobalMaxPooling1D, Embedding

from ..preprocessing import word_embeddings as wemb, vectorize


def main():
    '''
    Run as script.
    '''
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument(
        '-t', '--train', action='store_true',
        help='train a new CNN ranking model '
             '(instead of loading a previously trained one)')
    ap.add_argument(
        '-p', '--predict', action='store_true',
        help='use the model to produce rankings')
    ap.add_argument(
        '-m', '--model', metavar='PATH',
        help='path for dumping and loading a trained model')
    ap.add_argument(
        '-d', '--dataset', required=True,
        help='which dataset to use')
    args = ap.parse_args()
    _run(**vars(args))


def _run(conf, train=True, predict=True, dumpfn=None, **kwargs):
    emb_lookup, emb_matrix = wemb.load(conf)
    if train:
        model = _train(conf, emb_lookup, emb_matrix, subset='train', **kwargs)
        if dumpfn is not None:
            _dump(model, dumpfn)
    else:
        model = _load(conf, dumpfn)
    if predict:
        _predict(conf, model)


def _train(conf, emb_lookup, emb_matrix, **kwargs):
    model = _create_model(conf, emb_matrix)
    x_q, x_a, y = vectorize.load(conf, emb_lookup, **kwargs)
    model.fit([x_q, x_a], y, epochs=conf.rank.epochs,
              batch_size=conf.rank.batch_size)
    return model


def _dump(model, fn):
    pass


def _load(conf, fn):
    pass


def _predict(conf, model):
    pass


def _create_model(conf, embeddings=None):
    inp_q, inp_a = (Input(shape=(conf.rank.sample_size,)) for _ in range(2))
    emb = _embedding_layer(conf, embeddings)
    sem_q = _semantic_layers(conf, emb(inp_q))
    sem_a = _semantic_layers(conf, emb(inp_a))
    v_sem = Dot(-1)([sem_q, sem_a])
    join_layer = Concatenate()([sem_q, v_sem, sem_a])
    hidden_layer = Dense(units=1+2*conf.rank.n_kernels,
                         activation=conf.rank.activation)(join_layer)
    logistic_regression = Dense(units=1, activation='sigmoid')(hidden_layer)

    model = Model(inputs=(inp_q, inp_a), outputs=logistic_regression)
    model.compile(optimizer=conf.rank.optimizer, loss=conf.rank.loss)
    return model


def _embedding_layer(conf, matrix=None):
    if matrix is not None:
        vocab_size, embedding_dim = matrix.shape
        layer = Embedding(vocab_size,
                          embedding_dim,
                          weights=[matrix],
                          input_length=conf.rank.sample_size,
                          trainable=False)
    else:
        layer = Embedding(conf.rank.embedding_voc,
                          conf.rank.embedding_dim,
                          input_length=conf.rank.sample_size)
    return layer


def _semantic_layers(conf, x):
    x = Conv1D(conf.rank.n_kernels,
               kernel_size=conf.rank.filter_width,
               activation=conf.rank.activation,
              )(x)
    x = GlobalMaxPooling1D()(x)
    return x


if __name__ == '__main__':
    main()
