#!/usr/bin/env python3
# coding: utf8

# Author: Lenz Furrer, 2018


'''
Convolutional Neural Network for ranking mention-candidate pairs.
'''


import argparse

from keras.models import Model, load_model
from keras.layers import Input, Dense, Concatenate, Layer
from keras.layers import Conv1D, GlobalMaxPooling1D, Embedding
from keras import backend as K

from ..conf.config import Config
from ..preprocessing import samples
from .predictions import handle_predictions


def main():
    '''
    Run as script.
    '''
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        '-t', '--train', action='store_true',
        help='train a new CNN ranking model '
             '(instead of loading a previously trained one)')
    ap.add_argument(
        '-p', '--predict', action='store_true',
        help='use the model to produce rankings')
    ap.add_argument(
        '-m', '--model', metavar='PATH', dest='dumpfn',
        help='path for dumping and loading a trained model')
    ap.add_argument(
        '-c', '--config', metavar='PATH', nargs='+', default=(),
        help='config file(s) for overriding the defaults')
    ap.add_argument(
        '-d', '--dataset', required=True,
        help='which dataset to use')
    args = ap.parse_args()
    run(**vars(args))


def run(config, dataset, **kwargs):
    '''
    Run the CNN (incl. preprocessing).
    '''
    if not isinstance(config, Config):
        if isinstance(config, str):
            config = [config]
        config = Config(*config)
    config.general.dataset = dataset
    _run(config, **kwargs)


def _run(conf, train=True, predict=True, test=True, dumpfn=None, **kwargs):
    sampler = samples.Sampler(conf)
    if train:
        model = _train(conf, sampler, **kwargs)
        if dumpfn is not None:
            _dump(model, dumpfn)
    else:
        model = _load(dumpfn)
    if predict or test:
        data = _predict(conf, sampler, model, **kwargs)
        handle_predictions(conf, predict, test, data)


def _train(conf, sampler, **kwargs):
    model = _create_model(conf, sampler.emb_matrix)
    data = sampler.training_samples(**kwargs)
    model.fit(data.x, data.y, sample_weight=data.weights,
              epochs=conf.rank.epochs,
              batch_size=conf.rank.batch_size)
    return model


def _dump(model, fn):
    model.save(fn)


def _load(fn):
    model = load_model(fn, custom_objects={
        'PairwiseSimilarity': PairwiseSimilarity,
    })
    return model


def _predict(conf, sampler, model, **kwargs):
    data = sampler.prediction_samples(**kwargs)
    data.scores = model.predict(data.x, batch_size=conf.rank.batch_size)
    return data


def _create_model(conf, embeddings=None):
    inp_q, inp_a = (Input(shape=(conf.rank.sample_size,)) for _ in range(2))
    emb = _embedding_layer(conf, embeddings)
    sem_q = _semantic_layers(conf, emb(inp_q))
    sem_a = _semantic_layers(conf, emb(inp_a))
    v_sem = PairwiseSimilarity()([sem_q, sem_a])
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


class PairwiseSimilarity(Layer):
    '''
    Join layer with a trainable similarity matrix.

    v_sem = sim(v_q, v_a) = v_q^T M v_a
    '''

    def __init__(self, **kwargs):
        self.M = None  # set in self.build()
        super().__init__(**kwargs)

    def build(self, input_shape):
        try:
            shape_q, shape_a = input_shape
        except ValueError:
            raise ValueError('input_shape must be a 2-element list')
        self.M = self.add_weight(name='M',
                                 shape=(shape_q[1], shape_a[1]),
                                 initializer='uniform',
                                 trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        q, a = inputs
        # https://github.com/wglassly/cnnormaliztion/blob/master/src/nn_layers.py#L822
        return K.batch_dot(q, K.dot(a, K.transpose(self.M)), axes=1)

    @staticmethod
    def compute_output_shape(input_shape):
        return (input_shape[0][0], 1)


if __name__ == '__main__':
    main()
