#!/usr/bin/env python3
# coding: utf8

# Author: Lenz Furrer, 2018


'''
Convolutional Neural Network for ranking mention-candidate pairs.
'''


import logging
import tempfile

from keras.models import Model, load_model
from keras.layers import Input, Dense, Concatenate, Layer
from keras.layers import Conv1D, GlobalMaxPooling1D, Embedding
from keras import backend as K

from ..preprocessing import samples
from .predictions import handle_predictions
from .callback import EarlyStoppingRankingAccuracy


def run(conf, train=True, dumpfn=None, **evalparams):
    '''
    Run the CNN (incl. preprocessing).
    '''
    if dumpfn is None:
        if not train:
            raise ValueError('no model to train or load')
        with tempfile.NamedTemporaryFile(delete=False) as f:
            dumpfn = f.name

    sampler = samples.Sampler(conf)
    logging.info('preprocessing validation data...')
    val_data = sampler.prediction_samples()
    if train:
        _train(conf, sampler, val_data, dumpfn)
    logging.info('load best model...')
    model = _load(dumpfn)
    logging.info('predict scores for validation data...')
    val_data.scores = model.predict(val_data.x,
                                    batch_size=conf.rank.batch_size)
    logging.info('evaluate and/or serialize...')
    handle_predictions(conf, val_data, **evalparams)
    logging.info('done.')


def _train(conf, sampler, val_data, dumpfn):
    logging.info('compiling model architecture...')
    model = _create_model(conf, sampler.emb_matrices)
    logging.info('preprocessing training data...')
    tr_data = sampler.training_samples()
    logging.info('training CNN...')
    earlystopping = EarlyStoppingRankingAccuracy(conf, val_data, dumpfn)
    model.fit(tr_data.x, tr_data.y, sample_weight=tr_data.weights,
              validation_data=(val_data.x, val_data.y, val_data.weights),
              callbacks=[earlystopping],
              epochs=conf.rank.epochs,
              batch_size=conf.rank.batch_size)
    logging.info('done training.')


def _load(fn):
    model = load_model(fn, custom_objects={
        'PairwiseSimilarity': PairwiseSimilarity,
    })
    return model


def _create_model(conf, emb_matrices=None):
    inp_q, inp_a = [], []
    inp_score = Input(shape=(1,))  # candidate score
    emb_q, emb_a = [], []
    for matrix in emb_matrices or [None]:
        emb = _embedding_layer(conf, matrix)
        for i, e in ((inp_q, emb_q), (inp_a, emb_a)):
            inp = Input(shape=(conf.emb.sample_size,))
            i.append(inp)
            e.append(emb(inp))
    if len(emb_q) > 1:
        emb_q, emb_a = (Concatenate()(e) for e in (emb_q, emb_a))
    else:
        emb_q, emb_a = emb_q[0], emb_a[0]
    sem_q = _semantic_layers(conf, emb_q)
    sem_a = _semantic_layers(conf, emb_a)
    v_sem = PairwiseSimilarity()([sem_q, sem_a])
    join_layer = Concatenate()([sem_q, v_sem, sem_a, inp_score])
    hidden_layer = Dense(units=1+2*conf.rank.n_kernels,
                         activation=conf.rank.activation)(join_layer)
    logistic_regression = Dense(units=1, activation='sigmoid')(hidden_layer)

    model = Model(inputs=(*inp_q, *inp_a, inp_score),
                  outputs=logistic_regression)
    model.compile(optimizer=conf.rank.optimizer, loss=conf.rank.loss)
    return model


def _embedding_layer(conf, matrix=None):
    if matrix is not None:
        vocab_size, embedding_dim = matrix.shape
        layer = Embedding(vocab_size,
                          embedding_dim,
                          weights=[matrix],
                          input_length=conf.emb.sample_size,
                          trainable=conf.emb.trainable)
    else:
        layer = Embedding(conf.emb.embedding_voc,
                          conf.emb.embedding_dim,
                          input_length=conf.emb.sample_size)
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
