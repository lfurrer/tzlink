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
    model = _create_model(conf, sampler)
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


def _create_model(conf, sampler):
    inp_words, emb_nodes = _word_layers(conf, sampler)
    inp_scores = Input(shape=(sampler.cand_gen.scores,))
    inp_overlap = Input(shape=(1,))  # token overlap between q and a

    sem_q, sem_a = (_semantic_layers(conf, e) for e in emb_nodes)
    v_sem = PairwiseSimilarity()([sem_q, sem_a])
    join_layer = Concatenate()([sem_q, v_sem, sem_a, inp_scores, inp_overlap])
    hidden_layer = Dense(units=2*conf.rank.n_kernels+sampler.cand_gen.scores+2,
                         activation=conf.rank.activation)(join_layer)
    logistic_regression = Dense(units=1, activation='sigmoid')(hidden_layer)

    model = Model(inputs=(*inp_words, inp_scores, inp_overlap),
                  outputs=logistic_regression)
    model.compile(optimizer=conf.rank.optimizer, loss=conf.rank.loss)
    return model


def _word_layers(conf, sampler):
    inp_q, inp_a = [], []
    emb_q, emb_a = [], []
    for emb in conf.rank.embeddings:
        matrix = sampler.emb[emb].emb_matrix
        emb_layer = _embedding_layer(conf[emb], matrix)
        for i, e in ((inp_q, emb_q), (inp_a, emb_a)):
            inp = Input(shape=(conf[emb].sample_size,))
            i.append(inp)
            e.append(emb_layer(inp))

    inp_nodes = inp_q + inp_a
    emb_nodes = [Concatenate()(e) if len(e) > 1 else e[0]
                 for e in (emb_q, emb_a)]

    return inp_nodes, emb_nodes


def _embedding_layer(econf, matrix=None):
    if matrix is not None:
        vocab_size, embedding_dim = matrix.shape
        layer = Embedding(vocab_size,
                          embedding_dim,
                          weights=[matrix],
                          input_length=econf.sample_size,
                          trainable=econf.trainable)
    else:
        layer = Embedding(econf.embedding_voc,
                          econf.embedding_dim,
                          input_length=econf.sample_size)
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
