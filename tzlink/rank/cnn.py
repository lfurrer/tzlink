#!/usr/bin/env python3
# coding: utf8

# Author: Lenz Furrer, 2018


'''
Convolutional Neural Network for ranking mention-candidate pairs.
'''


import logging
import tempfile

import numpy as np
from keras.models import Model, load_model
from keras.layers import Input, Dense, Concatenate, Layer
from keras.layers import Conv1D, GlobalMaxPooling1D, Embedding
from keras import backend as K

from ..preprocessing import samples
from .predictions import handle_predictions
from .callback import EarlyStoppingRankingAccuracy


def run(conf, train=True, dumpfns=(), **evalparams):
    '''
    Run the CNN (incl. preprocessing).
    '''
    if train:
        if not dumpfns:
            with tempfile.NamedTemporaryFile(delete=False) as f:
                dumpfns = [f.name]
        elif len(dumpfns) > 1:
            raise ValueError('cannot save model to multiple files')
        _run_train(conf, dumpfns[0], **evalparams)
    else:
        if not dumpfns:
            raise ValueError('no model to train or load')
        _run_predict(conf, dumpfns, **evalparams)


def _run_train(conf, dumpfn, **evalparams):
    '''
    Train a model and evaluate/predict.
    '''
    sampler = samples.Sampler(conf)
    tr_data = sampler.training_samples()
    val_data = sampler.prediction_samples()
    logging.info('compiling model architecture...')
    model = _create_model(conf, sampler)
    logging.info('training CNN...')
    earlystopping = EarlyStoppingRankingAccuracy(conf, val_data, dumpfn, evalparams)
    model.fit(tr_data.x, tr_data.y, sample_weight=tr_data.weights,
              validation_data=(val_data.x, val_data.y, val_data.weights),
              callbacks=[earlystopping],
              epochs=conf.rank.epochs,
              batch_size=conf.rank.batch_size)
    logging.info('done.')


def _run_predict(conf, dumpfns, **evalparams):
    '''
    Load a model for evaluation/predictions.
    '''
    val_data = samples.Sampler(conf).prediction_samples()
    scores = []
    for fn in dumpfns:
        logging.info('load pretrained model from %s...', fn)
        model = _load(fn)
        logging.info('predict scores for validation data...')
        scores.append(model.predict(val_data.x,
                                    batch_size=conf.rank.batch_size))
    val_data.scores = np.mean(scores, axis=0)
    logging.info('evaluate and/or serialize...')
    handle_predictions(conf, val_data, **evalparams)
    logging.info('done.')


def _load(fn):
    model = load_model(fn, custom_objects={
        'PairwiseSimilarity': PairwiseSimilarity,
    })
    return model


def _create_model(conf, sampler):
    # Embedding layers are shared among all inputs.
    emb = [_embedding_layer(conf[emb], sampler.emb[emb].emb_matrix)
           for emb in conf.rank.embeddings]
    inp_mentions, sem_mentions = _semantic_repr_qa(conf, emb, 'sample_size')
    inp_context, sem_context = _semantic_repr_qa(conf, emb, 'context_size')
    inp_scores = Input(shape=(len(conf.candidates.generator.split('\n')),))
    inp_overlap = Input(shape=(1,))  # token overlap between q and a

    v_sem = PairwiseSimilarity()(sem_mentions)
    join_layer = Concatenate()(
        [*sem_mentions, v_sem, *sem_context, inp_scores, inp_overlap])
    hidden_layer = Dense(units=K.int_shape(join_layer)[-1],
                         activation=conf.rank.activation)(join_layer)
    logistic_regression = Dense(units=1, activation='sigmoid')(hidden_layer)

    model = Model(inputs=(*inp_mentions, *inp_context, inp_scores, inp_overlap),
                  outputs=logistic_regression)
    model.compile(optimizer=conf.rank.optimizer, loss=conf.rank.loss)
    return model


def _semantic_repr_qa(conf, emb_layers, size_name):
    sizes = [conf[emb][size_name] for emb in conf.rank.embeddings]
    emb_info = [(s, e) for s, e in zip(sizes, emb_layers) if s]
    if not emb_info:  # sample/context size is 0 -> omit entirely
        return [], []
    nodes = (_semantic_layers(conf, emb_info) for _ in range(2))
    (inp_q, inp_a), (sem_q, sem_a) = zip(*nodes)
    return inp_q + inp_a, [sem_q, sem_a]


def _semantic_layers(conf, emb_info):
    inp, emb = _word_layer(emb_info)
    sem = _conv_pool_layers(conf, emb)
    return inp, sem


def _word_layer(emb_info):
    inp, emb = [], []
    for size, emb_layer in emb_info:
        i = Input(shape=(size,))
        inp.append(i)
        emb.append(emb_layer(i))
    emb = _conditional_concat(emb)
    return inp, emb


def _embedding_layer(econf, matrix=None, **kwargs):
    if matrix is not None:
        args = matrix.shape
        kwargs.update(weights=[matrix],
                      trainable=econf.trainable)
    else:
        args = (econf.embedding_voc, econf.embedding_dim)
    return Embedding(*args, **kwargs)


def _conv_pool_layers(conf, inputs):
    outputs = [_convolution_pooling(conf, width, inputs)
               for width in conf.rank.filter_width]
    return _conditional_concat(outputs)


def _convolution_pooling(conf, width, x):
    x = Conv1D(conf.rank.n_kernels,
               kernel_size=width,
               activation=conf.rank.activation,
              )(x)
    x = GlobalMaxPooling1D()(x)
    return x


def _conditional_concat(layers):
    if len(layers) > 1:
        return Concatenate()(layers)
    return layers[0]


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
