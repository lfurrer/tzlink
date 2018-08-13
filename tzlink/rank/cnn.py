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
    inp_mentions, sem_mentions = _semantic_repr(conf, sampler, 'sample_size')
    inp_context, sem_context = _semantic_repr(conf, sampler, 'context_size')
    inp_scores = Input(shape=(sampler.cand_gen.scores,))
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


def _semantic_repr(conf, sampler, size):
    # Embedding layers are shared between Q and A, but not between mentions
    # and context, because the text size differs.
    emb = list(_embedding_info(conf, sampler, size))
    if not emb:
        return [], []
    nodes = (_semantic_layers(conf, emb) for _ in range(2))
    (inp_q, inp_a), sem = zip(*nodes)
    return inp_q + inp_a, list(sem)  # zip returns tuples


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


def _embedding_info(conf, sampler, size_name):
    for emb in conf.rank.embeddings:
        size = conf[emb][size_name]
        if not size:  # sample/context size is 0 -> omit entirely
            continue
        matrix = sampler.emb[emb].emb_matrix
        emb_layer = _embedding_layer(conf[emb], matrix, input_length=size)
        yield size, emb_layer


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
