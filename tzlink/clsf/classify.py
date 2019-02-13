#!/usr/bin/env python3
# coding: utf8

# Author: Lenz Furrer, 2019


'''
CNN for classifying mentions as concepts.
'''


import logging

import numpy as np
from keras.models import Model
from keras.layers import Dense
from keras.callbacks import EarlyStopping

from ..preprocessing.samples import Sampler, _deduplicated
from ..rank import cnn
from ..datasets.load import load_data
from .callback import EarlyStoppingClassificationAccuracy as AccEarlyStopping


def run_training(conf, dumpfn, **evalparams):
    '''
    Train a model and evaluate/predict.
    '''
    resources = Sampler(conf)
    labelset = getlabels(conf, resources)
    logging.info('compiling model architecture...')
    model = _create_model(conf, resources, len(labelset))

    # Pretraining.
    logging.info('pretrain on terminology entries...')
    pre_data = pre_samples(conf, resources, labelset)
    earlystopping = EarlyStopping('acc', patience=conf.stop.patience)
    _fit_model(conf, model, pre_data, callbacks=[earlystopping])

    # Actual training.
    logging.info('train on annotated corpus data...')
    tr_data, val_data = (f(conf, resources, labelset)
                         for f in (tr_samples, val_samples))
    earlystopping = AccEarlyStopping(conf, val_data, dumpfn, evalparams)
    _fit_model(conf, model, tr_data, callbacks=[earlystopping])
    logging.info('done.')


def _create_model(conf, resources, nlabels):
    embs = {e: cnn.embedding_layer(conf[e], resources.emb[e].emb_matrix)
            for e in conf.rank.embeddings}
    emb_info = [(size, embs[e]) for e, size in _input_sizes(conf)]
    inp, sem = cnn.semantic_layers(conf, emb_info)
    hidden = Dense(300, activation=conf.rank.activation)(sem)
    out = Dense(nlabels, activation='softmax')(hidden)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer=conf.rank.optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def _fit_model(conf, model, data, **kwargs):
    model.fit(data.x, data.y, sample_weight=data.weights,
              epochs=conf.rank.epochs, batch_size=conf.rank.batch_size,
              **kwargs)


def getlabels(conf, rsrc):
    '''
    An ordered list of prediction labels.

    These strings correspond to the last layer of the network.
    '''
    ids = rsrc.terminology.ids(rsrc.terminology.iter_names())
    ids.discard(conf.general.nil_symbol)
    labels = [conf.general.nil_symbol, *sorted(ids)]
    return labels


def pre_samples(conf, rsrc, labels, skip_nil=False):
    '''A DataSet of pretraining samples.'''
    logging.info('collecting names from terminology...')
    samples = list(_pretraining_samples(conf, rsrc, labels, skip_nil))
    x, y = vectorize(conf, rsrc, samples, len(labels))
    return DataSet(labels=labels, x=x, y=y)


def _pretraining_samples(conf, rsrc, labels, skip_nil=False):
    for nid, cid in enumerate(labels):
        if skip_nil and cid == conf.general.nil_symbol:
            continue
        # Train the names and definitions ("context") separately.
        for name in rsrc.terminology.names([cid]):
            yield (nid, name, '')
        for def_ in rsrc.terminology.definitions(id_=cid):
            if def_:  # skip empty definitions
                yield (nid, '', def_)


def tr_samples(conf, rsrc, labelset):
    '''A DataSet of corpus samples for training.'''
    return _corpus_data(conf, rsrc, labelset, conf.general.training_subset)


def val_samples(conf, rsrc, labelset):
    '''A DataSet of corpus samples for validation.'''
    return _corpus_data(conf, rsrc, labelset, conf.general.prediction_subset)


def _corpus_data(conf, rsrc, labels, subset):
    logging.info('collecting corpus samples from %s set...', subset)
    data = DataSet(labels=labels, mentions=[], refids=[], occs=[])
    samples = []
    cid2nid = {cid: nid for nid, cid in enumerate(labels)}
    corpus = load_data(conf, subset, rsrc.terminology)
    for (text, refs, context), occs in _deduplicated(corpus).items():
        nid = _label_index(refs, cid2nid)
        samples.append((nid, text, context))
        data.mentions.append(text)
        data.refids.append(refs)
        data.occs.append(occs)
    data.x, data.y = vectorize(conf, rsrc, samples, len(labels))
    return data


def _label_index(refs, cid2nid):
    '''
    Get the index of this label.

    Usually, there is only one label, but composite mentions
    get multiple labels.  Therefore, a list of indices is
    returned always.  Because of numpy0s advanced slice
    assignment, we can still write `y[nids] = 1`.
    '''
    # If there are multiple alternative reference IDs, get the
    # first one (alphabetically lowest).
    # In evaluation, any of the alternatives is regarded as correct.
    id_ = sorted(refs)[0]
    try:
        # Model composite mentions by setting multiple nodes to 1.
        nids = [cid2nid[i] for i in id_.split('|')]
    except KeyError:
        # Don't model multiple-concept annotations (using "+").
        nids = [0]  # NIL
    return nids


def vectorize(conf, rsrc, samples, nlabels):
    '''Produce input/output matrices for Keras.'''
    logging.info('vectorizing %d samples...', len(samples))
    xdata = [np.zeros((len(samples), size), dtype=int)
             for _, size in _input_sizes(conf)]
    ydata = np.zeros((len(samples), nlabels), dtype=int)
    vectorizers = rsrc.vectorizers
    def _vectors(text, context):
        for t, vs in ((text, vectorizers['mention']),
                      (context, vectorizers['context'])):
            for v in vs:  # vs == [] if the corresponding size == 0
                yield v.vectorize(t)

    for i, (label, text, context) in enumerate(samples):
        ydata[i][label] = 1
        for x, vec in zip(xdata, _vectors(text, context)):
            x[i] = vec

    return xdata, ydata


class DataSet:
    '''
    Container for original and vectorized input/output data.
    '''
    def __init__(self, **kwargs):
        # Original data.
        self.labels = None    # label set: sorted list of strings
        self.mentions = None  # list of strings
        self.refids = None    # list of RefID
        self.occs = None      # list of lists of triples <doc, start, end>
        # Vectorized data.
        self.x = None         # input sequences: list of 2D arrays
        self.y = None         # 1-hot encoded labels: 2D array
        self.scores = None    # hook for predictions
        # Consume constructor info.
        self.__dict__.update(kwargs)

    @property
    def weights(self):
        '''Repetition counts.'''
        if self.occs is None:
            return None
        else:
            return np.array(list(map(len, self.occs)))

    def zip(self, scores=False):
        '''
        Iterate over non-vectorized sample tuples.

        If scores is False, iterate over triples
            <mention, ref, occs>.
        If scores is True, iterate over quadruples
            <mention, ref, occs, score>.
        '''
        scores = [self.scores] if scores else []
        return zip(self.mentions, self.refids, self.occs, *scores)


def _input_sizes(conf):
    for size_name in ('sample_size', 'context_size'):
        for e in conf.rank.embeddings:
            size = conf[e][size_name]
            if size:
                yield e, size
