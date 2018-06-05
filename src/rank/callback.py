#!/usr/bin/env python3
# coding: utf8


"""
An early-stopping callback based on ranking accuracy.
"""


import numpy as np
from keras.callbacks import Callback

from .predictions import Evaluator


class EarlyStoppingRankingAccuracy(Callback):
    '''Stop training when ranking accuracy has stopped improving.

    Based on keras.callbacks.EarlyStopping.

    #Arguments
        conf: a conf.config.Config instance with settings.
        val_data: a preprocessing.samples.DataSet instance
            with validation data.
        min_delta: minimum change in the monitored quantity
            to qualify as an improvement.
        patience: number of epochs with no improvement
            after which training will be stopped.
        verbose: verbosity mode.
        baseline: Baseline value for ranking accuracy
            to reach. Training will stop if the model does
            not show improvement over the baseline.
    '''
    def __init__(self,
                 conf,
                 val_data,
                 min_delta=0,
                 patience=0,
                 verbose=0,
                 baseline=None):
        super().__init__()

        self.conf = conf
        self.val_data = val_data

        self.patience = patience
        self.verbose = verbose
        self.baseline = baseline
        self.min_delta = min_delta

        self.wait = 0
        self.stopped_epoch = 0
        self.best = None

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = self.evaluate()
        if np.greater(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))

    def evaluate(self):
        '''
        Compute accuracy with the current model.
        '''
        self.val_data.scores = self.model.predict(
            self.val_data.x, batch_size=self.conf.rank.batch_size)
        accuracy = Evaluator.from_data(self.conf, self.val_data).accuracy
        return accuracy
