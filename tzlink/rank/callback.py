#!/usr/bin/env python3
# coding: utf8


"""
An early-stopping callback based on ranking accuracy.
"""


import logging

from keras.callbacks import Callback

from .predictions import Evaluator


class EarlyStoppingRankingAccuracy(Callback):
    '''Stop training when ranking accuracy has stopped improving.

    Based on keras.callbacks.EarlyStopping.

    #Arguments
        conf: a conf.config.Config instance with settings.
        val_data: a preprocessing.samples.DataSet instance
            with validation data.
        dumpfn: path to save the best model.
    '''
    def __init__(self, conf, val_data, dumpfn):
        super().__init__()

        self.conf = conf
        self.val_data = val_data
        self.dumpfn = dumpfn

        self.wait = 0
        self.stopped_epoch = 0
        self.best = None

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        self.best = self.conf.stop.baseline

    def on_epoch_end(self, epoch, logs=None):
        current = self.evaluate()
        logging.info('Ranking accuracy: %g', current)
        if current - self.conf.stop.min_delta > self.best:
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.conf.stop.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
        if current > self.best:
            self.best = current
            self.model.save(self.dumpfn)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            logging.info('Epoch %05d: early stopping', self.stopped_epoch + 1)

    def evaluate(self):
        '''
        Compute accuracy with the current model.
        '''
        self.val_data.scores = self.model.predict(
            self.val_data.x, batch_size=self.conf.rank.batch_size)
        accuracy = Evaluator.from_data(self.conf, self.val_data).accuracy
        return accuracy
