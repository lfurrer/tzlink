#!/usr/bin/env python3
# coding: utf8


"""
An early-stopping callback based on ranking accuracy.
"""


import io
import sys
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

    evaltype = Evaluator

    def __init__(self, conf, val_data, dumpfn, evalparams):
        super().__init__()

        self.conf = conf
        self.val_data = val_data
        self.dumpfn = dumpfn
        self.evalparams = evalparams

        self.wait = 0
        self.stopped_epoch = 0
        self.best = None
        self.summary = io.StringIO()

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        self.best = self.conf.stop.baseline

    def on_epoch_end(self, epoch, logs=None):
        evaluator = self.evaluate()
        current = evaluator.accuracy
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
            self.summary.truncate(0)
            evaluator.summary(self.summary)
            evaluator.dump_predictions()

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            logging.info('Epoch %05d: early stopping', self.stopped_epoch + 1)
        for file in self.evalparams.get('summary', (sys.stdout,)):
            file.write(self.summary.getvalue())

    def evaluate(self):
        '''
        Compute accuracy with the current model.
        '''
        self.val_data.scores = self.model.predict(
            self.val_data.x, batch_size=self.conf.rank.batch_size)
        writers = self.evalparams.get('predict', ())
        return self.evaltype.from_data(self.conf, self.val_data, writers)
