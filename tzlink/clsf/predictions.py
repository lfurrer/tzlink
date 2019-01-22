#!/usr/bin/env python3
# coding: utf8

# Author: Lenz Furrer, 2019


'''
Dump and evaluate predictions.
'''


import numpy as np

from ..rank.predictions import BaseEvaluator, SummaryWriter as BaseWriter
from ..rank.predictions import _rm_tabs_nl


class SummaryWriter(BaseWriter):
    '''
    Write a summary line for each occurrence of a mention.
    '''

    fields = (
        'DOC_ID',
        'START',      # document character offset
        'END',        # ditto
        'MENTION',    # text
        'REF_ID',     # 1 or more
        'PRED_ID',
        'SCORE',      # classification confidence
        'CORRECT',    # prediction matches reference
    )

    def update(self, mention, occs, *info):
        '''Update with outcome information per occurrence.'''
        mention = _rm_tabs_nl(mention)
        for occ in occs:
            self._entries.append((*occ, mention, *info))


class Evaluator(BaseEvaluator):
    '''
    Count a selection of outcomes and compute accuracy.
    '''
    _writer_names = {
        'summary': SummaryWriter,
    }

    def evaluate(self, data):
        '''
        Compute accuracy with the current model.
        '''
        for mention, ref, occs, pred in data.zip(scores=True):
            i = np.argmax(pred)
            id_ = data.labels[i]
            score = pred[i]
            outcome = id_ in ref

            self.total += len(occs)
            self.correct += outcome*len(occs)

            for writer in self.writers:
                writer.update(mention, occs, ref, id_, score, outcome)
