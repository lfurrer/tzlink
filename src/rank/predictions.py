#!/usr/bin/env python3
# coding: utf8

# Author: Lenz Furrer, 2018


'''
Dump and evaluate predictions.
'''


import sys
import csv

from ..util.util import smart_open


NIL = 'NIL'


def handle_predictions(conf, dump, evaluate, data):
    '''
    Write predictions to a TSV file and/or print evaluation figures.
    '''
    if dump:
        data.occs.sort()  # put in corpus order (easier comparison across runs)

    if evaluate is True:
        evaluate = [sys.stdout]
    elif evaluate is False:
        evaluate = []

    dumper = TSVWriter(conf, dump)
    evaluator = Evaluator()
    for entry in _itermentions(conf, data):
        dumper.write(entry)
        evaluator.update(entry)
    dumper.close()
    for file in evaluate:
        evaluator.summary(file)


def _itermentions(conf, data):
    for *annotation, refs, start, end in data.occs:
        if start != end:
            i = max(range(start, end), key=data.scores.__getitem__)
            score = data.scores[i]
            ids = data.ids[i]
        else:
            score = None
            ids = []
        all_ids = set().union(*data.ids[start:end])
        reachable = any(id_ in refs for id_ in all_ids)
        id_ = _disambiguate(conf, ids, score)
        correct = id_ in refs
        yield (*annotation, refs, id_, correct, len(ids), reachable)


def _disambiguate(conf, ids, score):
    if not ids or score < conf.rank.min_score:
        return NIL
    return min(ids)  # just pick one -- use min() to be deterministic


class TSVWriter:
    '''
    Write a TSV line for each mention.
    '''

    fields = (
        'DOC_ID',
        'START',      # document character offset
        'END',        # ditto
        'MENTION',    # text
        'REF_ID',     # 1 or more
        'PRED_ID',
        'CORRECT',    # prediction matches reference
        'N_IDS',      # number of IDs for the top-ranked candidate name
        'REACHABLE',  # reference ID is among the candidates
    )

    def __init__(self, conf, enabled):
        if enabled:
            self.write = self._write
            self.close = self._close
            self._file = smart_open(conf.logging.prediction_fn, 'w')
            self._writer = csv.writer(self._file, quotechar=None,
                                      delimiter='\t', lineterminator='\n')
            self.write(self.fields)  # add a header line
        else:
            # Dummy mode.
            self.write = lambda _: None
            self.close = lambda: None

    def _write(self, entry):
        self._writer.writerow(entry)

    def _close(self):
        self._file.close()


class Evaluator:
    '''
    Count a selection of outcomes and compute accuracy.
    '''
    def __init__(self):
        self.correct = 0
        self.total = 0
        self.unreachable = 0
        self.ambiguous = 0
        self.nocandidates = 0

    @property
    def accuracy(self):
        '''Proportion of mentions with correct top-ranked ID.'''
        return self.correct/self.total

    @classmethod
    def from_data(cls, conf, data):
        '''
        Create an already populated Evaluator instance.
        '''
        evaluator = cls()
        for entry in _itermentions(conf, data):
            evaluator.update(entry)
        return evaluator

    def update(self, entry):
        '''Update counts.'''
        *_, correct, n_ids, reachable = entry
        self.total += 1
        self.correct += correct
        self.unreachable += not reachable
        if n_ids == 0:
            self.nocandidates += 1
        elif n_ids > 1:
            self.ambiguous += 1

    def summary(self, outfile):
        '''Write an evaluation summary to outfile.'''
        labels = 'accuracy correct total unreachable nocandidates ambiguous'
        for label in labels.split():
            outfile.write('{:12} {:5}\n'.format(label, getattr(self, label)))
