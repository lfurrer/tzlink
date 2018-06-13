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
        all_ids = data.ids[start:end]
        scored = sorted(zip(data.scores[start:end], all_ids), reverse=True)
        id_ = _disambiguate(conf, scored)
        correct = id_ in refs
        n_ids = len(scored) and len(scored[0][1])
        reachable = any(id_ in refs for id_ in set().union(*all_ids))
        yield (*annotation, refs, id_, correct, n_ids, reachable)


def _disambiguate(conf, scored):
    try:
        score, top_ids = scored[0]
    except IndexError:
        # No candidates.
        return NIL

    if score < conf.rank.min_score:
        # Score too low.
        return NIL

    if len(top_ids) == 1:
        # Unambiguous case.
        return next(iter(top_ids))

    # Look for cues in the lower-ranked candidates.
    for _, ids in scored[1:]:
        common = top_ids.intersection(ids)
        if common:
            # Another name of the top-ranked concept(s) was among the
            # candidates. Remove the concepts to which it doesn't map.
            top_ids = common
            if len(top_ids) == 1:
                break

    # If still ambiguous, pick the lowest-ordering ID to be deterministic.
    return min(top_ids)


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
