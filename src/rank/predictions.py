#!/usr/bin/env python3
# coding: utf8

# Author: Lenz Furrer, 2018


'''
Dump and evaluate predictions.
'''


import os
import csv


NIL = 'NIL'


def handle_predictions(conf, dump, evaluate, data):
    '''
    Write predictions to a TSV file and/or print evaluation figures.
    '''
    dumper = TSVWriter(conf, dump)
    evaluator = Evaluator(evaluate)
    for entry in _itermentions(conf, data):
        dumper.write(entry)
        evaluator.update(entry)
    dumper.close()
    evaluator.summary()


def _itermentions(conf, data):
    for *annotation, refs, start, end in data.occs:
        if start != end:
            i = max(range(start, end), key=data.scores.__getitem__)
            score = data.scores[i]
            ids = data.ids[i]
        else:
            score = None
            ids = []
        id_ = _disambiguate(conf, ids, score)
        reachable = len(refs) == 1 and any(data.ids[i].intersection(refs[0])
                                           for i in range(start, end))
        yield (*annotation, refs, id_, len(ids), reachable)


def _disambiguate(conf, ids, score):
    if not ids or score < conf.rank.min_score:
        return NIL
    return min(ids)  # just pick one -- use min() to be deterministic


class TSVWriter:
    '''
    Write a TSV line for each mention.

    The fields are:
        doc ID
        start (document character offset)
        end (ditto)
        mention text
        reference ID(s)
        predicted ID
        number of IDs for the top-ranked candidate name
        reachable (reference ID is among the candidates)
    '''
    def __init__(self, conf, enabled):
        if enabled:
            self.write = self._write
            self.close = self._close
            self._file = self._open(conf.io.prediction_fn)
            self._writer = csv.writer(self._file, quotechar=None,
                                      delimiter='\t', lineterminator='\n')
        else:
            # Dummy mode.
            self.write = lambda _: None
            self.close = lambda: None

    def _write(self, entry):
        # Format the ref IDs.
        fields = list(entry)
        fields[4] = '+'.join('|'.join(comp) for comp in fields[4])
        self._writer.writerow(fields)

    @staticmethod
    def _open(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        file = open(path, 'w', encoding='utf8')
        return file

    def _close(self):
        self._file.close()


class Evaluator:
    def __init__(self, enabled):
        self.correct = 0
        self.total = 0
        self.unreachable = 0
        self.ambiguous = 0
        self.compound = 0
        self.nocandidates = 0

        if enabled:
            self.update = self._update
            self.summary = self._summary
        else:
            # Dummy mode.
            self.update = lambda _: None
            self.summary = lambda: None

    @property
    def accuracy(self):
        '''Proportion of mentions with correct top-ranked ID.'''
        return self.correct/self.total

    def _update(self, entry):
        *_, refs, id_, n_ids, reachable = entry
        self.total += 1
        self.unreachable += not reachable
        if len(refs) > 1:
            # No chance to get these right.
            self.compound += 1
        elif id_ in refs[0]:
            self.correct += 1
        if n_ids == 0:
            self.nocandidates += 1
        elif n_ids > 1:
            self.ambiguous += 1

    def _summary(self):
        labels = '''accuracy correct total
                    unreachable nocandidates ambiguous compound'''.split()
        for label in labels:
            print('{:12} {:5}'.format(label, getattr(self, label)))
