#!/usr/bin/env python3
# coding: utf-8


'''
Convert the ShARe/CLEF corpus to our document-interchange format.
'''


import os
import itertools as it


def parse_ShARe_CLEF_corpus(dir_, terminology=None):
    '''
    Parse a portion of the corpus.

    @Args:
        dir_: path to a directory containing the corpus
              The following directory structure is required:
              dir_/
                  reports/
                      report-1.txt
                      report-2.txt
                      ...
                  annotations/
                      report-1.pipe.txt
                      report-2.pipe.txt
                      ...

    @Returns:
        iter(dict(...)): iterator over documents (nested dicts/lists)
    '''
    del terminology  # unused compatibility argument

    subdirs = [os.path.join(dir_, s) for s in ('reports', 'annotations')]
    fns = (sorted(os.listdir(s)) for s in subdirs)

    for rep, anno in it.zip_longest(*fns):
        # Take some precautions to make sure the files are zipped correctly.
        assert rep.split('.')[0] == anno.split('.')[0], 'filename mismatch'
        paths = (os.path.join(*c) for c in zip(subdirs, (rep, anno)))
        yield _parse_report(*paths)


def _parse_report(rep_fn, anno_fn):
    with open(rep_fn, "r", encoding='ascii') as f:
        rep = f.read()
    with open(anno_fn, "r", encoding='ascii') as f:
        return _join_standoff(f, rep)


def _join_standoff(anno, report):
    meta, report = report.split('\n', 1)
    docid = _get_docid(meta)
    mentions = list(_parse_anno(anno, report, len(meta)))
    body = {
        'text': report,
        'offset': 0,
        'mentions': mentions
    }
    doc = {'docid': docid, 'sections': [body]}
    return doc


def _parse_anno(lines, report, sec_start):
    for line in lines:
        _, type_, cui, *offsets = line.split('||')
        offsets = [int(o)-sec_start for o in offsets]
        spans = [report[s:e] for s, e in pairs(offsets)]
        text = ' […] '.join(spans)
        yield {
            'start': offsets[0],
            'end': offsets[-1],
            'gaps': list(pairs(offsets[1:-1])),
            'text': text,
            'type': type_,
            'id': (cui,)  # singleton (1-tuple)!
        }


def _get_docid(report):
    id2, id1, _, type_, _ = report.split('||||', 4)
    return _docid_template.format(int(id1), int(id2), type_.strip())

_docid_template = '{:05d}-{:06d}-{}.txt'


def pairs(sequence):
    '''
    Iterate over non-overlapping bigrams.
    '''
    return zip(sequence[::2], sequence[1::2])
