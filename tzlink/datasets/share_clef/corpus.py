#!/usr/bin/env python3
# coding: utf-8


'''
Convert the ShARe/CLEF corpus to our document-interchange format.
'''


import os

from . import subsets


def parse_ShARe_CLEF_corpus(dir_, subset, terminology=None):
    '''
    Parse a subset of the corpus.

    @Args:
        dir_: path to a directory containing the corpus
              The following directory structure is required:
              dir_/
                  train/
                      reports/
                          report-1.txt
                          report-2.txt
                          ...
                      annotations/
                          report-1.pipe.txt
                          report-2.pipe.txt
                          ...
                  test/
                      [same substructure]
        subset: "test", "dev", "dev1" ...
        terminology: ignored

    @Returns:
        iter(dict(...)): iterator over documents (nested dicts/lists)
    '''
    del terminology  # unused compatibility argument

    if subset == 'all':
        for ss in ('train', 'dev', 'test'):
            yield from _parse_ShARe_CLEF_corpus(dir_, ss)
    else:
        yield from _parse_ShARe_CLEF_corpus(dir_, subset)


def _parse_ShARe_CLEF_corpus(dir_, subset):
    subdir, ids = subsets.docs(subset)
    rep_tmpl = os.path.join(dir_, subdir, 'reports', '{}.txt')
    ann_tmpl = os.path.join(dir_, subdir, 'annotations', '{}.pipe.txt')

    for docid in ids:
        with open(rep_tmpl.format(docid), "r", encoding='ascii') as f:
            report = f.read()
        with open(ann_tmpl.format(docid), "r", encoding='ascii') as f:
            yield _join_standoff(f, report, docid)


def _join_standoff(anno, report, docid):
    meta, report = report.split('\n', 1)
    assert docid == _get_docid(meta), 'metadata/filename mismatch'
    mentions = list(_parse_anno(anno, report, len(meta)+1))
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

_docid_template = '{:05d}-{:06d}-{}'


def pairs(sequence):
    '''
    Iterate over non-overlapping bigrams.
    '''
    return zip(sequence[::2], sequence[1::2])
