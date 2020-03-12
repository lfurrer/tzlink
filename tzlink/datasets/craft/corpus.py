#!/usr/bin/env python3
# coding: utf-8


'''
Convert the CRAFT corpus to our document-interchange format.
'''


import os
import io

from . import subsets


def parse_CRAFT_corpus(dir_, subset, terminology=None):
    '''
    Parse a subset of the corpus.

    @Args:
        dir_: path to a corpus subdirectory for one anno type.
              The following directory structure is required:
              dir_/          # eg. CHEBI/
                  test/
                      11319941.bionlp
                      11604102.bionlp
                      ...
                  train/
                      11532192.bionlp
                      11597317.bionlp
                      ...
                  txt/
                      11319941.txt
                      11532192.txt
                      ...
        subset: "test", "dev", "dev0" ...
        terminology: ignored

    @Returns:
        iter(dict(...)): iterator over documents (nested dicts/lists)
    '''
    del terminology  # unused compatibility argument

    if subset == 'all':
        for ss in ('train', 'dev', 'test'):
            yield from _parse_CRAFT_corpus(dir_, ss)
    else:
        yield from _parse_CRAFT_corpus(dir_, subset)


def _parse_CRAFT_corpus(dir_, subset):
    ids = subsets.docs(subset)
    subdir = 'test' if subset == 'test' else 'train'
    txt_tmpl = os.path.join(dir_, 'txt', '{}.txt')
    ann_tmpl = os.path.join(dir_, subdir, '{}.bionlp')

    for docid in ids:
        with open(txt_tmpl.format(docid), "r", encoding='utf-8') as f:
            text = f.read()
        try:
            f = open(ann_tmpl.format(docid), "r", encoding='utf-8')
        except FileNotFoundError:
            if subset == 'test':  # missing annotations allowed for the test set
                f = io.StringIO()
            else:
                raise
        with f:
            yield _join_standoff(f, text, docid)


def _join_standoff(anno, text, docid):
    mentions = list(_parse_anno(anno, text, 0))
    body = {
        'text': text,
        'offset': 0,
        'mentions': mentions
    }
    doc = {'docid': docid, 'sections': [body]}
    return doc


def _parse_anno(lines, text, sec_start):
    for line in lines:
        if not line.startswith('T'):
            continue
        _, info, mention = line.rstrip().split('\t')
        id_, offsets = info.split(' ', 1)
        offsets = [[int(o)-sec_start for o in span.split()]
                   for span in offsets.split(';')]
        extracted = ' ... '.join(text[s:e] for s, e in offsets)
        assert extracted == mention, \
            f'inconsistent span: {repr(extracted)} vs. {repr(mention)}'
        yield {
            'start': offsets[0][0],
            'end': offsets[-1][-1],
            'gaps': _get_gaps(offsets),
            'text': mention,
            'type': None,
            'id': (id_,)  # singleton (1-tuple)!
        }


def _get_gaps(offset_pairs):
    starts, ends = zip(*offset_pairs)
    return tuple(zip(ends[:-1], starts[1:]))
