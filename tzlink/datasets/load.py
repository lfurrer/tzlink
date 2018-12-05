#!/usr/bin/env python3
# coding: utf8

# Author: Lenz Furrer, 2018


'''
Dataset loading utilities.
'''


from .terminology import Terminology
from .ncbi_disease import parse_MEDIC_terminology, parse_NCBI_disease_corpus
from .share_clef import parse_ShARe_CLEF_corpus, parse_SNOMED_terminology


# Make the data/terminology loaders accessible through config names.
corpus_loader = {
    'ncbi-disease': parse_NCBI_disease_corpus,
    'share-clef': parse_ShARe_CLEF_corpus,
}
dict_loader = {
    'ncbi-disease': parse_MEDIC_terminology,
    'share-clef': parse_SNOMED_terminology,
}


def load_data(conf, subset, terminology=None):
    '''
    Pick and parse the right corpus file.

    The parameter subset is one of "train", "dev", "test",
    or "devN"/"trainN" with an integer indicating the fold.
    '''
    dataset = conf.general.dataset
    dir_ = conf[dataset].corpus_dir
    if terminology is None:
        terminology = load_dict(conf)
    return corpus_loader[dataset](dir_, subset, terminology)


def load_dict(conf):
    '''
    Read a dict file into a Terminology instance.
    '''
    return Terminology(_dict_entries(conf))


def _dict_entries(conf):
    dataset = conf.general.dataset
    fn = conf[dataset].dict_fn
    return dict_loader[dataset](fn)


def itermentions(corpus):
    '''
    Iterate over pairs <text, IDs>.
    '''
    for doc in corpus:
        for sec in doc['sections']:
            for mention in sec['mentions']:
                yield mention['text'], mention['id']


def itertext_corpus(conf, subset, mentions=True, context=True):
    '''
    Iterate over textual contents of the corpus.
    '''
    # The NCBID loader requires a terminology, but we ignore its actions here.
    dummy = Terminology([])
    dummy.canonical_ids = lambda id_: (id_,)
    for doc in load_data(conf, subset, dummy):
        for sec in doc['sections']:
            if context:
                yield sec['text']
            if mentions:
                for mention in sec['mentions']:
                    yield mention['text']


def itertext_terminology(conf, names=True, syn=True, defs=True):
    '''
    Iterate over textual contents of the terminology.
    '''
    for entry in _dict_entries(conf):
        if names:
            yield entry.name
        if syn:
            yield from entry.syn
        if defs and entry.def_:
            yield entry.def_
