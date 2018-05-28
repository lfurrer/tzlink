#!/usr/bin/env python3
# coding: utf8

# Author: Lenz Furrer, 2018


'''
Common loading utilities.
'''


from .terminology import Terminology
from .parse_MEDIC_terminology import parse_MEDIC_terminology
from .parse_NCBI_disease_corpus import parse_NCBI_disease_corpus


# Make the data/terminology loaders accessible through config names.
corpus_loader = {
    'ncbi-disease': parse_NCBI_disease_corpus,
}
dict_loader = {
    'ncbi-disease': parse_MEDIC_terminology,
}

def load_data(conf, dataset, subset):
    '''
    Pick and parse the right corpus or dict file.

    The parameter subset is one of "train", "dev", "test",
    and "dict".
    '''
    loader = dict_loader if subset == 'dict' else corpus_loader
    fn = conf[dataset]['{}_fn'.format(subset)]
    return loader[dataset](fn)


def load_dict(conf, dataset):
    '''
    Read a dict file into a Terminology instance.
    '''
    return Terminology(load_data(conf, dataset, 'dict'))


def itermentions(corpus):
    '''
    Iterate over pairs <text, IDs>.
    '''
    for doc in corpus:
        for sec in doc['sections']:
            for mention in sec['mentions']:
                yield mention['text'], mention['id']
