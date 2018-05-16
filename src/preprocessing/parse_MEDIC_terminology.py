#!/usr/bin/env python3
# coding: utf8

# Author: Lenz Furrer, 2018


'''
Parse the MEDIC terminology used for the NCBI disease corpus.
'''


import csv

from .terminology import DictEntry


def parse_MEDIC_terminology(source):
    '''
    Parse the MEDIC TSV into an iterator of namedtuples.
    '''
    if hasattr(source, 'read'):
        yield from _parse_MEDIC_terminology(source)
    else:
        with open(source, encoding='utf-8') as f:
            yield from _parse_MEDIC_terminology(f)


def _parse_MEDIC_terminology(file):
    '''
    Input fields:
        DiseaseName
        DiseaseID
        AltDiseaseIDs
        Definition
        ParentIDs
        TreeNumbers
        ParentTreeNumbers
        Synonyms

    Output fields:
        name (str):  preferred name
        id (str):    concept ID
        alt (tuple): alternative IDs (if any)
        def_ (str):  definition (if any)
        syn (tuple): synonyms (if any)
    '''
    reader = csv.reader(file, delimiter='\t', quotechar=None)
    for row in reader:
        if row[0].startswith('#'):
            continue
        name, id_, alt, def_, _, _, _, syn = row
        yield DictEntry(
            name,
            id_,
            tuple(alt.split('|')) if alt else (),
            def_,
            tuple(syn.split('|')) if syn else (),
        )
