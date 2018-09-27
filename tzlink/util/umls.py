#!/usr/bin/env python3
# coding: utf8

# Author: Lenz Furrer, 2018


'''
Utilities for reading UMLS Metathesaurus data.
'''


import os
import csv
from enum import IntEnum

from .util import co_init


# Column names of the UMLS MRCONSO, MRDEF and MRSTY tables.
# https://www.ncbi.nlm.nih.gov/books/NBK9685/
C = IntEnum('MRCONSO', 'CUI LAT TS LUI STT SUI ISPREF AUI SAUI SCUI SDUI '
                       'SAB TTY CODE STR SRL SUPPRESS CVF', start=0)
D = IntEnum('MRDEF', 'CUI AUI ATUI SATUI SAB DEF SUPPRESS CVF', start=0)
S = IntEnum('MRSTY', 'CUI TUI STN STY ATUI CVF', start=0)


def read_RRF(metadir, name):
    '''
    Find an RRF file in metadir and CSV-parse it.
    '''
    fn = os.path.join(metadir, 'MR{}.RRF'.format(name))
    with open(fn, encoding='utf-8') as f:
        yield from csv.reader(f, delimiter='|', quotechar=None)


def nonenglish_suppressflag(row):
    '''Exclude non-English and suppress-flagged rows.'''
    if row[C.LAT] != 'ENG':
        return True
    if row[C.SUPPRESS] != 'N':
        return True
    return False


def iterconcepts(reader, skip_row=nonenglish_suppressflag):
    '''
    Iterate over contiguous rows of the same concept.
    '''
    cui = None
    accu = []
    for row in reader:
        if skip_row(row):
            continue
        if row[C.CUI] != cui:
            if accu:
                yield accu
            cui = row[C.CUI]
            accu = []
        accu.append(row)
    if accu:
        yield accu


@co_init
def co_rows(reader):
    '''
    Coroutine iterating over selected rows.

    For each CUI provided through .send(), yield a list of
    matching rows (which may be empty).

    Every CUI should always be greater (alphabetically
    later) than any previously provided CUI, otherwise
    nothing will be found.

    This generator never raises StopIteration. When the
    underlying iterator is exhausted, the generator will
    continue to yield empty lists for each sent item.
    '''
    target = (yield None)
    accu = []
    for row in reader:
        current = row[0]  # the CUI is in the first field in all tables
        while current > target:
            # Wait for target to catch up.
            target = (yield accu)
            accu = []
        if current == target:
            # Accumulate rows as long as their CUI matches the target CUI.
            accu.append(row)
    # The reader is exhausted. Yield the last batch of accumulated rows,
    # then continue yielding empty lists.
    while True:
        yield accu
        accu = []
