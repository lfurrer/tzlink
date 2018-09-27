#!/usr/bin/env python3
# coding: utf8

# Author: Lenz Furrer, 2018


'''
Import terminology from SNOMED and other sources through UMLS Metathesaurus.
'''


import csv

from ..terminology import DictEntry
from ...util.util import smart_open
from ...util.umls import read_RRF, iterconcepts, co_rows, C, D, S


# IDs of the targeted semantic types.
SEMTYPES = frozenset((
    'T019',  # Congenital Abnormality
    'T020',  # Acquired Abnormality
    'T037',  # Injury or Poisoning
    'T046',  # Pathologic Function
    'T047',  # Disease or Syndrome
    'T048',  # Mental or Behavioral Dysfunction
    'T049',  # Cell or Molecular Dysfunction
    'T050',  # Experimental Model of Disease
    'T184',  # Sign or Symptom
    'T190',  # Anatomical Abnormality
    'T191',  # Neoplastic Process
))


def preprocess_SNOMED_terminology(metadir, target):
    '''
    Extract names, IDs and definitions from UMLS RRF files.
    '''
    with smart_open(target, 'w') as f:
        writer = csv.writer(f, delimiter='\t', quotechar=None)
        writer.writerows(_preprocess_SNOMED_terminology(metadir))


def _preprocess_SNOMED_terminology(metadir):
    conso_reader, def_reader, sty_reader = (
        read_RRF(metadir, n) for n in ('CONSO', 'DEF', 'STY'))
    concepts = iterconcepts(conso_reader)
    definitions = co_rows(def_reader)
    semtypes = co_rows(sty_reader)

    for conc in concepts:
        cui = conc[0][C.CUI]
        sty = (r[S.TUI] for r in semtypes.send(cui))
        if not SEMTYPES.intersection(sty):  # not a disorder
            continue
        defs = definitions.send(cui)
        if not any(r[C.SAB] == 'SNOMEDCT' for r in conc):
            cui = 'CUI-less'
        for def_, synonyms in _group_by_definition(conc, defs):
            def_ = def_.replace('\t', ' ')
            yield (cui, def_, *synonyms)


def _group_by_definition(conc, defs):
    '''
    Heuristically group synonyms by definitions.

    The DictEntry allows only one definition per entry, but
    UMLS can give one for each atom, not concept. Therefore
    create a separate entry for each definition, trying to
    distribute the synonyms in a sensible way.

    This function also determines which synonyms end up as
    "preferred names". This is encoded differently for each
    original resource and thus hard to achieve. Also, this
    information isn't used anywhere, so we just sort the
    names alphabetically and pick the first, to be
    deterministic.
    '''
    atoms = {r[C.AUI]: r[C.STR] for r in conc}
    names = set(atoms.values())
    # Pair up definitions with their corresponding atoms' ID.
    defs = [(r[D.DEF], r[D.AUI]) for r in defs if r[D.AUI] in atoms]
    # Sort the definitions by length (longest first).
    defs.sort(key=lambda p: -len(p[0]))
    while len(defs) > 1:
        def_, aui = defs.pop()  # pick the shortest remaining
        name = atoms[aui]
        names.discard(name)
        yield def_, [name]
    # Yield the longest with all left-over names.
    if defs:
        longest_def, aui = defs[0]
        names.add(atoms[aui])  # make sure the linked name is present
    else:
        longest_def = ''  # there was no definition for this concept
    left_over_names = sorted(names)
    yield longest_def, left_over_names


def parse_SNOMED_terminology(source):
    '''
    Parse the preprocessed TSV into an iterator of namedtuples.
    '''
    if hasattr(source, 'read'):
        yield from _parse_preprocessed(source)
    else:
        with open(source, encoding='utf-8') as f:
            yield from _parse_preprocessed(f)


def _parse_preprocessed(file):
    '''
    Input fields (variable, 3 to n):
        CUI
        Definition
        PreferredName
        Synonym1
        Synonym2
        ...

    Output fields:
        name (str):  preferred name
        id (str):    concept ID
        alt (tuple): alternative IDs (always empty)
        def_ (str):  definition (if any)
        syn (tuple): synonyms (if any)
    '''
    reader = csv.reader(file, delimiter='\t', quotechar=None)
    for row in reader:
        id_, def_, name, *synonyms = row
        yield DictEntry(
            name,
            id_,
            (),
            def_,
            tuple(synonyms),
        )
