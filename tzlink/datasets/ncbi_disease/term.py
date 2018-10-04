#!/usr/bin/env python3
# coding: utf8

# Author: Lenz Furrer, 2018


'''
Parse the MEDIC terminology used for the NCBI disease corpus.
'''


import csv

from ..terminology import DictEntry
from ...util.util import smart_open
from ...util.umls import read_RRF, iterconcepts, C


def parse_MEDIC_terminology(source):
    '''
    Parse the MEDIC TSV into an iterator of namedtuples.
    '''
    if hasattr(source, 'read'):
        yield from _parse_MEDIC_terminology(source)
    else:
        with open(source, encoding='utf-8') as f:
            yield from _parse_MEDIC_terminology(f)


def _parse_MEDIC_format(file):
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
        yield name, id_, alt, def_, syn


def _parse_MEDIC_terminology(file):
    for name, id_, alt, def_, syn in _parse_MEDIC_format(file):
        yield DictEntry(
            name,
            id_,
            tuple(alt.split('|')) if alt else (),
            def_,
            tuple(syn.split('|')) if syn else (),
        )


def extend_MEDIC_with_UMLS(medic, metadir, target):
    '''
    Create an extended version of the MEDIC terminology TSV.
    '''
    with smart_open(medic) as r, smart_open(target, 'w') as w:
        _extend_MEDIC_with_UMLS(r, metadir, w)


def _extend_MEDIC_with_UMLS(medic, metadir, target):
    concepts = iterconcepts(read_RRF(metadir, 'CONSO'))
    with MedicBuffer(medic, target) as mbuf:
        for conc in concepts:
            ids = list(_medic_ids(conc))
            if not ids:
                continue
            names = (r[C.STR] for r in conc)
            mbuf.update(ids, names)


def _medic_ids(conc):
    for row in conc:
        try:
            prefix = _prefixes[row[C.SAB]]
        except KeyError:
            pass
        else:
            yield prefix + row[C.CODE]

_prefixes = {'MSH': 'MESH:', 'OMIM': 'OMIM:'}


class MedicBuffer:
    '''
    Keep the MEDIC vocabulary in memory while adding synonyms.
    '''
    def __init__(self, infile, outfile):
        self._index = {}
        self._lines = []
        self._outfile = outfile
        self._read(infile)

    def _read(self, infile):
        _D = {}  # reused empty dummy dict
        for line in infile:
            line = line.rstrip('\n\r')
            if line.startswith('#'):
                self._lines.append((line, _D))
            else:
                self._parse(line)

    def _parse(self, line):
        name, id_, alt, _, _, _, _, syn = line.split('\t')
        syn, alt = ((s.split('|') if s else ()) for s in (syn, alt))
        names = {n.lower(): None for n in (name, *syn)}
        self._lines.append((line, names))
        for i in (id_, *alt):
            self._index.setdefault(i, []).append(names)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def update(self, ids, newnames):
        '''
        Add new synonyms to all rows that match the IDs.
        '''
        for i in ids:
            for names in self._index.get(i, ()):
                for n in newnames:
                    names.setdefault(n.lower(), n)

    def close(self):
        '''
        Write the modified file to disk.
        '''
        for line, names in self._lines:
            newnames = [n for n in names.values() if n is not None]
            self._outfile.write(line)
            if newnames:
                self._outfile.write('|')
                self._outfile.write('|'.join(newnames))
            self._outfile.write('\n')
