#!/usr/bin/env python3
# coding: utf8

# Author: Lenz Furrer, 2018


'''
Parse the MEDIC terminology used for the NCBI disease corpus.
'''


import csv
from collections import OrderedDict

from ..terminology import DictEntry
from ...util.util import smart_open
from ...util.umls import read_RRF, iterconcepts, co_rows, C, D


# Column IDs of the MEDIC TSV.
IDM = 1  # main ID
ALT = 2
DEF = 3
SYN = 7


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
    for name, id_, alt, def_, _, _, _, syn in _parse_MEDIC_format(file):
        yield DictEntry(name, id_, alt, def_, syn)


def _parse_MEDIC_format(file, skip_comments=True):
    reader = csv.reader(file, delimiter='\t', quotechar=None)
    for row in reader:
        if row[0].startswith('#'):
            if skip_comments:
                continue
        else:
            _unpack_list_value_fields(row)
        yield row


def _unpack_list_value_fields(row):
    '''Break up the list-value fields "alt" and "syn".'''
    for i in (ALT, SYN):
        row[i] = tuple(row[i].split('|') if row[i] else ())


def _pack_list_value_fields(row):
    for i in (ALT, SYN):
        row[i] = '|'.join(row[i])


def extend_MEDIC_with_UMLS(medic, metadir, target, **flags):
    '''
    Create an extended version of the MEDIC terminology TSV.
    '''
    with smart_open(medic) as r, smart_open(target, 'w') as w:
        _extend_MEDIC_with_UMLS(r, metadir, w, **flags)


def _extend_MEDIC_with_UMLS(medic, metadir, target, **flags):
    lines = list(_parse_MEDIC_format(medic, skip_comments=False))
    writer = csv.writer(target, delimiter='\t', quotechar=None)

    # Get rid of the header lines right away.
    while lines[0][0].startswith('#'):
        writer.writerow(lines.pop(0))
    # Apply all modifications in-place.
    _apply_UMLS_extension(lines, metadir, **flags)
    # Write the modified terminology to disk.
    for row in lines:
        try:
            _pack_list_value_fields(row)
        except IndexError:  # mid-file comments
            pass  # no packing needed
        writer.writerow(row)


def _apply_UMLS_extension(lines, metadir,
                          divide=True, names=True, definitions=True):
    if names:
        _add_UMLS_names(lines, metadir)
    if divide:
        _divide_polysemous_concepts(lines, metadir)
    if definitions:
        _add_UMLS_definitions(lines, metadir)


class _MedicBuffer:
    '''
    Abstract base class for UMLS-extension filters.

    Subclasses are used for modifying MEDIC lines in-place
    while applying UMLS-based extensions.

    The filters are implemented as classes for reasons of
    inheritance and shared state, but are to be used like
    functions with side effect.
    '''
    def __init__(self, lines, metadir):
        self._lines = lines
        self._concepts = iterconcepts(read_RRF(metadir, 'CONSO'))

        self._setup()
        self._run()
        self._flush()

    def _setup(self):
        raise NotImplementedError

    def _run(self):
        for conc in self._concepts:
            ids = self._medic_ids(conc)
            self._use(ids, conc)

    def _use(self, ids, conc):
        raise NotImplementedError

    def _flush(self):
        '''
        Propagate the changes to the original lines.
        '''
        raise NotImplementedError

    def _iterlines(self):
        '''
        Iterate over lines, skipping mid-file comments.
        '''
        for line in self._lines:
            if not line[0].startswith('#'):
                yield line

    @classmethod
    def _medic_ids(cls, conc):
        for row in conc:
            try:
                prefix = cls._prefixes[row[C.SAB]]
            except KeyError:
                pass
            else:
                yield prefix + row[C.CODE]

    _prefixes = {'MSH': 'MESH:', 'OMIM': 'OMIM:'}


class _divide_polysemous_concepts(_MedicBuffer):
    '''
    Split up polysemous concepts into sub-concepts.
    '''
    def __init__(self, lines, metadir):
        self._names = {}
        self._targets = {}
        super().__init__(lines, metadir)

    def _setup(self):
        '''
        Find polysemous entries.
        '''
        for line in self._iterlines():
            name, id_, alt, *_, syn = line
            if len(alt) > 1:
                self._index(id_, alt, name, *syn)

    def _index(self, main, alt, *names):
        self._names[main] = OrderedDict((n.lower(), n) for n in names)
        for id_ in alt:
            self._names[id_] = {}
            self._targets[id_] = main

    def _use(self, ids, conc):
        names = [r[C.STR].lower() for r in conc]
        for id_ in ids:
            self._separate(id_, names)

    def _separate(self, id_, synonyms):
        '''
        Use the synonyms for this ID to separate it from its siblings.
        '''
        try:
            main = self._targets[id_]
        except KeyError:
            return

        for name_lc in synonyms:
            try:
                name = self._names[main].pop(name_lc)
            except KeyError:
                continue
            self._names[id_][name_lc] = name

    def _flush(self):
        copy = []
        for line in self._lines:
            if line[0].startswith('#'):  # also copy mid-file comments
                copy.append(line)
                continue

            name, id_, alt, def_, *hierarchy, syn = line
            if len(alt) <= 1:
                copy.append(line)
                continue

            for i in (id_, *alt):
                try:
                    name, *syn = self._names[i].values()
                except ValueError:
                    # The division wasn't perfect. Fall back to using
                    # the superordinate preferred name.
                    name, syn = line[0], ()
                syn = tuple(syn)  # don't change type
                copy.append([name, i, (), def_, *hierarchy, syn])
                def_ = ''  # use definition for main ID only
        self._lines[:] = copy


class _add_UMLS_names(_MedicBuffer):
    '''
    Add synonyms to the MEDIC vocabulary.
    '''
    def __init__(self, lines, metadir):
        self._index = {}
        self._names = []
        super().__init__(lines, metadir)

    def _setup(self):
        for line in self._iterlines():
            self._add_line(line)

    def _add_line(self, line):
        try:
            name, id_, alt, _, _, _, _, syn = line
        except ValueError:
            names = {}
        else:
            names = {n.lower(): None for n in (name, *syn)}
            for i in (id_, *alt):
                self._index.setdefault(i, []).append(names)
        self._names.append(names)

    def _use(self, ids, conc):
        names = [r[C.STR] for r in conc]
        for id_ in ids:
            self._add_synonyms(id_, names)

    def _add_synonyms(self, id_, synonyms):
        '''
        Add new synonyms to all rows that match this ID.
        '''
        for names in self._index.get(id_, ()):
            for n in synonyms:
                names.setdefault(n.lower(), n)

    def _flush(self):
        for line, names in zip(self._iterlines(), self._names):
            newnames = tuple(n for n in names.values() if n is not None)
            if newnames:
                line[SYN] += newnames


class _add_UMLS_definitions(_MedicBuffer):
    '''
    Complement missing definitions from UMLS.
    '''
    def __init__(self, lines, metadir):
        self._definitions = co_rows(read_RRF(metadir, 'DEF'))
        self._targets = {}
        super().__init__(lines, metadir)

    def _setup(self):
        for line in self._iterlines():
            if not line[DEF]:
                for i in (line[IDM], *line[ALT]):
                    self._targets.setdefault(i, [])

    def _use(self, ids, conc):
        targets = [self._targets[i] for i in ids if i in self._targets]
        if targets:
            cui = conc[0][C.CUI]
            defs = [r[D.DEF] for r in self._definitions.send(cui)]
            self._add_definitions(targets, defs)

    @staticmethod
    def _add_definitions(targets, defs):
        targets.sort(key=len)  # serve first the entries with fewer definitions
        defs.sort(key=len, reverse=True)  # prefer longer definitions
        for tgt, def_ in zip(targets, defs):
            tgt.append(def_)

    def _flush(self):
        for line in self._iterlines():
            for i in (line[IDM], *line[ALT]):
                newdefs = self._targets.get(i)
                if newdefs:
                    line[DEF] = max(newdefs, key=len)
