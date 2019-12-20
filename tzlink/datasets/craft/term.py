#!/usr/bin/env python3
# coding: utf8

# Author: Lenz Furrer, 2019


'''
Import terminology from a CRAFT OBO ontology.
'''


import io
import re
import csv
import logging
import zipfile
from pathlib import Path

from ...util.util import smart_open
from ..terminology import DictEntry


ONTOLOGIES = {
    'CHEBI': 'chemical',
    'CL': 'cell',
    'GO_BP': 'biological_process',
    'GO_CC': 'cellular_component',
    'GO_MF': 'molecular_function',
    'MOP': 'molecular_process',
    'NCBITaxon': 'organism',
    'PR': 'gene/protein',
    'SO': 'sequence',
    'UBERON': 'organ/tissue',
}
ONTOLOGIES.update({f'{o}_EXT': e for o, e in ONTOLOGIES.items()})
SYNONYM_TYPES = ['EXACT', 'BROAD', 'NARROW', 'RELATED']


def preprocess_CRAFT_terminology(craftdir, ontology, target, **kwargs):
    """
    Extract names, IDs and definitions from an OBO file.
    """
    with smart_open(target, 'w') as f:
        writer = csv.writer(f, delimiter='\t', quotechar=None)
        writer.writerows(
            _preprocess_CRAFT_terminology(craftdir, ontology, **kwargs))


def _preprocess_CRAFT_terminology(craftdir, onto, **kwargs):
    etype = ONTOLOGIES[onto]
    idprefix = onto.split('_')[0]
    obo_zip = _get_obo_zip(Path(craftdir), onto)
    stream = _read_obo_zip(obo_zip)
    id_map = dict(_get_id_mapping(obo_zip.parent))
    for concept in _iter_concepts(stream, idprefix, etype, id_map, **kwargs):
        concept['syn'].discard(concept['name'])
        yield (concept['id'],
               concept.get('def', ''),
               concept['name'], *concept['syn'])


def _get_obo_zip(craftdir, onto):
    if onto.endswith('_EXT'):
        base = onto[:-4]
        sub = f'{base}+extensions'
    else:
        base = sub = onto
    (path,) = (craftdir/'concept-annotation'/base/sub).glob('*.obo.zip')
    return path


def _read_obo_zip(path):
    with zipfile.ZipFile(str(path)) as z:
        (obo,) = z.filelist
        with io.TextIOWrapper(z.open(obo), encoding='utf8') as f:
            yield from f


def _get_id_mapping(ontodir):
    try:
        (path,) = ontodir.glob('unused*.txt')
    except ValueError as e:
        if 'not enough' in str(e):
            return  # glob didn't match
        raise ValueError(f'ambiguous glob: {ontodir}/unused*.txt')
    with path.open() as f:
        for line in f:
            k, *v = line.strip().split('\t')
            yield k, v


def _iter_concepts(stream, idprefix, etype, id_mapping, **kwargs):
    for concept in _iter_stanzas(stream, **kwargs):
        if concept.get('obsolete'):
            continue
        id_ = concept['id']
        if not (id_.startswith(idprefix) or id_.split(':')[0].endswith('_EXT')):
            continue
        if idprefix == 'GO' and concept.get('entity_type') not in (None, etype):
            continue
        for mapped in id_mapping.get(id_, (id_,)):
            if mapped != id_:
                yield dict(concept, id=mapped)
            else:
                yield concept


def _iter_stanzas(stream, synonym_types=('EXACT',)):
    tag_value = re.compile(r'(\w+): (.+)')
    synonym_type = re.compile(r'"((?:[^"]|\\")*)" ([A-Z]+)')
    definition = re.compile(r'"((?:[^"]|\\")*)"')

    inside = False
    concept = {}
    for line in _fix_broken_defs(stream):
        line = line.strip()
        if not line:
            # Stanza has ended.
            if 'id' in concept:
                yield concept
            inside = False
            concept = {}
        elif line == '[Term]':
            # Stanza starts.
            inside = True
            concept['syn'] = set()
        elif inside:
            try:
                tag, value = tag_value.match(line).groups()
            except AttributeError:
                logging.warning('invalid OBO line: %r', line)
                continue
            if tag == 'id':
                concept['id'] = value
            elif tag == 'namespace':
                concept['entity_type'] = value
            elif tag == 'name':
                concept['name'] = value
            elif tag == 'synonym':
                synonym, syntype = synonym_type.match(value).groups()
                if syntype in synonym_types:
                    # Unescape quotes.
                    synonym = synonym.replace('\\"', '"')
                    concept['syn'].add(synonym)
            elif (tag, value) == ('is_obsolete', 'true'):
                concept['obsolete'] = True
            elif tag == 'def':
                try:
                    concept['def'] = definition.match(value).group(1)
                except AttributeError:
                    logging.warning('invalid definition: %s', value)
                    continue
    if 'id' in concept:
        # After the final stanza: last yield.
        yield concept


def _fix_broken_defs(stream):
    definition = re.compile(r'"((?:[^"]|\\")*)"')
    for line in stream:
        if line.startswith('def: '):
            while not definition.match(line[5:]):
                try:
                    line = line.rstrip('\n\r') + next(stream)
                except StopIteration:
                    raise ValueError('Unexpected EOF')
        yield line


def parse_CRAFT_terminology(source):
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
        ID
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
