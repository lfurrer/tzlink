#!/usr/bin/env python3
# coding: utf-8


'''
Convert the NCBI-disease corpus to our document-interchange format.
'''


import itertools as it


def parse_NCBI_disease_corpus(filename, terminology):
    '''
    Parse one file of the corpus.

    @Args:
        filenames: "NCBItrainset_corpus.txt", "NCBIdevelopset_corpus.txt",
                   or "NCBItestset_corpus.txt"
    @Returns:
        iter(dict(...)): iterator over documents (nested dicts/lists)
    '''
    with open(filename, "r", encoding='ascii') as file:
        for doc in _split_documents(file):
            yield _parse_document(doc, terminology)


def _split_documents(file):
    entry = []
    for line in file:
        line = line.rstrip('\n')
        if line:
            entry.append(line)
        elif entry:
            yield entry
            entry.clear()
    # Don't miss the last instance!
    if entry:
        yield entry


def _parse_document(lines, terminology):
    docid, _, title = lines[0].split('|')
    _, _, abstract = lines[1].split('|')
    abstract_offset = len(title)+1
    title_mentions = []
    abstract_mentions = []
    for mention in lines[2:]: #the mentions are documented from the third line
        cache_mention = mention.split('\t')
        cache_dict = {
            'start': int(cache_mention[1]),
            'end': int(cache_mention[2]),
            'text': cache_mention[3],
            'type': cache_mention[4],
            'id': _ref_id(cache_mention[5], terminology)}
        if cache_dict['start'] < abstract_offset:
            title_mentions.append(cache_dict)
            text = title
        else:
            cache_dict['start'] -= abstract_offset
            cache_dict['end'] -= abstract_offset
            abstract_mentions.append(cache_dict)
            text = abstract
        # Sanity check (quotes inside mentions were removed in the table).
        cache_dict['text'] = text[cache_dict['start']:cache_dict['end']]
        assert cache_dict['text'].replace('"', ' ') == cache_mention[3]
    sections = [
        {
            'text': title,
            'offset': 0,
            'mentions': title_mentions
        },
        {
            'text': abstract,
            'offset': abstract_offset,
            'mentions': abstract_mentions
        }
    ]
    doc = {'docid': docid, 'sections': sections}
    return doc


_ref_id_cache = {}

def _ref_id(ref, terminology):
    '''
    Factory function for avoiding duplicate instantiation.

    This allows detecting repeated mentions in the corpus.
    '''
    try:
        return _ref_id_cache[ref, terminology]
    except KeyError:
        existing = _ref_id_cache[ref, terminology] = RefID(ref, terminology)
        return existing


class RefID:
    '''
    Rich representation of a reference ID.

    The annotation guidelines are briefly explained in
    the corpus paper (DOI 10.1016/j.jbi.2013.12.006),
    Section 2.1.2.

    The concepts of composite mentions separated by "|",
    eg. "colorectal, endometrial, and ovarian cancers" is
    annotated with "D010051|D016889|D015179".

    In rare cases, a mention maps to multiple concepts,
    which are separated by "+" in the annotation.

    We assume that "+" would have a stronger binding than
    "|" if both were present, but there is no such case in
    the corpus.

    The IDs used are not always the preferred ID according
    to MEDIC; sometimes it even maps to multiple concepts.
    This class maps all alternative IDs to their preferred
    (canonical) MEDIC ID and joins them with "/" if there
    is more than one.

    Also, the "MESH:" prefix is missing most of the time.

    Correct predictions are required to produce all IDs of
    composite and multiple-concept mentions. However, the
    order is not enforced. Also, all reference IDs are
    mapped to preferred IDs before lookup.

    The __contains__ method of this class takes all this
    into account when comparing a prediction to the
    reference.
    '''
    def __init__(self, reference, terminology):
        self._ids = self._parse_canonical(reference, terminology)
        self._shape = self._get_shape(self._ids)
        self._str = '|'.join('+'.join('/'.join(alt) for alt in comp)
                             for comp in self._ids)
        self._alternatives = frozenset(
            '|'.join('+'.join(comp) for comp in alt)
            for alt in it.product(*(it.product(*comp)
                                    for comp in self._ids)))

    def __str__(self):
        return self._str

    def __iter__(self):
        return iter(self._alternatives)

    def _parse_canonical(self, reference, terminology):
        return tuple(tuple(terminology.canonical_ids(id_)
                           for id_ in comp)
                     for comp in self._parse(reference))

    @staticmethod
    def _parse(reference):
        return tuple(
            tuple(
                id_ if id_.startswith(('MESH', 'OMIM')) else 'MESH:'+id_
                for id_ in comp.split('+')
            ) for comp in reference.strip().split('|')
        )

    @staticmethod
    def _get_shape(ids):
        return sorted(len(comp) for comp in ids)

    def __contains__(self, other):
        '''
        Compare to a predicted ID (str).
        '''
        # Quickly check a common case.
        if other == str(self) or other in self._alternatives:
            return True

        # Take a closer look: same number of components?
        other = self._parse(other)
        if self._get_shape(other) != self._shape:
            return False

        # Pair up all components across both levels.
        # Since order is free, all combinations are checked with brute force.
        # This looks bad, but most of the time all sequences are singletons.
        for perm in _nested_permutations(other):
            try:
                if all(pred in refs
                       for refs, pred in _nested_zip(self._ids, perm)):
                    return True
            except LengthMismatch:
                continue

        return False


def _nested_zip(a, b):
    '''
    Flat iteratrion over two equally nested structures.
    '''
    for aa, bb in zip(a, b):
        if len(aa) != len(bb):
            raise LengthMismatch
        yield from zip(aa, bb)


def _nested_permutations(seq):
    '''
    Example:

    _nested_permutations(['abc', 'xy']) ->
        (('a', 'b', 'c'), ('x', 'y'))
        (('a', 'b', 'c'), ('y', 'x'))
        (('a', 'c', 'b'), ('x', 'y'))
        ...
        (('c', 'b', 'a'), ('y', 'x'))
        (('x', 'y'), ('a', 'b', 'c'))
        (('x', 'y'), ('a', 'c', 'b'))
        ...
        (('y', 'x'), ('c', 'b', 'a'))
    '''
    for perm in it.permutations(seq):
        yield from it.product(*(it.permutations(item) for item in perm))


class LengthMismatch(Exception):
    '''Sequences of unequal lengths cannot be paired.'''
