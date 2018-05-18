#!/usr/bin/env python3
# coding: utf-8


'''
Convert the NCBI-disease corpus to our document-interchange format.
'''


def parse_NCBI_disease_corpus(filename):
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
            yield _parse_document(doc)


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


def _parse_document(lines):
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
            'id': _parse_ids(cache_mention[5])}
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


def _parse_ids(ids):
    '''
    Divide into alternative and component IDs.

    Alternatives are separated by "|".
    The components of compound concepts are separated by "+".

    Also, the "MESH:" prefix is missing most of the time.

    Return a tuple of components, each of which is a
    frozenset of alternatives.
    '''
    return tuple(
        frozenset(
            alt if alt.startswith(('MESH', 'OMIM')) else 'MESH:'+alt
            for alt in component.split('|')
        ) for component in ids.strip().split('+')
    )
