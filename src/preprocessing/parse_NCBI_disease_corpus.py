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
    title_line = lines[0].split('|')
    docid = title_line[0] #id is in the first position
    title = max(title_line, key=len)
    abstract = max(lines[1].split('|'), key=len)
    abstract_offset = len(title)+1
    title_mentions = []
    abstract_mentions = []
    for mention in lines[2:]: #the mentions are documented from the third line
        cache_mention = mention.split('\t')
        # Some lists contain empty elements because of the ugly format :P
        cache_mention[:] = [item for item in cache_mention if item]
        cache_dict = {
            'start': int(cache_mention[1]),
            'end': int(cache_mention[2]),
            'text': cache_mention[3],
            'type': cache_mention[4],
            'id': cache_mention[5]}
        if cache_dict['start'] < abstract_offset:
            title_mentions.append(cache_dict)
        else:
            cache_dict['start'] -= abstract_offset
            cache_dict['end'] -= abstract_offset
            abstract_mentions.append(cache_dict)
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
