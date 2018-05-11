#!/usr/bin/env python3
# coding: utf-8


def parse_NCBI_disease_corpus(filename):
    '''
    Parse the NCBI-disease corpus format.

    @Args:
        filenames: "NCBItrainset_corpus.txt", "NCBIdevelopset_corpus.txt",
                   or "NCBItestset_corpus.txt"
    @Returns:
        a list of instances in our document-interchange format
    '''
    with open(filename, "r") as file:
        file = file.readlines() #read all the things into a list
        cache_entry = ''
        entry_list = [] #organize the list so that each instance is grouped into an element
        countdown = len(file) #count down for the last line of file to avoid missing the last instance
        for line in file:
            countdown += -1
            if line == '\n'and cache_entry == '': #pass very first line
                pass
            elif line == '\n':
                entry_list.append(cache_entry)
                cache_entry = ''
            else:
                cache_entry = cache_entry+line
                if countdown == 1:
                    entry_list.append(cache_entry)

    #sanity check, should be 593 instances for training corpus
    #print('Number of instances:',len(entry_list))

    #now parse each instance into a list of the interchange format
    parsed_corpus_list = []
    for instance in entry_list:
        cache_entry = instance.split('\n') #break the instance into lines by \n
        cache_entry[:] = [item for item in cache_entry if item != '']
        title_line = cache_entry[0].split('|')
        docid = title_line[0] #id is in the first position
        title = max(title_line, key=len)
        abstract = max(cache_entry[1].split('|'), key=len)
        abstract_offset = len(title)+1
        title_mentions = []
        abstract_mentions = []
        for mention in cache_entry[2:]: #the mentions are documented from the third line
            cache_mention = mention.split('\t')
            cache_mention[:] = [item for item in cache_mention if item != ''] #some lists contain empty elements because of the ugly format :P
            if int(cache_mention[1]) < abstract_offset:
                cache_dict = {
                    'start': int(cache_mention[1]),
                    'end': int(cache_mention[2]),
                    'text': cache_mention[3],
                    'type': cache_mention[4],
                    'id': cache_mention[5]}
                title_mentions.append(cache_dict)
            else:
                cache_dict = {
                    'start':  int(cache_mention[1])-abstract_offset,
                    'end': int(cache_mention[2])-abstract_offset,
                    'text': cache_mention[3],
                    'type': cache_mention[4],
                    'id': cache_mention[5]}
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
        parsed_corpus_list.append({'docid': docid, 'sections': sections})
    return parsed_corpus_list
