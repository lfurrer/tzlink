#!/usr/bin/env python3
# coding: utf8

# Author: Lenz Furrer, 2018


'''
Candidate generation.
'''


import re
import ast
import itertools as it
from collections import defaultdict, Counter


def candidate_generator(conf, terminology):
    '''
    Select and instantiate a candidate generator.
    '''
    # Get generator name and arguments from the config value.
    value = conf.candidates.generator
    name, args = re.fullmatch(r'(\w+)(.*)', value).groups()

    cls = {
        'sgramfixedset': SGramFixedSetCandidates,
    }[name.lower()]
    if args:
        args = ast.literal_eval(args)

    return cls(terminology, *args)


class _BaseCandidateGenerator:
    def __init__(self, terminology):
        self.terminology = terminology

    def samples(self, mention, ref_ids, oracle=False):
        '''
        Iterate over pairs <candidate, label>.

        The candidate is a name string.
        The label is True and False for positive and negative
        samples, respectively.

        Each synonym generates a separate positive sample.

        If oracle is True, positive samples are generated
        also for names that weren't found through the
        candidate retrieval mechanism.
        '''
        candidates = self.candidates(mention)
        positive = self._positive_samples(ref_ids)
        negative = candidates.difference(positive)
        if not oracle:
            positive = candidates.intersection(positive)

        for subset, label in ((positive, True), (negative, False)):
            for cand in subset:
                yield cand, label

    def _positive_samples(self, ref_ids):
        ids = self._select_ids(ref_ids)
        positive = self.terminology.names(ids)
        return positive

    def candidates(self, mention):
        '''
        Compute a set of candidate names from the dictionary.
        '''
        raise NotImplementedError

    @staticmethod
    def _select_ids(ids):
        '''
        Account for alternative and compound IDs in the reference.
        '''
        if len(ids) > 1:
            # Don't generate positive examples for compound concepts.
            return set()
        return ids[0]


class SGramFixedSetCandidates(_BaseCandidateGenerator):
    '''
    N best candidates based on absolute skip-gram overlap.
    '''

    def __init__(self, terminology, size=20, sgrams=((2, 1), (3, 1))):
        super().__init__(terminology)
        self.size = size
        self.shapes = sgrams  # <n, k>
        self._sgram_index = self._create_index()

    def _create_index(self):
        index = defaultdict(Counter)
        for name in self.terminology.iter_names():
            for sgram in self._preprocess(name):
                index[sgram][name] += 1
        # Freeze the s-gram index.
        return dict(index)

    def candidates(self, mention):
        candidates = self.all_candidates(mention)
        return set(c for c, _ in candidates.most_common(self.size))

    def all_candidates(self, mention):
        '''
        Create a Counter of all entries with *any* overlap.
        '''
        # Compute the absolute overlap of skip-grams.
        candidates = Counter()
        for sgram, m_count in Counter(self._preprocess(mention)).items():
            for cand, c_count in self._sgram_index.get(sgram, _D).items():
                candidates[cand] += min(m_count, c_count)
        # When sorting the candidates, resolve ties by preferring shorter names.
        for cand, count in candidates.items():
            candidates[cand] = (count, -len(cand))
        return candidates

    def _preprocess(self, text):
        text = self._lookup_normalize(text)
        for n, k in self.shapes:
            yield from self._skipgrams(text, n, k)

    @staticmethod
    def _lookup_normalize(text):
        return text.lower()

    @staticmethod
    def _skipgrams(text, n, k):
        for i in range(len(text)-n+1):
            head = (text[i],)
            for tail in it.combinations(text[i+1 : i+n+k], n-1):
                yield head + tail

# Empty dict instance used for some optimisation.
_D = {}
