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

import numpy as np
from scipy.sparse import csr_matrix
from gensim.models.keyedvectors import KeyedVectors, Vocab


def candidate_generator(sampler):
    '''
    Select and instantiate a candidate generator.

    Provide a sampler object to the candidate generators to
    allow sharing resources.
    '''
    # Account for multiple generators.
    values = sampler.conf.candidates.generator.split('\n')
    generators = [_create_generator(v.strip(), sampler) for v in values]
    if len(generators) > 1:
        return _MultiGenerator(sampler, generators)
    return generators[0]


def _create_generator(value, sampler):
    # Get generator name and arguments from the config value.
    name, args = re.fullmatch(r'(\w+)(.*)', value).groups()

    cls = {
        'sgramfixedset': SGramFixedSetCandidates,
        'sgramcosine': SGramCosineCandidates,
        'phrasevecfixedset': PhraseVecFixedSetCandidates,
    }[name.lower()]
    if args:
        args = ast.literal_eval(args)

    return cls(sampler, *args)


class _BaseCandidateGenerator:
    def __init__(self, shared):
        self.terminology = shared.terminology

    def samples(self, mention, ref_ids, oracle=False):
        '''
        Iterate over triples <candidate, score, label>.

        The candidate is a name string.
        The score correspondes to the confidence of the
        candidate generator.
        The label is True and False for positive and negative
        samples, respectively.

        Each synonym generates a separate positive sample.

        If oracle is True, positive samples are generated
        also for names that weren't found through the
        candidate retrieval mechanism.
        '''
        candidates = self.scored_candidates(mention)
        positive = self._positive_samples(ref_ids)
        negative = set(candidates).difference(positive)
        if not oracle:
            positive = positive.intersection(candidates)

        for subset, label in ((positive, True), (negative, False)):
            for cand in subset:
                score = candidates.get(cand, 0)
                yield cand, score, label

    def _positive_samples(self, ref_ids):
        ids = self._select_ids(ref_ids)
        positive = self.terminology.names(ids)
        return positive

    def candidates(self, mention):
        '''
        Compute a set of candidate names from the dictionary.
        '''
        raise NotImplementedError

    def scored_candidates(self, mention):
        '''
        Create a dict of candidate names mapped to a score.
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


class _MultiGenerator(_BaseCandidateGenerator):
    '''
    Wrapper for combining candidates from multiple generators.
    '''
    def __init__(self, shared, generators):
        super().__init__(shared)
        self.generators = generators

    def candidates(self, mention):
        return set().union(*(g.candidates(mention) for g in self.generators))

    def scored_candidates(self, mention):
        candidates = Counter()  # works with float values just fine
        for g in self.generators:
            # On collisions, Counter.update sums the values.
            candidates.update(g.scored_candidates(mention))
        return candidates

    def sorted_candidates(self, mention):
        '''
        Iterate over sorted candidates from all generators.

        The generators take turns in yielding a candidate.
        This requires that all generators support a
        sorted_candidates method.
        '''
        merged = it.zip_longest(*(g.sorted_candidates(mention)
                                  for g in self.generators))
        for round_ in merged:
            for candidate in round_:
                if candidate is not None:
                    yield candidate


class SGramFixedSetCandidates(_BaseCandidateGenerator):
    '''
    N best candidates based on absolute skip-gram overlap.
    '''

    def __init__(self, shared, size=20, sgrams=((2, 1), (3, 1))):
        super().__init__(shared)
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
        return set(self.sorted_candidates(mention))

    def scored_candidates(self, mention):
        scored = self.all_candidates(mention).most_common(self.size)
        return rank_scored(scored)

    def sorted_candidates(self, mention):
        '''
        Iterate over candidates, sorted by decreasing overlap.
        '''
        candidates = self.all_candidates(mention)
        return (c for c, _ in candidates.most_common(self.size))

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


class SGramCosineCandidates(SGramFixedSetCandidates):
    '''
    Candidates based on cosine similarity of skip-grams.

    If the size parameter is a non-zero number, the candidate
    sets are truncated to that length.
    Also, candidates with a cosine similarity below threshold
    are removed from the set.
    Inversely: if both size and threshold are 0, the
    candidate set consists of the entire terminology.
    '''

    def __init__(self, shared, threshold=.7, *args, **kwargs):
        super().__init__(shared, *args, **kwargs)
        self.threshold = threshold
        self._names = list(self.terminology.iter_names())
        self._sgram_matrix = self._create_matrix(self._names, update=True)

    def _create_index(self):
        '''
        Initialize a dict for the s-gram vocabulary.

        This is mainly a placeholder for overriding the superclass method.
        '''
        return {}

    def _create_matrix(self, names, update=False):
        data_triple = self._matrix_data(names, update)
        rows = len(data_triple[2]) - 1     # len(indptr) == rows+1
        cols = len(self._sgram_index) + 1  # add a column for unseen s-grams
        return csr_matrix(data_triple, shape=(rows, cols))

    def _matrix_data(self, names, update):
        '''
        Create a <data, indices, indptr> triple for a CSR matrix.
        '''
        indptr = [0]
        indices = []
        data = []
        vocabulary = self._sgram_index
        lookup = vocabulary.setdefault if update else vocabulary.get
        for name in names:
            sgrams = Counter(self._preprocess(name))
            indices.extend(lookup(s, len(vocabulary)) for s in sgrams)
            indptr.append(len(indices))
            row = np.fromiter(sgrams.values(), dtype=float, count=len(sgrams))
            L2normalize(row)
            data.append(row)
        return np.concatenate(data), indices, indptr

    def scored_candidates(self, mention):
        return next(iter(self.scored_candidates_many([mention])))

    def sorted_candidates(self, mention):
        return next(iter(self.sorted_candidates_many([mention])))

    def all_candidates(self, mention):
        '''
        Iterate over candidates and cosine similarity scores.
        '''
        return next(iter(self._scored_candidates_many([mention])))

    def scored_candidates_many(self, mentions):
        '''
        Iterate over a dict of scored candidates for each mention.
        '''
        return map(rank_scored, self._scored_candidates_many(mentions))

    def sorted_candidates_many(self, mentions):
        '''
        Nested iterator over candidate names, sorted by cosine similarity.
        '''
        for scored in self._scored_candidates_many(mentions):
            yield (c for c, _ in scored)

    def _scored_candidates_many(self, mentions):
        '''
        Nested iterator over candidate names with cosine similarity.
        '''
        sgrams = self._create_matrix(mentions)
        sim = self._sgram_matrix.dot(sgrams.T)
        sim = sim.toarray()
        for i in range(sim.shape[1]):
            yield self._cosine_scored_candidates(sim[:, i])

    def _cosine_scored_candidates(self, sim):
        if self.size:
            indices = np.argpartition(-sim, range(self.size))[:self.size]
        else:
            indices = np.argsort(-sim)
        for i in indices:
            cos = sim[i]
            if cos < self.threshold:
                break
            yield self._names[i], cos


class PhraseVecFixedSetCandidates(_BaseCandidateGenerator):
    '''
    N best candidates based on phrase-vector similarity.
    '''

    def __init__(self, shared, size=20, comb='sum'):
        super().__init__(shared)
        self.size = size
        self.combine = getattr(np, comb)
        self._vectorizer = shared.vectorizer
        self._wv = shared.emb_matrix
        self._pv = self._create_pv()

    def _create_pv(self):
        try:
            vectors = KeyedVectors()
        except TypeError:
            # Newer versions of gensim require a constructor argument.
            vectors = KeyedVectors(self._wv.shape[1])
        for name in self.terminology.iter_names():
            # This iterates over unique names.
            vectors.vocab[name] = Vocab(index=len(vectors.vocab), count=None)
            vectors.index2word.append(name)
            vectors.syn0.append(self._phrase_vector(name))
        vectors.syn0 = vectors.syn0norm = np.array(vectors.syn0)
        return vectors

    def _phrase_vector(self, phrase):
        indices = self._vectorizer.indices(phrase)
        vectors = [self._wv[i] for i in indices]
        phrase = self.combine(vectors, axis=0)
        L2normalize(phrase)
        return phrase

    def candidates(self, mention):
        return set(self.sorted_candidates(mention))

    def scored_candidates(self, mention):
        return rank_scored(self._scored_candidates(mention))

    def sorted_candidates(self, mention):
        '''
        Iterate over candidates, sorted by decreasing similarity.
        '''
        return (c for c, _, in self._scored_candidates(mention))

    def _scored_candidates(self, mention):
        '''
        Candidates with actual similarity scores.
        '''
        vector = self._phrase_vector(mention)
        return self._pv.most_similar(positive=[vector], topn=self.size)


def rank_scored(scored):
    '''
    Convert arbitrary scores to 1/rank scores.
    '''
    candidates = {}
    rank, previous = 0, None
    for cand, score in scored:
        if score != previous:
            rank += 1
            previous = score
        candidates[cand] = 1/rank
    return candidates


def L2normalize(vector):
    '''
    Divide all components by the vector's magnitude.
    '''
    vector /= np.sqrt((vector**2).sum())
