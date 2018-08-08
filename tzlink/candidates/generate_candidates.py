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
        'sgramcosine': SGramCosineCandidates,
        'phrasevecfixedset': PhraseVecFixedSetCandidates,
        'symbolreplacement': SymbolReplacementCandidates,
        'hyperonym': HyperonymCandidates,
        'abbreviation': AbbreviationCandidates,
    }[name.lower()]
    if args:
        args = ast.literal_eval(args)

    return cls(sampler, *args)


class _BaseCandidateGenerator:
    def __init__(self, shared):
        self.terminology = shared.terminology
        self.scores = 1  # number of scores (>1 for multi-generator)
        self.null_score = 0  # score for the worst possible candidate

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
        return self._samples(mention, ref_ids, oracle)

    def samples_many(self, items, oracle=False):
        '''
        Generate samples for multiple mentions.

        Args:
            items: an iterable of pairs <mention, ref_ids>.
                The iterable must allow repeated iteration.
            oracle: flag for including unreachable positive
                examples

        Return a nested iterator:
            <mention, samples> for each mention
                <candidate, score, label> for each sample
        '''
        self.precompute([m for m, _ in items])
        for mention, ref_ids in items:
            samples = self._samples(mention, ref_ids, oracle)
            yield mention, samples

    def _samples(self, mention, ref_ids, oracle):
        candidates = self.scored_candidates(mention)
        positive = self._positive_samples(ref_ids)
        negative = set(candidates).difference(positive)
        if not oracle:
            positive = positive.intersection(candidates)

        for subset, label in ((positive, True), (negative, False)):
            for cand in subset:
                score = candidates.get(cand, self.null_score)
                yield cand, score, label

    def _positive_samples(self, ref_ids):
        positive = self.terminology.names(ref_ids)
        return positive

    @staticmethod
    def precompute(mentions):
        '''
        Precompute some cache values, if applicable.
        '''
        # Default is a no-op.
        del mentions

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


class _MultiGenerator(_BaseCandidateGenerator):
    '''
    Wrapper for combining candidates from multiple generators.
    '''
    def __init__(self, shared, generators):
        super().__init__(shared)
        self.generators = generators
        self.scores = len(self.generators)
        self.null_score = [g.null_score for g in self.generators]

    def candidates(self, mention):
        return set().union(*(g.candidates(mention) for g in self.generators))

    def scored_candidates(self, mention):
        return self._comb_scores(g.scored_candidates(mention)
                                 for g in self.generators)

    def _comb_scores(self, scored):
        candidates = defaultdict(lambda: list(self.null_score))  # copy!
        for i, c_s in enumerate(scored):
            for cand, score in c_s.items():
                candidates[cand][i] = score
        candidates.default_factory = None  # avoid closure (?)
        return candidates


class SGramCosineCandidates(_BaseCandidateGenerator):
    '''
    Candidates based on cosine similarity of skip-grams.

    If the size parameter is a non-zero number, the candidate
    sets are truncated to that length.
    Also, candidates with a cosine similarity below threshold
    are removed from the set.
    Inversely: if both size and threshold are 0, the
    candidate set consists of the entire terminology.
    '''

    def __init__(self, shared, threshold=.7, size=20, sgrams=((2, 1), (3, 1))):
        super().__init__(shared)
        self.threshold = threshold
        self.size = size
        self.shapes = sgrams  # <n, k>
        self._names = list(self.terminology.iter_names())
        self._sgram_index = {}  # s-gram vocabulary
        self._sgram_matrix = self._create_matrix(self._names, update=True)
        self._cache = None  # lookup, matrix

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

    def _similarities(self, name):
        voc = len(self._sgram_index)
        sgrams = np.zeros(voc+1)
        for gram in self._preprocess(name):
            i = self._sgram_index.get(gram, voc)
            sgrams[i] += 1
        L2normalize(sgrams)
        return self._sgram_matrix.dot(sgrams)

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

    def candidates(self, mention):
        return set(self.sorted_candidates(mention))

    def scored_candidates(self, mention):
        return dict(self.all_candidates(mention))

    def sorted_candidates(self, mention):
        '''
        Iterate over candidates, sorted by decreasing similarity.
        '''
        return (c for c, _ in self.all_candidates(mention))

    def precompute(self, mentions):
        '''
        Create a similarity matrix and a lookup table.
        '''
        sgrams = self._create_matrix(mentions)
        sim = self._sgram_matrix.dot(sgrams.T)
        sim = sim.toarray()
        lookup = {m: i for i, m in enumerate(mentions)}
        self._cache = (lookup, sim)

    def all_candidates(self, mention):
        '''
        Iterate over candidates and cosine similarity scores.
        '''
        try:
            i = self._cache[0][mention]
        except (TypeError, KeyError):
            # Compute the similarity with every candidate.
            sim = self._similarities(mention)
        else:
            sim = self._cache[1][:, i]
        return self._cosine_scored_candidates(sim)

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

    def __init__(self, shared, size=20, comb='sum', emb='emb'):
        super().__init__(shared)
        self.size = size
        self.combine = getattr(np, comb)
        self._vectorizer = shared.emb[emb].vectorizer
        self._wv = shared.emb[emb].emb_matrix
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
        return dict(self._scored_candidates(mention))

    def sorted_candidates(self, mention):
        '''
        Iterate over candidates, sorted by decreasing similarity.
        '''
        return (c for c, _ in self._scored_candidates(mention))

    def _scored_candidates(self, mention):
        '''
        Candidates with actual similarity scores.
        '''
        vector = self._phrase_vector(mention)
        return self._pv.most_similar(positive=[vector], topn=self.size)


class _NonRankedCandidates(_BaseCandidateGenerator):
    '''
    Abstract class for generators without complete ranking.
    '''

    def candidates(self, mention):
        return set(self._candidates(mention))

    def scored_candidates(self, mention):
        return {c: 1. for c in self._candidates(mention)}

    def _candidates(self, mention):
        '''Iterate over candidate names.'''
        raise NotImplementedError


class SymbolReplacementCandidates(_NonRankedCandidates):
    '''
    Names reachable through symbol/word substitution.
    '''

    numerals = {
        '1': ('one', 'single'),
        '2': ('two', 'double', 'ii'),
        '3': ('three', 'triple'),
        '4': ('four', 'quadruple'),
        '5': ('five',),
        '6': ('six',),
        '7': ('seven',),
        '8': ('eight',),
        '9': ('nine',),
    }

    other_symbols = {
        'and/or': 'and',
        '/': ' and ',
        ' (': '',
        '(': '',
        ')': '',
    }

    def __init__(self, shared, include_original=True):
        super().__init__(shared)
        self.include_original = include_original
        self._names = frozenset(self.terminology.iter_names())
        self._subs = self._compile_subs()
        self._symbols = self._build_regex()

    def _compile_subs(self):
        subs = {}
        for digit, words in self.numerals.items():
            subs[digit] = (digit, *words)
            for word in words:
                subs[word] = (word, digit)
        for sym, word in self.other_symbols.items():
            subs[sym] = (sym, word)
        return subs

    def _build_regex(self):
        # Make sure "and/or" comes before "/" in the list of alternatives.
        targets = sorted(self._subs, key=len, reverse=True)
        alts = '|'.join(re.escape(tgt) for tgt in targets)
        return re.compile('({})'.format(alts))

    def _candidates(self, mention):
        return (c for c, _ in self._scored_candidates(mention))

    def scored_candidates(self, mention):
        return {c: 1/(s+1) for c, s in self._scored_candidates(mention)}

    def sorted_candidates(self, mention):
        '''
        Iterate over candidates, sorted by number of substitutions.
        '''
        for c, _ in sorted(self._scored_candidates(mention),
                           key=lambda c: c[1]):
            yield c

    def _scored_candidates(self, mention):
        '''
        Candidates with scores indicating the number of substitutions.
        '''
        for variant, subs in self._generate_variants(mention):
            if subs or self.include_original:
                if variant in self._names:
                    yield variant, subs

    def _generate_variants(self, mention):

        # re.split() with a capturing group puts all separators (= targets)
        # at odd positions, ie. parts[1::2].
        parts = self._symbols.split(mention)
        if len(parts) <= 1:
            yield mention, 0  # include the unchanged word anyway
            return  # no match

        # Make a nested list of alternatives to produce combinatoric variants.
        # Each inner list has the non-changed alternative first.
        alternatives = [self._subs[match] for match in parts[1::2]]
        combinations = it.product(*alternatives)
        subs_counts = self._count_subsitutions(alternatives)
        for combination, subs in zip(combinations, subs_counts):
            parts[1::2] = combination
            yield ''.join(parts), subs

    @staticmethod
    def _count_subsitutions(alternatives):
        counts = [range(len(a)) for a in alternatives]
        for comb in it.product(*counts):
            yield sum(map(bool, comb))


class HyperonymCandidates(_NonRankedCandidates):
    '''
    Names containing only parts of the mention.
    '''

    stopwords = frozenset((
        'disease', 'diseases',
        'disorder', 'disorders',
        'condition', 'conditions',
    ))

    def __init__(self, shared):
        super().__init__(shared)
        self._token = re.compile(r'\w+')
        self._names = self._index_names()

    def _index_names(self):
        names = {}
        for name in self.terminology.iter_names():
            tokens = set(self._preprocess(name)).difference(self.stopwords)
            if len(tokens) == 1:
                (token,) = tokens
                names.setdefault(token, set()).add(name)
        return names

    def _preprocess(self, term):
        return self._token.findall(term.lower())

    def _candidates(self, mention):
        for token in self._preprocess(mention):
            if token in self._names:
                yield from self._names[token]


class AbbreviationCandidates(_NonRankedCandidates):
    '''
    Derived abbreviations.
    '''

    def __init__(self, shared):
        super().__init__(shared)
        self._abbrevs = self._index_names()

    def _index_names(self):
        names = {}
        for name in self.terminology.iter_names():
            abbrev = self._generate_abbrev(name)
            if not self.terminology.ids([name]) < self.terminology.ids([abbrev]):
                # Only use abbreviations that actually add new information.
                names.setdefault(abbrev, []).append(name)
        return names

    def _generate_abbrev(self, name):
        return ''.join(self._abbrev_tokens(name))

    @staticmethod
    def _abbrev_tokens(name):
        # Reverse comma-separated chunks.
        for chunk in reversed(re.split(r', *', name)):
            # Take all alphabetic initials and whole numbers.
            for token in re.findall(r'\w+', chunk):
                if token.isdigit():
                    yield token
                else:
                    initial = token[0]
                    if initial.isalpha():
                        yield initial

    def _candidates(self, mention):
        yield from self._abbrevs.get(mention, ())


def L2normalize(vector):
    '''
    Divide all components by the vector's magnitude.
    '''
    vector /= np.sqrt((vector**2).sum())
