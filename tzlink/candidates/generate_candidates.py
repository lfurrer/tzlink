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

from ..datasets.terminology import DictEntry
from ..util.util import identity, OrderedCounter


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
        'composite': CompositeCandidates,
    }[name.lower()]
    if args:
        args = ast.literal_eval(args)

    return cls(sampler, *args)


class _BaseCandidateGenerator:
    def __init__(self, shared):
        self.terminology = shared.terminology
        self.scores = 1  # number of scores (>1 for multi-generator)
        self.null_score = 0  # score for the worst possible candidate

    def samples(self, mention, ref_ids, oracle=0):
        '''
        Iterate over quadruples <candidate, score, definition, label>.

        The candidate is a name string.
        The score correspondes to the confidence of the
        candidate generator.
        The label is True and False for positive and negative
        samples, respectively.

        Each synonym generates a separate positive sample.

        If oracle is 0, some names might be lacking positive
        samples (whenever the candidate retrieval mechanism
        can't find any). Otherwise, positive samples are
        taken from the ground-truth data until the specified
        number is reached (if there are that many names for
        the reference ID).
        '''
        return self._samples(mention, ref_ids, oracle)

    def samples_many(self, items, oracle=0):
        '''
        Generate samples for multiple mentions.

        Args:
            items: an iterable of triples <mention, ref_ids, context>.
                The iterable must allow repeated iteration.
            oracle: include unreachable positive examples?

        Return a nested iterator:
            <mention, context, samples> for each mention
                <candidate, score, definition, label> for each sample
        '''
        self.precompute([m for m, _, _ in items])
        for mention, ref_ids, context in items:
            samples = self._samples(mention, ref_ids, oracle)
            yield mention, context, samples

    def _samples(self, mention, ref_ids, oracle):
        candidates = self.scored_candidates(mention)
        all_positive = self._positive_samples(ref_ids)
        positive = all_positive.intersection(candidates)
        negative = set(candidates).difference(positive)
        if len(positive) < oracle:
            missing = list(all_positive.difference(positive))
            positive.update(self.topN(mention, missing, oracle-len(positive)))

        for subset, label in ((positive, True), (negative, False)):
            for cand in subset:
                score = candidates.get(cand, self.null_score)
                def_ = self._definition(ref_ids, cand)
                yield cand, score, def_, label

    def _positive_samples(self, ref_ids):
        self.terminology_update(ref_ids)
        positive = self.terminology.names(ref_ids)
        return positive

    def _definition(self, ref_ids, name):
        defs = self.terminology.definitions(ref_ids, name)
        return max(defs, key=len, default='')

    @staticmethod
    def terminology_update(ref_ids):
        '''
        Subclass hook for dynamic terminology changes.
        '''
        # Default is a no-op.
        del ref_ids

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

    def topN(self, mention, candidates, n):
        '''
        Rank the given candidates for mention and take the best n.
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

    def terminology_update(self, ref_ids):
        for g in self.generators:
            g.terminology_update(ref_ids)

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

    def topN(self, mention, candidates, n):
        best = OrderedCounter()
        for g in self.generators:
            best.update(g.topN(mention, candidates, n))
        return (c for c, _ in best.most_common(n))


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
        self._name_index = {n: i for i, n in enumerate(self._names)}
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

    def _sgram_vector(self, name):
        voc = len(self._sgram_index)
        sgrams = np.zeros(voc+1)
        for gram in self._preprocess(name):
            i = self._sgram_index.get(gram, voc)
            sgrams[i] += 1
        L2normalize(sgrams)
        return sgrams

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
        sim = self._similarities(mention)
        return self._select_candidates(
            sim, self.size, self.threshold, self._names)

    def topN(self, mention, candidates, n):
        candidates = [c for c in candidates if c in self._name_index]
        cand_i = [self._name_index[c] for c in candidates]
        sim = self._similarities(mention, limit=cand_i)
        return (c for c, _ in self._select_candidates(sim, n, 0, candidates))

    def _similarities(self, mention, limit=None):
        if limit is None:
            select = identity
            limit = slice(None)
        else:
            select = lambda m: m[limit]

        try:
            # Try to find it in the cache.
            i = self._cache[0][mention]
        except (TypeError, KeyError):
            # Compute the similarity with every candidate.
            sgrams = self._sgram_vector(mention)
            sim = select(self._sgram_matrix).dot(sgrams)
        else:
            sim = self._cache[1][limit, i]
        return sim

    @staticmethod
    def _select_candidates(sim, size, threshold, names):
        if size and len(names) > size:
            indices = np.argpartition(-sim, range(size))[:size]
        else:
            indices = np.argsort(-sim)
        for i in indices:
            cos = sim[i]
            if cos < threshold:
                break
            yield names[i], cos


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
        phrases = []
        for name in self.terminology.iter_names():
            # This iterates over unique names.
            vectors.vocab[name] = Vocab(index=len(vectors.vocab), count=None)
            vectors.index2word.append(name)
            phrases.append(self._phrase_vector(name))
        vectors.syn0 = vectors.syn0norm = np.array(phrases)
        return vectors

    def _phrase_vector(self, phrase):
        indices = self._vectorizer.indices(phrase)
        if not indices:
            indices = [self._vectorizer.PAD]  # avoid empty arrays
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

    def topN(self, mention, candidates, n):
        candidates = [c for c in candidates if c in self._pv.vocab]
        cand_i = [self._pv.vocab[c].index for c in candidates]
        cand_matrix = self._pv.syn0[cand_i]
        sim = cand_matrix.dot(self._phrase_vector(mention))
        if len(candidates) > n:
            top_i = np.argpartition(-sim, range(n))[:n]
        else:
            top_i = np.argsort(-sim)
        for i in top_i:
            yield candidates[i]


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

    def topN(self, mention, candidates, n):
        # There's no way to rank the candidates, just pick the first n.
        del mention
        return candidates[:n]


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


class CompositeCandidates(_NonRankedCandidates):
    '''
    Detect and decompose syntactically conflated terms.

    Address cases like "colorectal, endometrial, and ovarian cancers".

    This candidate generator does two things:
     1. It detects cases like the above and creates an
        unfolded candidate name like "colorectal cancers
        and endometrial cancers and ovarian cancers".
        This is done by combining existing terminology
        entries with the string " and ".
     2. It detects reference IDs that are linked to
        composite mentions and updates the terminology
        (a shared object controlled by the sampler) with
        new entries, constructed in the same way as above
        (concatenating names with " and ").
    These two actions are independent of one another.
    '''

    def __init__(self, shared, combinationlimit=1000):
        super().__init__(shared)
        self._name_index = self._index_names()
        self._seen = set()
        self._trigger = re.compile(r'\b(?:and/or|and|or|/)\b')
        self._comb_limit = max(1, combinationlimit)

    def _index_names(self):
        index = {}
        for name in self.terminology.iter_names():
            tokens = self._tokenize(name)
            index.setdefault(tokens, []).append(name)
        return index

    @staticmethod
    def _tokenize(text):
        return tuple(re.findall(r'\w+', text.lower()))

    def terminology_update(self, ref_ids):
        '''
        Create new dict entries for composite mentions.
        '''
        if ref_ids in self._seen:
            return
        self._seen.add(ref_ids)

        for id_ in ref_ids:
            if '|' not in id_:
                continue
            components = id_.split('|')
            comp_names = [self.terminology.names([i]) for i in components]
            names = tuple(self._compose(comp_names))
            if names:
                entry = DictEntry(names[0], id_, (), '', names[1:])
                self.terminology.add(entry)

    def _compose(self, comp_names):
        comp_names = self._limit_combinations(comp_names)
        for combination in it.product(*comp_names):
            yield ' and '.join(combination)

    def _limit_combinations(self, comp_names):
        '''
        Prevent combinatoric explosion when composing names.

        Heuristically remove unnecessary names until the length
        product is below the combination limit.
        '''
        if self._below_limit(comp_names):
            return comp_names
        comp_names = [set(c) for c in comp_names]  # copy to avoid side effect
        filters = (self._find_case_dups, self._find_parens, self._find_commas)
        filters = it.chain(filters, it.repeat(self._find_longest_half))
        for filter_ in filters:
            for comp in comp_names:
                comp.difference_update(filter_(comp))
                if self._below_limit(comp_names):
                    return comp_names

    def _below_limit(self, comp_names):
        p = 1
        for comp in comp_names:
            p *= len(comp)
            if p > self._comb_limit:
                return False
        return True

    @staticmethod
    def _find_case_dups(names):
        '''Find case-related duplicates.'''
        unique = {n.lower(): n for n in names}
        return names.difference(unique.values())

    @staticmethod
    def _find_parens(names):
        '''Find names with parentheses or brackets.'''
        return [n for n in names if re.search(r'[()\[\]{}]', n)]

    @staticmethod
    def _find_commas(names):
        '''Find names with commas.'''
        return [n for n in names if ',' in n]

    @staticmethod
    def _find_longest_half(names):
        '''Find the longer half of the names.'''
        bylength = sorted(names, key=len)
        cutoff = len(names) // 2
        return bylength[cutoff:]

    def _candidates(self, mention):
        for unfolding in self._unfold(mention):
            comp_names = [self._name_index.get(n, ()) for n in unfolding]
            yield from self._compose(comp_names)

    def _unfold(self, mention):
        '''
        Iterate over possible unfoldings.

        Convert "mod1, mod2 and mod_3 head" to
        [mod1+head, mod2+head, mod3+head].

        If there is no trigger ("and" etc.), yield nothing.
        If the part after the trigger has n tokens,
        yield n-1 items, one for each possible split into
        modifier/head.

        Each item is a list of modifier/head combinations.
        Each modifier/head combination is a tuple of tokens.
        '''
        try:
            mods, full = self._trigger.split(mention)
        except ValueError:  # too few or too many parts
            return
        mods = mods.strip().strip(',').split(',')
        *mods, full = (self._tokenize(p) for p in (*mods, full))
        mods = [mod for mod in mods if mod]  # remove empty matches
        if not mods:
            return
        for i in range(1, len(full)):
            last_mod, head = full[:i], full[i:]
            yield [mod+head for mod in (*mods, last_mod)]


def L2normalize(vector):
    '''
    Divide all components by the vector's magnitude.
    '''
    vector /= np.sqrt((vector**2).sum())
