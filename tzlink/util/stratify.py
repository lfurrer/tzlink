#!/usr/bin/env python
# coding: utf8

# Author: Lenz Furrer, 2015


'''
Tools for multi-category, incremental stratification.
'''


from collections import Counter, defaultdict
import random
import numbers
import itertools as it


def shuffle_TSV(cat_cols, infile, outfile):
    '''
    Stratify-shuffle TSV data.

    Shuffle the lines in a way that any subset starting
    from the beginning of the file looks like a stratified
    fold, ie. its distribution resembles that of the
    complete data set with regard to certain categories.
    '''
    lines, cats = {}, {}
    for id_, line in enumerate(infile):
        fields = line.split('\t', max(cat_cols)+1)
        lines[id_] = line
        cats[id_] = [fields[c] for c in cat_cols]

    for id_ in shuffle(cats):
        outfile.write(lines[id_])


def shuffle(data):
    '''
    Iterate over stratified-shuffled data.

    Args:
        data (dict):
            keys (any hashable): member ID
            values (iterable): stratification categories

    Returns:
        iter: iterate over member IDs
    '''
    yield from IncrementalSampler.from_data(data)


def folds(data, n):
    '''
    Divide data into n folds using multiple independent categories..

    An instance of IncrementalSampler is created for
    each fold, with a shared pool of data points to
    sample from.

    Args:
        data (dict):
            keys (any hashable): member ID
            values (iterable): stratification categories
        n (int): number of folds

    Returns:
        list of lists: a list of data keys for each fold
    '''
    shared_resources = IncrementalSampler.scan_data(data)
    samplers = [IncrementalSampler(*shared_resources) for _ in range(n)]
    folds_ = [[] for _ in range(n)]
    try:
        for i in it.cycle(range(n)):
            folds_[i].append(next(samplers[i]))
    except StopIteration:
        return folds_


class LabelSuggester:
    """
    Infinite label suggester for incremental sampling.

    Used as an (infinite) iterator, reproduces the original
    distribution of the class labels over and over.

    By using a combination of .due() and .update(), the
    suggester can be used for relaxed stratification:
    Use .due() to inspect the list of next-due labels,
    then inform the suggester about the label actually
    used with .update().

    If the class labels are all numeric, ties are broken
    by favoring those labels that make the mean the closest
    to that of the original distribution.
    """
    def __init__(self, dist):
        '''
        dist (Counter(key: label, value: frequency)):
            distribution of the class labels
        '''
        self._dist = dist
        self._used = {k: 0 for k in self._dist}
        self._target_mean = None
        if all(isinstance(k, numbers.Number) for k in dist):
            self._target_mean = sum(dist.elements())/sum(dist.values())

    def __iter__(self):
        return self

    def __next__(self):
        '''
        Pop the next due label.
        '''
        due = self.due()[0]
        self.update(due)
        return due

    def due(self):
        '''
        Sort all labels by dueness.

        Returns:
            list: all label keys, most due first
        '''
        if self._target_mean is None:
            _sort_key = self._sort_key
        else:
            optimal_sum = self._target_mean*(sum(self._used.values())+1)
            optimal_label = optimal_sum - sum(k*f for k, f in self._used.items())
            def _sort_key(label):
                return (self._sort_key(label), abs(label-optimal_label))
        return sorted(self._dist, key=_sort_key)

    def _sort_key(self, label):
        return self._used[label]/self._dist[label]

    def update(self, label):
        '''
        Inform the stratifier about the label actually used.
        '''
        self._used[label] += 1


class IncrementalSampler:
    """
    Incremental sampler for multiple, independent categories.

    Incrementally create a stratified-like sample by
    iterating over an instance of IncrementalSampler.
    Each category is stratified independently.
    At any stage, the cumulated sample is distributed as
    close to the whole data as possible.
    """
    def __init__(self, index, dists, comb):
        self._index = index
        self._dists = tuple(LabelSuggester(d) for d in dists)
        self._combinations = comb

    @classmethod
    def from_data(cls, data):
        '''
        data (dict):
            keys (any hashable): member ID
            values (iterable): stratification class label
        '''
        return cls(*cls.scan_data(data))

    @classmethod
    def scan_data(cls, data):
        '''
        Create index and compute class distributions.

        Args:
            data (dict): identifier mapped to class labels

        Returns:
            a triple (dict of list,
                      list of Counter,
                      callable):
                index, distributions, combinations
        '''
        index = defaultdict(list)
        dists = defaultdict(Counter)
        for key, categories in data.items():
            index[tuple(categories)].append(key)
            for i, label in enumerate(categories):
                dists[i][label] += 1

        for keys in index.values():
            random.shuffle(keys)

        index = dict(index)
        dists = [dists[i] for i in sorted(dists)]
        comb = cls._setup_comb(map(len, dists))
        return index, dists, comb

    def __iter__(self):
        return self

    def __next__(self):
        '''
        Get the next entry key.
        '''
        due_labels = [d.due() for d in self._dists]
        for comb in self._combinations(due_labels):
            try:
                best = self._index[comb].pop()
            except (KeyError, IndexError):
                # This combination doesn't exist (anymore)
                # -> back off to trying another combination.
                continue
            else:
                # The best combination has been found.
                # Inform each level's suggester of the choice made.
                for s, d in zip(comb, self._dists):
                    d.update(s)
                break
        else:
            # All combinations have been tried without success
            raise StopIteration

        return best

    @staticmethod
    def _setup_comb(cat_lengths):
        '''
        Set up the order of the backoff category combinations.

        Example:
        Two categories: [A B C D] and [x y].
        Suppose the class labels are due as [D C B A] and
        [y x], respectively. The best match for the next
        item would be (D, y). If there is no item with
        this combination, however, we need to back off,
        using a slightly worse combination. The more
        classes there are per category, the less harm
        is in taking the wrong class. Thus, first
        back off in the most diverse category,
        ie. try to use (C, y), then (D, x), (B, y) etc.

        Args:
            cat_lengths (iterable of int):
                number of class labels per category

        Returns:
            callable: item selector
        '''
        indices = [range(l) for l in cat_lengths]
        # Indices normalised to the range [0..1):
        normalised = [[j/len(f) for j in f] for f in indices]
        # Index combinations sorted by the sum of the normalised indices.
        combinations = sorted(
            it.product(*indices),
            key=lambda t: sum(normalised[c][l] for c, l in enumerate(t)))

        def selector(sequences):
            'Iterate over category combinations in backoff order.'
            for comb in combinations:
                yield tuple(sequences[c][l] for c, l in enumerate(comb))

        return selector
