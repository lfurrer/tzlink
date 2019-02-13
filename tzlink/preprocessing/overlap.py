#!/usr/bin/env python3
# coding: utf8

# Author: Lenz Furrer, 2018


'''
Overlap between token sequences.
'''


from .tokenization import create_tokenizer
from .stem import PorterStemmer


class TokenOverlap:
    '''
    Compute token overlap between two texts.
    '''
    def __init__(self):
        self._tokenize = create_tokenizer('charclass')
        self._stem = PorterStemmer().stem
        self._cached_text = None
        self._cached_tokens = None

    def overlap(self, query, answer):
        '''
        Compute the Jaccard index of the stemmed tokens.
        '''
        if not isinstance(answer, str):  # allow a sequence of str for answer
            return max(self.overlap(query, a) for a in answer)
        q_toks = self.tokens(query, cache=True)
        a_toks = self.tokens(answer)
        intersection = q_toks.intersection(a_toks)
        union = q_toks.union(a_toks)
        return len(intersection)/len(union)

    def tokens(self, text, cache=False):
        '''
        Get a set of stemmed tokens.
        '''
        if cache and text == self._cached_text:
            toks = self._cached_tokens
        else:
            toks = self._tokens(text)
            if cache:
                self._cached_text, self._cached_tokens = text, toks
        return toks

    def _tokens(self, text):
        return set(self._stem(t) for t in self._tokenize(text))
