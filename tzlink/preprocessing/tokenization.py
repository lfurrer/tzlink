#!/usr/bin/env python3
# coding: utf8

# Author: Lenz Furrer, 2018


'''
Simple tokenization.
'''


import re


def create_tokenizer(name, model=None):
    '''
    Select and instantiate a tokenizer.
    '''
    if name == 'characters':
        # Iterate over the characters of a string.
        return iter
    if name == 'whitespace':
        # Simply split on whitespace.
        return str.split
    if name == 'charclass':
        # Tokenize on change of character class.
        pattern = re.compile(
            r'''\d+|            # match contiguous runs of digits
                [^\W\d_]+|      # or letters
                (?:[^\w\s]|_)+  # or other non-whitespace characters
                ''', re.VERBOSE)
        return pattern.findall
    if name == 'bpe':
        from subword_nmt.apply_bpe import BPE
        pretokenizer = create_tokenizer('charclass')
        with open(model, encoding='utf8') as f:
            bpe = BPE(f)
        def _tokenize(text):
            pretok = ' '.join(pretokenizer(text))
            tokens = bpe.segment(pretok)
            return tokens.split()
        return _tokenize
    raise ValueError('unknown tokenizer: {}'.format(name))
