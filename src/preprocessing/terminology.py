#!/usr/bin/env python3
# coding: utf8

# Author: Lenz Furrer, 2018


'''
Common tools for terminology resource processing.
'''


from collections import namedtuple


# Common format for terminology entries.
DictEntry = namedtuple('DictEntry', 'name id alt def_ syn')
