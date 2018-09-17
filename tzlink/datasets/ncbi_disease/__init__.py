#!/usr/bin/env python3
# coding: utf8

# Author: Lenz Furrer, 2018


'''
Terminology and corpus for the NCBI Disease dataset.
'''


from .term import parse_MEDIC_terminology, extend_MEDIC_with_UMLS
from .corpus import parse_NCBI_disease_corpus
