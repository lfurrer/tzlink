#!/usr/bin/env python3
# coding: utf8

# Author: Lenz Furrer, 2019


'''
Entry point for extracting terminology from CRAFT OBO files.
'''


import argparse

from . import preprocess_CRAFT_terminology, SYNONYM_TYPES


def main():
    '''
    Run as script.
    '''
    ap = argparse.ArgumentParser(description='Extract SNOMED from UMLS RRF.')
    ap.add_argument(
        'craftdir', metavar='CRAFTDIR',
        help='path to the root of the CRAFT distribution')
    ap.add_argument(
        'ontology', metavar='ONTOLOGY',
        help='one of CHEBI, CHEBI_EXT, CL etc.')
    ap.add_argument(
        'target', nargs='?', default='-', metavar='TARGET',
        help='destination file (default: write to STDOUT)')
    ap.add_argument(
        '-s', '--synonym-types', nargs='+', default=['EXACT'],
        choices=SYNONYM_TYPES,
        help='include only synonyms of this type (default: %(default)s)')
    args = ap.parse_args()

    preprocess_CRAFT_terminology(**vars(args))


if __name__ == '__main__':
    main()
