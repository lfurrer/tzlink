#!/usr/bin/env python3
# coding: utf8

# Author: Lenz Furrer, 2018


'''
Entry point for extracting terminology from UMLS RRF files.
'''


import argparse


def main():
    '''
    Run as script.
    '''
    ap = argparse.ArgumentParser(description='Extract SNOMED from UMLS RRF.')
    ap.add_argument(
        'meta', metavar='METADIR',
        help='path to UMLS Metathesaurus directory')
    ap.add_argument(
        'target', nargs='?', default='-', metavar='TARGET',
        help='destination file (default: write to STDOUT)')
    args = ap.parse_args()

    from . import preprocess_SNOMED_terminology
    preprocess_SNOMED_terminology(args.meta, args.target)


if __name__ == '__main__':
    main()
