#!/usr/bin/env python3
# coding: utf8

# Author: Lenz Furrer, 2018


'''
Entry point for extending MEDIC terminology with UMLS.
'''


import argparse


def main():
    '''
    Run as script.
    '''
    ap = argparse.ArgumentParser(description='Extend MEDIC with UMLS.')
    ap.add_argument(
        'medic', metavar='MEDIC',
        help='path to MEDIC TSV')
    ap.add_argument(
        'metadir', metavar='METADIR',
        help='path to UMLS Metathesaurus directory')
    ap.add_argument(
        'target', nargs='?', default='-', metavar='TARGET',
        help='destination file (default: write to STDOUT)')
    ap.add_argument(
        '-d', '--divide', action='store_true',
        help='divide polysemous concepts using UMLS')
    ap.add_argument(
        '-n', '--names', action='store_true',
        help='add more synonyms from UMLS')
    ap.add_argument(
        '-f', '--definitions', action='store_true',
        help='complement missing definitions using UMLS')
    args = ap.parse_args()

    if not any((args.divide, args.names, args.definitions)):
        ap.error('nothing to do without at least one of -d/-n/-f')

    from . import extend_MEDIC_with_UMLS
    extend_MEDIC_with_UMLS(**vars(args))


if __name__ == '__main__':
    main()
