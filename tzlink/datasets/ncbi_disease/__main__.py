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
        'meta', metavar='METADIR',
        help='path to UMLS Metathesaurus directory')
    ap.add_argument(
        'target', nargs='?', default='-', metavar='TARGET',
        help='destination file (default: write to STDOUT)')
    args = ap.parse_args()

    from . import extend_MEDIC_with_UMLS
    extend_MEDIC_with_UMLS(args.medic, args.meta, args.target)


if __name__ == '__main__':
    main()
