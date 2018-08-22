#!/usr/bin/env python3
# coding: utf8


'''
Entry point for running the CNN.
'''


import argparse


def main():
    '''
    Run as script.
    '''
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        '-t', '--train', action='store_true',
        help='train a new CNN ranking model '
             '(instead of loading a previously trained one)')
    ap.add_argument(
        '-p', '--predict', action='store_true',
        help='produce a TSV with occurrence-wise predictions')
    ap.add_argument(
        '-d', '--detailed', action='store_true',
        help='produce a rich output file for inspecting ranking decisions')
    ap.add_argument(
        '-e', '--evaluation', action='store_true',
        help='produce two evaluation scripts for trec')
    ap.add_argument(
        '-r', '--record', action='store_true',
        help='create a summary with results and config info')
    ap.add_argument(
        '-m', '--model', metavar='PATH', dest='dumpfn',
        help='path for dumping and loading a trained model')
    ap.add_argument(
        '-c', '--config', metavar='PATH', nargs='+', default=(),
        help='config file(s) for overriding the defaults')
    args = ap.parse_args()

    from . import run
    run(**vars(args))


if __name__ == '__main__':
    main()
