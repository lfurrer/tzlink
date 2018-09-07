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
    ap = argparse.ArgumentParser(description='Run the CNN.')
    ap.add_argument(
        '-t', '--train', action='store_true',
        help='train a new CNN ranking model '
             '(instead of loading a previously trained one)')
    ap.add_argument(
        '-p', '--predict', nargs='+', choices=['summary', 'rich', 'trec'],
        type=str.lower, default=[], metavar='FMT',
        help='produce predictions in one or more formats '
             '(summary: a line per occurrence,'
             ' rich: all candidates scored,'
             ' trec: two tables for TREC)')
    ap.add_argument(
        '-r', '--record', action='store_true',
        help='create a summary with results and config info')
    ap.add_argument(
        '-m', '--model', metavar='PATH', nargs='+', default=(), dest='dumpfns',
        help='path(s) for dumping and loading trained models')
    ap.add_argument(
        '-c', '--config', metavar='PATH', nargs='+', default=(),
        help='config file(s) for overriding the defaults')
    args = ap.parse_args()

    from . import run
    run(**vars(args))


if __name__ == '__main__':
    main()
