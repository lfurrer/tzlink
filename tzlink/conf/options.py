#!/usr/bin/env python3
# coding: utf8

# Author: Lenz Furrer, 2019


'''
Shared CLI options.
'''


import argparse


def get_argparser(desc='Run the CNN.', pre=(), post=()):
    '''
    Common argument parser for CLIs.
    '''
    ap = argparse.ArgumentParser(description=desc)

    # Custom prepended arguments.
    for args, kwargs in pre:
        ap.add_argument(*args, **kwargs)

    # Shared arguments.
    ap.add_argument(
        '-t', '--train', action='store_true',
        help='train a new model '
             '(instead of loading a previously trained one)')
    ap.add_argument(
        '-p', '--predict', nargs='+', type=str.lower, default=[], metavar='FMT',
        choices=['summary', 'rich', 'bionlp', 'trec'],
        help='produce predictions in one or more formats '
             '(summary: a line per occurrence,'
             ' rich: all candidates scored,'
             ' bionlp: stand-off annotations,'
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

    # Custom appended arguments.
    for args, kwargs in post:
        ap.add_argument(*args, **kwargs)

    return ap
