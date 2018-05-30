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
        help='use the model to produce rankings')
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
