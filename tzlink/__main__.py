#!/usr/bin/env python3
# coding: utf8


'''
Entry point for running the CNN.
'''


from .conf.options import get_argparser
from .util.util import pack


def main():
    '''
    Run as script.
    '''
    # Insert a command-line arg for mode switching.
    mode = pack(
        'mode', choices=['rank', 'clsf'],
        help='ranking or classification approach?')
    ap = get_argparser(pre=[mode])
    args = ap.parse_args()

    from . import launch
    launch.launch(**vars(args))


if __name__ == '__main__':
    main()
