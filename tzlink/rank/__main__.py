#!/usr/bin/env python3
# coding: utf8


'''
Entry point for running the ranking CNN.
'''


from ..conf.options import get_argparser


def main():
    '''
    Run as script.
    '''
    ap = get_argparser(desc='Run the CNN in ranking mode.')
    args = ap.parse_args()

    from .. import launch
    launch.launch(mode='rank', **vars(args))


if __name__ == '__main__':
    main()
