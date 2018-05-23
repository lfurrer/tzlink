#!/usr/bin/env python3
# coding: utf8

# Author: Lenz Furrer, 2018


'''
Configuration handling.
'''


import os
import time
import logging
import configparser as cp


HERE = os.path.dirname(__file__)
DEFAULTS = os.path.join(HERE, 'defaults.cfg')


class _Namespace:
    '''A plain object with subscript access to its __dict__.'''
    def __getitem__(self, name):
        return getattr(self, name)

    def __setitem__(self, name, value):
        setattr(self, name, value)


class Config(_Namespace):
    '''
    Gobally usable class for accessing config parameters.

    All parameters are accessible as nested attributes, eg.
        >>> cfg = Config()
        >>> cfg.rank.n_kernels
        50
    If the parameter name is determined at runtime, use subscript:
        >>> cfg[dataset].train_fn
        'path/to/training.txt'
    '''

    # Default values determined at runtime.
    DYNAMIC_DEFAULTS = {
        'rootpath': os.path.realpath(os.path.join(HERE, '..', '..')),
        'timestamp': time.strftime('%Y%m%d-%H%M%S'),
    }

    def __init__(self, *filenames):
        '''
        Override the defaults with any number of files.
        '''
        parser = cp.ConfigParser(defaults=self.DYNAMIC_DEFAULTS,
                                 interpolation=cp.ExtendedInterpolation(),
                                 empty_lines_in_values=False)
        parser.read([DEFAULTS, *filenames])
        self._store(parser.items())
        self._setup_logging()

    def _store(self, items):
        for sec_name, sec_proxy in items:
            if sec_name == 'DEFAULT':
                continue
            section = _Namespace()
            self[sec_name] = section
            for param in sec_proxy:
                value = self._guess_type(sec_proxy, param)
                section[param] = value

    @classmethod
    def _guess_type(cls, section, param):
        '''
        Try typecasting to int, float, boolean (in that order).
        '''
        for func in (section.getint, section.getfloat, section.getboolean):
            try:
                return func(param)
            except ValueError:
                pass
        return section.get(param)

    def _setup_logging(self):
        logging.basicConfig(**{p: self['logging'][p]
                               for p in ('format', 'level')})
