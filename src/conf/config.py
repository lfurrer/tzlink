#!/usr/bin/env python3
# coding: utf8

# Author: Lenz Furrer, 2018


'''
Configuration handling.
'''


import os
import configparser as cp


HERE = os.path.dirname(__file__)
DEFAULTS = os.path.join(HERE, 'defaults.cfg')


class Config:
    '''
    Gobally usable class for accessing config parameters.

    All parameters are accessible as nested attributes, eg.
        >>> cfg = Config()
        >>> cfg.rank.n_kernels
        50
    '''

    # Default values determined at runtime.
    DYNAMIC_DEFAULTS = {
        'rootpath': os.path.realpath(os.path.join(HERE, '..', '..')),
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

    def _store(self, items):
        for sec_name, sec_proxy in items:
            if sec_name == 'DEFAULT':
                continue
            section = _Namespace()
            setattr(self, sec_name, section)
            for param in sec_proxy:
                value = self._guess_type(sec_proxy, param)
                setattr(section, param, value)

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


class _Namespace:
    '''A plain object with a __dict__ attribute.'''
    pass
