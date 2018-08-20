#!/usr/bin/env python3
# coding: utf8

# Author: Lenz Furrer, 2018


'''
Miscellaneous utilities.
'''


import io
import os
import sys
import gzip
import subprocess as sp
from collections import defaultdict, Counter, OrderedDict
from contextlib import contextmanager

from ..conf.config import Config


def get_config(config=None):
    '''
    Resolve different ways of specifying config.
    '''
    # Already instantiated Config object.
    if isinstance(config, Config):
        return config
    # Nothing given -- use the plain default.
    if config is None:
        return Config()
    # A string -- interpret as single file name.
    if isinstance(config, str):
        return Config(config)
    # Anything else -- treat as a sequence of file names.
    return Config(*config)


def smart_open(path, mode='r', **kwargs):
    '''
    Try to be clever at opening a file.
    '''
    if 'b' not in mode:
        kwargs.setdefault('encoding', 'utf8')

    if path is None:
        return io.StringIO()  # return a memory buffer to act as a dummy
    if path == '-':
        return std(mode)

    # For actual on-disk files, create missing parent dirs.
    if 'w' in mode:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    if path.endswith('.gz'):
        if 'b' not in mode and 't' not in mode:
            # If text/binary isn't specified, default to text mode.
            mode += 't'
        return gzip.open(path, mode, **kwargs)
    return open(path, mode, **kwargs)


@contextmanager
def std(mode):
    '''
    Allow using STDIN/OUT in a with statement without closing it on exit.
    '''
    # Enter: Get and yield the right file object.
    if 'w' in mode:
        f = sys.stdout
    else:
        f = sys.stdin
    if 'b' in mode:
        f = f.buffer
    yield f
    # Exit: Don't do anything.


def get_commit_info(spec, fallback):
    '''
    Get some info about the current git commit.
    '''
    args = ['git', 'log', '-1', '--pretty=%{}'.format(spec)]
    compl = sp.run(args, stdout=sp.PIPE, cwd=os.path.dirname(__file__))
    if compl.returncode == 0:
        return compl.stdout.decode('utf8').strip()
    return '<no commit {}>'.format(fallback)


class CacheDict(defaultdict):
    '''
    A default dictionary with a 1-argument factory.
    '''

    __slots__ = ()

    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        self[key] = value = self.default_factory(key)
        return value


def identity(arg):
    '''Return a single argument unchanged.'''
    return arg


class OrderedCounter(Counter, OrderedDict):
    '''
    A Counter remembering the order of insertion.

    When using .most_common(), ties are broken by insertion order.
    '''


def co_init(func):
    '''
    Decorator for initializing a coroutine.

    Before something can be sent to a generator, it needs
    to be fast-forwarded to the first yield statement.
    This decorator takes care of this.
    '''
    def _wrap(*args, **kwargs):
        cr = func(*args, **kwargs)
        next(cr)
        return cr
    return _wrap
