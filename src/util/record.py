#!/usr/bin/env python3
# coding: utf8

# Author: Lenz Furrer, 2018


'''
Utility for recording results and configurations.
'''


import io

from ..util.util import smart_open, get_commit_info


TEMPLATE = '''\
# Commit hash:

{}


# Commit message:

{}


# Results:

{}


# Configuration:

{}
'''


class Recorder:
    '''
    Collect and format information about results and config.
    '''

    def __init__(self, conf):
        self.conf = conf
        self.commit_hash = get_commit_info('H', 'hash')
        self.commit_msg = get_commit_info('B', 'message')
        self.results = io.StringIO()

    def dump(self):
        '''
        Write a formatted record.
        '''
        destination = self.conf.logging.summary_fn
        with smart_open(destination, 'w') as f:
            f.write(TEMPLATE.format(self.commit_hash,
                                    self.commit_msg,
                                    self.results.getvalue(),
                                    self.conf.dump))
