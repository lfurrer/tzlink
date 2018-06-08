#!/usr/bin/env python3
# coding: utf8

# Author: Lenz Furrer, 2018


'''
Utility for recording results and configurations.
'''


import io
import time
import logging

from ..util.util import smart_open, get_commit_info


TEMPLATE = '''\
# Commit hash:

{}


# Commit message:

{}


# Execution timestamp:

{}


# Results:

{}


# Log:

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
        self.timestamp = self._reformat_timestamp(conf.general.timestamp)
        self.log = self._capture_log()
        self.results = io.StringIO()

    @staticmethod
    def _reformat_timestamp(timestamp):
        return time.strftime('%Y-%m-%d %H:%M:%S',
                             time.strptime(timestamp, '%Y%m%d-%H%M%S'))

    def _capture_log(self):
        logger = logging.getLogger()  # root logger
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setLevel(self.conf.logging.level)
        handler.setFormatter(logging.Formatter(self.conf.logging.format))
        logger.addHandler(handler)
        return stream

    def dump(self):
        '''
        Write a formatted record.
        '''
        destination = self.conf.logging.summary_fn
        with smart_open(destination, 'w') as f:
            f.write(TEMPLATE.format(self.commit_hash,
                                    self.commit_msg,
                                    self.timestamp,
                                    self.results.getvalue(),
                                    self.log.getvalue(),
                                    self.conf.dump))
