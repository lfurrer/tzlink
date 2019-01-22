#!/usr/bin/env python3
# coding: utf8


"""
An early-stopping callback based on classification accuracy.
"""


from ..rank.callback import EarlyStoppingRankingAccuracy
from .predictions import Evaluator


class EarlyStoppingClassificationAccuracy(EarlyStoppingRankingAccuracy):
    '''Stop training when classification accuracy has stopped improving.'''

    evaltype = Evaluator
