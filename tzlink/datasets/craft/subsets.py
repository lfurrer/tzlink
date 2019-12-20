#!/usr/bin/env python3
# coding: utf8

# Author: Lenz Furrer, 2019


'''
Filename listings for train/dev/test and different folds.
'''


import itertools as it


def docs(subset):
    '''
    Get document IDs for the given subset.
    '''
    # Predefined test set.
    if subset == 'test':
        return _test

    # Folded train/dev split.
    if subset in ('dev', 'train'):  # no fold number: default to 0.
        subset += '0'

    label, n = subset[:-1], int(subset[-1])
    if label == 'dev':
        return _folds[n]
    if label == 'train':
        ids = it.chain(*_folds[:n], *_folds[n+1:])
        return list(ids)
    raise ValueError('invalid subset: {}'.format(subset))


# Test set defined by the shared-task organisers (30 docs).
_test = '''
11319941
11604102
14624252
14675480
14691534
15018652
15070402
15238161
15328538
15560850
15615595
15619330
15784609
15850489
15882093
16026622
16027110
16410827
16517939
16611361
16787536
16800892
16968134
17029558
17201918
17206865
17465682
17503968
17565376
17677002
'''.split()


# 6 folds over the training data (12/11/11/11/11/11 docs)
_dev0 = '''
16870721
12079497
15760270
15492776
15588329
17696610
15819996
15836427
15921521
16103912
16539743
17244351
'''.split()


_dev1 = '''
16221973
15328533
15630473
15917436
16279840
16507151
16098226
17069463
17590087
15876356
17194222
'''.split()


_dev2 = '''
15005800
15676071
15061865
15938754
17608565
16504174
12925238
16362077
16110338
12585968
16700629
'''.split()


_dev3 = '''
11897010
17447844
17083276
17022820
16121255
16255782
17425782
14737183
16433929
15550985
16628246
'''.split()


_dev4 = '''
12546709
15320950
16216087
16121256
16670015
16504143
15345036
14723793
17002498
17020410
16109169
'''.split()


_dev5 = '''
15040800
17078885
15314655
14611657
14609438
15207008
11597317
16579849
16462940
15314659
11532192
'''.split()


_folds = [_dev0, _dev1, _dev2, _dev3, _dev4, _dev5]
