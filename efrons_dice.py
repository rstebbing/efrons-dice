############################################
# File: efrons_dice.py                     #
# Copyright Richard Stebbing 2015.         #
# Distributed under the GPLv3 License.     #
# (See accompany file LICENSE or copy at   #
#  http://opensource.org/licenses/GPL-3.0) #
############################################

# Imports
from __future__ import print_function, division

import argparse
import sys
from itertools import chain
from operator import add
from time import time

import numpy as np


# num_unique_dice
def num_unique_dice(num_sides, num_labels):
    if num_labels <= 0:
        raise ValueError('num_labels <= 0 (= {})'.format(num_labels))
    if num_sides <= 0:
        raise ValueError('num_sides <= 0 (= {})'.format(num_sides))

    n = np.zeros(num_sides + 1, dtype=int)
    n[0] = 1
    for i in range(num_labels - 1):
        np.cumsum(n, out=n)
    return n.sum()


# enumerate_unique_dice
def enumerate_unique_dice(num_sides, num_labels):
    if num_labels <= 0:
        raise ValueError('num_labels <= 0 (= {})'.format(num_labels))
    if num_sides <= 0:
        return []

    def _enumerate_unique_dice_reverse(i, j):
        if i >= num_labels - 2:
            yield [num_sides, j]
        else:
            for k in range(j, num_sides + 1):
                for s in _enumerate_unique_dice_reverse(i + 1, k):
                    s.append(j)
                    yield s

    A_r = np.array(list(_enumerate_unique_dice_reverse(-1, 0)))
    A = A_r[:, ::-1]
    D = A[:, 1:] - A[:, :-1]
    return D


# num_wins_matrix
def num_wins_matrix(D):
    E = np.empty_like(D)
    E[:, 0] = 0
    np.cumsum(D[:, :-1], axis=1, out=E[:, 1:])
    return np.dot(D, E.T)


# argmax
def argmax(x, is_valid=None, return_max=False):
    x = np.asarray(x)
    if x.size <= 0:
        raise ValueError('x.size <= 0')

    if is_valid is not None:
        is_valid = np.asarray(is_valid, dtype=bool)
        x = x[is_valid]
        if x.size <= 0:
            assert np.count_nonzero(is_valid) == 0
            raise ValueError('count_nonzero(is_valid) == 0')

    k = np.argwhere(x == np.amax(x)).ravel()
    m = x[k[0]]
    if is_valid is not None:
        k = np.argwhere(is_valid).ravel()[k]
    return (k, m) if return_max else k


# no_cycle_max_sum
def no_cycle_max_sum(num_dice, W, is_start_end=None):
    if num_dice < 2:
        raise ValueError('num_dice < 2 (= {})'.format(num_dice))

    (num_unique_dice, _) = W.shape
    r = np.arange(num_unique_dice)

    L, V = [], []
    def _next(Z, is_start_end=None):
        Li = [argmax(z, is_start_end) for z in Z]
        L.append(Li)

        li = [l[0] for l in Li]
        Vi = Z[r, li]
        V.append(Vi)

        return W + Vi

    Z = _next(W, is_start_end)
    for i in range(num_dice - 1):
        Z = _next(Z)

    def _labels(i, l0):
        if i >= num_dice:
            yield l0
        else:
            for j in L[num_dice - 1 - i][l0[i]]:
                for l in _labels(i + 1, l0 + [j]):
                    yield l

    i, s = argmax(V[-1], is_start_end, return_max=True)
    return chain(*[_labels(0, [_i]) for _i in i]), s


# single_cycle_max_sum_linear_scan
def single_cycle_max_sum_linear_scan(num_dice, W, verbose=False):
    if num_dice < 2:
        raise ValueError('num_dice < 2 (= {})'.format(num_dice))

    (num_unique_dice, _) = W.shape

    is_start_end = np.zeros(num_unique_dice, dtype=bool)
    s, l = -1, []

    for i in range(num_unique_dice):
        if verbose:
            sys.stdout.write('\r{}/{} ({})'.format(i + 1, num_unique_dice, s))
            sys.stdout.flush()

        is_start_end[i] = True
        li, si = no_cycle_max_sum(num_dice, W, is_start_end)
        if si > s:
            del l[:]
            l.append(li)
            s = si
        elif si == s:
            l.append(li)
        is_start_end[i] = False

    if verbose:
        sys.stdout.write('\n')

    if l:
        l = chain(*l)
    return l, s


# main
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('num_sides', nargs='?', type=int, default=6)
    parser.add_argument('num_labels', nargs='?', type=int, default=7)
    parser.add_argument('num_dice', nargs='?', type=int, default=4)
    parser.add_argument('--no-output-progress', dest='output_progress',
                        default=True, action='store_false')
    args = parser.parse_args()

    # Efrons' dice: `num_sides = 6`, `num_labels = 7`, `num_dice = 4`.
    num_sides, num_labels, num_dice = (args.num_sides,
                                       args.num_labels,
                                       args.num_dice)

    print('num_sides:', num_sides)
    print('num_labels:', num_labels)
    print('num_dice:', num_dice)

    t0 = time()

    _num_unique_dice = num_unique_dice(num_sides, num_labels)
    print('num_unique_dice: {}'.format(_num_unique_dice))
    D = enumerate_unique_dice(num_sides, num_labels)
    assert D.shape == (_num_unique_dice, num_labels)

    W = num_wins_matrix(D)

    num_outcomes = num_sides * num_sides
    greater_than_one_half = num_outcomes // 2 + 1
    ij = np.nonzero(W < greater_than_one_half)
    W[ij] = -num_dice * num_outcomes

    unique_W = np.unique(W)
    for i, w in enumerate(unique_W):
        print('\n{}/{}>'.format(i + 1, unique_W.shape[0]), '*' if w < 0 else w)
        Wi = W.copy()
        if w >= 0:
            Wi[Wi != w] = -num_dice * num_outcomes
        l, s = single_cycle_max_sum_linear_scan(num_dice, Wi,
                                                verbose=args.output_progress)
        if s < 0:
            continue
        print('s:', s)
        for i, li in enumerate(l, start=0):
            print('{}:'.format(i), li[:-1])
            print(D[li[:-1]])
        print('({})'.format(i + 1))

    t1 = time()
    print('Time taken: {:.3f}s'.format(t1 - t0))
