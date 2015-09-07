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
        if i >= num_labels - 1:
            yield [num_sides, j]
        else:
            for k in range(j, num_sides + 1):
                for s in _enumerate_unique_dice_reverse(i + 1, k):
                    s.append(j)
                    yield s

    A_r = np.array(list(_enumerate_unique_dice_reverse(0, 0)))
    A = A_r[:, ::-1]
    D = A[:, 1:] - A[:, :-1]
    return D


# num_wins_matrix
def num_wins_matrix(D):
    E = np.empty_like(D)
    E[:, 0] = 0
    np.cumsum(D[:, :-1], axis=1, out=E[:, 1:])
    return np.dot(D, E.T)


# max_sum
def max_sum(A, B):
    A, B = np.atleast_2d(A, B)
    (num_rows_A, num_cols_A), (num_rows_B, num_cols_B) = A.shape, B.shape
    if num_cols_A != num_rows_B:
        raise ValueError('num_cols_A != num_rows_B ({} vs {})'.format(
            num_cols_A, num_rows_B))

    S = np.empty((num_rows_A, num_cols_B),
                 dtype=np.promote_types(A.dtype, B.dtype))
    for i in range(num_rows_A):
        X = A[i][:, np.newaxis] + B
        X.max(axis=0, out=S[i])

    return S


# efrons_dice
def efrons_dice(num_dice, W, verbose=False):
    if num_dice < 2:
        raise ValueError('num_dice < 2 (= {})'.format(num_dice))

    S, Z = W, []
    for i in range(num_dice - 1):
        Z.append(S)
        if verbose:
            sys.stdout.write('\rmax_sum: {}/{}'.format(i + 1, num_dice - 1))
            sys.stdout.flush()
        S = max_sum(W, S)
    if verbose:
        sys.stdout.write('\n')

    s = np.diag(S)
    m = s.max()
    if m < 0:
        return None, m

    def _l(i, lx, l0):
        x = W[lx] + Z[i][:, l0]
        Ly = np.flatnonzero(x == x.max())
        if i <= 0:
            for ly in Ly:
                yield [ly]
        else:
            for ly in Ly:
                for l in _l(i - 1, ly, l0):
                    l.append(ly)
                    yield l

    l0s = np.flatnonzero(s == m)
    def _ls():
        for l0 in l0s:
            for l in _l(num_dice - 2, l0, l0):
                l.append(l0)
                yield l[::-1]

    return _ls(), m


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
    print('num_outcomes: {}'.format(num_outcomes))
    greater_than_one_half = num_outcomes // 2 + 1
    ij = np.nonzero(W < greater_than_one_half)
    W[ij] = -num_dice * num_outcomes

    unique_W = np.unique(W)
    for i, w in enumerate(unique_W):
        print('\n{}/{}>'.format(i + 1, unique_W.shape[0]), '*' if w < 0 else w)
        Wi = W.copy()
        if w >= 0:
            Wi[Wi != w] = -num_dice * num_outcomes
        l, s = efrons_dice(num_dice, Wi, verbose=args.output_progress)

        if s < 0:
            continue

        print('s:', s)
        for i, li in enumerate(l):
            print('{}:'.format(i), li)
            print(D[li])
        print('({})'.format(i + 1))

    t1 = time()

    print('Time taken: {:.3f}s'.format(t1 - t0))
