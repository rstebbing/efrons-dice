efrons-dice
===========

[efrons_dice.py](efrons_dice.py) is a Python program which finds non-transitive dice combinations (e.g. [Efron's dice](http://en.wikipedia.org/wiki/Nontransitive_dice#Efron.27s_dice)) by selecting combinations of dice to maximise the *sum* of the number of times each dice beats its (next) neighbour.
Although this discrete optimisation problem contains a cycle, it can be "broken" and an algorithm akin to [`min-sum`](http://en.wikipedia.org/wiki/Belief_propagation) is run multiple times to find *all* maximum configurations.

Author: Richard Stebbing

License: GPLv3 (refer to LICENSE)


Dependencies
------------

This repository is tested to work under Python 2.7 and Python 3.4.

This repository requires Numpy.


Getting Started
---------------

To dump all combinations of non-transitive dice with `num_sides = 6`, `num_labels = 7`, and `num_dice = 4` ([Efron's dice](http://en.wikipedia.org/wiki/Nontransitive_dice#Efron.27s_dice)):
```
python efrons_dice.py --no-output-progress > efrons_dice.log
```

The (partial) output is:
<pre>
num_sides: 6
num_labels: 7
num_dice: 4
num_unique_dice: 924

<b>1/19> *</b>
s: 104
0: [27, 83, 195, 340]
[[0 0 0 0 6 0 0]
 [0 0 0 6 0 0 0]
 [0 0 4 0 0 0 2]
 [0 2 0 0 0 4 0]]
...
(100)

2/19> 19
s: 76
0: [71, 166, 407, 479]
[[0 0 0 3 2 0 1]
 [0 0 2 2 0 1 1]
 [0 3 0 0 0 1 2]
 [1 0 0 0 3 2 0]]
...
<b>7/19> 24</b>
s: 96
0: [83, 195, 409, 728]
[[0 0 0 6 0 0 0]
 [0 0 4 0 0 0 2]
 [0 3 0 0 0 3 0]
 [2 0 0 0 4 0 0]]
1: [195, 409, 728, 83]
[[0 0 4 0 0 0 2]
 [0 3 0 0 0 3 0]
 [2 0 0 0 4 0 0]
 [0 0 0 6 0 0 0]]
2: [409, 728, 83, 195]
[[0 3 0 0 0 3 0]
 [2 0 0 0 4 0 0]
 [0 0 0 6 0 0 0]
 [0 0 4 0 0 0 2]]
3: [728, 83, 195, 409]
<b>[[2 0 0 0 4 0 0]
 [0 0 0 6 0 0 0]
 [0 0 4 0 0 0 2]
 [0 3 0 0 0 3 0]]</b>
(4)
...
</pre>
where:
* In each block, `s` is the maximum sum of the number of times each dice beats its (next) neighbour.
The dice indices and corresponding dice matrices follow, where for each dice matrix entry `i, j` is the number of sides with label `j` on dice `i`.
* The first block `1/19> *` does *not* require each dice to beat its neighbour the same number of times; the subsequent blocks do.
* Efron's dice are found under `7/19> 24`, where each dice beats its neighbour 24 times out 36.
(Four entries are given because all configurations are equivalent under rotation.)

Run `python efrons_dice.py -h` for further usage details.
