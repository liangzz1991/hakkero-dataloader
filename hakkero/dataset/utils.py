#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import itertools
import math
import random

import numpy as np
from scipy.stats import qmc

# Specifies a target value that is ignored and does not contribute to the input gradient
# see torch.nn.functional.cross_entropy
IGNORE_INDEX = -100


class MultinomialSampler:
    # quasi monte-carlo sampler for better coherence to the expected categorical distribution
    def __init__(self, seed=-1):
        if seed < 0:
            self.sampler = itertools.count()
        else:
            self.sampler = qmc.Sobol(d=1, seed=seed)

    def next(self, weights):
        weight_sum = weights.sum()
        if np.isclose(weight_sum, 0):
            raise StopIteration

        if isinstance(self.sampler, itertools.count):
            while True:
                i = next(self.sampler) % len(weights)
                if weights[i] > 0:
                    return i
        else:
            p_vals = weights / weight_sum

            residual = 1 - p_vals.sum()
            if residual > 0:
                p_vals[p_vals.argmin()] += residual
            else:
                p_vals[p_vals.argmax()] += residual

            return qmc.MultinomialQMC(p_vals, 1, engine=self.sampler).random(1).argmax()


def random_range(start, stop=None, step=None, seed=0):
    """Generator of non-repeated random permutation with the same interface of python `range`.

    Ref: https://stackoverflow.com/q/53551417

    The random.shuffle(list) and random.sample(list, len(list)) require materialize the lists, which result in a
    long initialization period.

    Args:
        start:
        stop:
        step:
        seed:

    Returns:

    """
    if stop is None:
        start, stop = 0, start

    if step is None:
        step = 1

    # use a mapping to convert a standard range into the desired range
    def mapping(i):
        return i * step + step

    # compute the number of numbers in this range
    maximum = int(math.ceil((stop - start) / step))

    # early return with empty range
    if maximum == 0:
        yield from ()
        return

    # seed range with a random integer
    rand = random.Random(seed)
    value = rand.randint(0, maximum)

    # Construct an offset, multiplier, and modulus for a linear
    # congruential generator. These generators are cyclic and
    # non-repeating when they maintain the properties:
    #
    #   1) "modulus" and "offset" are relatively prime.
    #   2) ["multiplier" - 1] is divisible by all prime factors of "modulus".
    #   3) ["multiplier" - 1] is divisible by 4 if "modulus" is divisible by 4.
    #
    # Pick a random odd-valued offset.
    offset = random.randint(0, maximum) * 2 + 1
    # Pick a multiplier 1 greater than a multiple of 4.
    multiplier = 4 * (maximum // 4) + 1
    # Pick a modulus just big enough to generate all numbers (power of 2).
    modulus = int(2 ** math.ceil(math.log2(maximum)))
    # Track how many random numbers have been returned.
    found = 0
    while found < maximum:
        # If this is a valid value, yield it in generator fashion.
        if value < maximum:
            found += 1
            yield mapping(value)
        # Calculate the next value in the sequence.
        value = (value * multiplier + offset) % modulus


class Range:
    def __init__(self, start, stop, step):
        self.start, self.stop, self.step = start, stop, step

    def __repr__(self):
        return f"Range({self.start}, {self.stop}, {self.step})"

    def iterate(self):
        yield from range(self.start, self.stop, self.step)

    def list(self):
        return list(range(self.start, self.stop, self.step))

    def sub_range(self, split, n_splits):
        # strided split of range
        # e.g., [0, 3, 5, 7, 9] can be split into [0, 5, 9] and [3, 7]
        return Range(self.start + self.step * split, self.stop, self.step * n_splits)

    def consecutive_sub_range(self, split, n_splits):
        # consecutively split of range
        # e.g., [0, 3, 5, 7, 9] can be split into [0, 3, 5] and [7, 9]
        start, stop = self.split2index(split, n_splits, len(self))
        return Range(self.start + start * self.step, self.start + stop * self.step, self.step)

    @staticmethod
    def split2index(split, n_splits, length):
        # get the start and stop index of a given consecutively split
        split_size = length // n_splits
        remain = length % n_splits

        if split < remain:
            start = split * (split_size + 1)
            stop = start + (split_size + 1)
        else:
            start = split * split_size + remain
            stop = start + split_size

        return start, stop

    @staticmethod
    def index2split(index, n_splits, length):
        # get the consecutively split of a given index
        split_size = length // n_splits
        remain = length % n_splits
        if index < (split_size + 1) * remain:
            split = index // (split_size + 1)
        else:
            split = remain + (index - (split_size + 1) * remain) // split_size
        return split

    @staticmethod
    def cover_split(split, n_splits, old_n_split, d_size):
        # legacy code for an abandoned feature
        # assume sequence is split consecutively with consecutive_sub_range
        # old: [1,2,3,4,5] -> (0, [1,2]), (1, [3,4]), (2, [5])
        # new: [1,2,3,4,5] -> (0, [1,2,3]), (1, [4,5])
        # we want to cover the indices of new split with that of old split
        # i.e., split 0 -> old splits [0, 1], split 1 -> old splits [2, 3]

        # map the start & stop of the current split to the old splits
        start, stop = Range.split2index(split, n_splits, d_size)
        old_split_start = Range.index2split(start, old_n_split, d_size)
        old_split_stop = Range.index2split(stop, old_split_start, d_size)
        return list(range(old_split_start, old_split_stop + 1))

    def random_iterate(self, seed=None):
        yield from random_range(self.start, self.stop, self.step, seed=seed)

    def __len__(self):
        return math.ceil((self.stop - self.start) / self.step)


class RunningAverage:
    def __init__(self, default=100):
        self._default = default
        self._value = 0
        self._count = 0

    @property
    def value(self):
        if self._count == 0:
            return self._default

        return self._value

    def reset(self):
        self._value = 0
        self._count = 0

    def update(self, x):
        if isinstance(x, (list, tuple)):
            if x:
                self._count += len(x)
                xsum = sum(x, start=0.0)
                self._value += (xsum - len(x) * self._value) / self._count
        else:
            self._count += 1
            self._value += (x - self._value) / self._count

    def __repr__(self):
        return str(self.value)


def format_size(size_bytes):
    if size_bytes == 0:
        return "0B"

    names = ("B", "KB", "MB", "GB", "TB", "PB")

    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s}{names[i]}"
