#!/usr/bin/env python
# coding: utf-8

import train_multiple_units as tmu

M = 32
seed_left = 1 # left limit of the seed range
seed_right = 100 # right limit of the seed range
n_cores = 34
tmu.train_multiple_units_f(M, seed_left, seed_right, n_cores)  