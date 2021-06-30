#!/usr/bin/env python
# coding: utf-8

import train_multiple_units as tmu

M = 2
seed_left = 101 # left limit of the seed range
seed_right = 200 # right limit of the seed range
n_cores = 30
tmu.train_multiple_units_f(M, seed_left, seed_right, n_cores)  