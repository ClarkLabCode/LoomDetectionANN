#!/usr/bin/env python
# coding: utf-8

import train_multiple_units_debug as tmud

M = 256
seed_left = 1 # left limit of the seed range
seed_right = 2 # right limit of the seed range
n_cores = 2
tmud.train_multiple_units_f(M, seed_left, seed_right, n_cores)  