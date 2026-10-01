"""This file contains all static Data variables used for the experiments"""
import numpy as np


DEFAULT_SEED = 42

S1 = np.array([1, 1, -1, -1, -1, 1, -1, 1], dtype=float)
S2 = np.array([1, -1, 1, -1, 1, -1, -1, -1], dtype=float)
SK = np.array([S1, S2], dtype=float) 