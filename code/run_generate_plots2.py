# Imports
import numpy as np
import matplotlib.pyplot as plt
import _pickle as cPickle
import os
from itertools import product

# PyTorch
import torch



with open('stat_data.pickle', "rb") as s:
    stat_data = cPickle.load(s)

x = []
y = []

for gen in range(stat_data.shape[1]):
    for p_ind in range(stat_data.shape[2]):
        x.append(gen)
        y.append(stat_data[1][gen][p_ind][0])

plt.plot(x, y, 'bo')
plt.show()

