#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 18:22:59 2023

@author: jacoboguzowski
"""

import numpy as np
import numpy as np

a = 0
b = 1
n = 4
h = (b - a) / n
x = np.linspace(a, b, n+1)  # n+1 points
f = np.exp(-x**2)
I_trap = h/2 * (f[0] + 2 * sum(f[1:-1]) + f[-1])  # Correct formula

print(I_trap)



a = 0
b = 1
n = 8
h = (b - a) / n
x = np.linspace(a, b, n+1)  # n+1 points
f = np.exp(-x**2)
I_trap = h/2 * (f[0] + 2 * sum(f[1:-1]) + f[-1])  # Correct formula

print(I_trap)