#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 22:08:05 2023

@author: jacoboguzowski
"""

import numpy as np
import matplotlib.pyplot as plt

# Function definitions
def f(t, y):
    return -75 * y

def y_true(t, y0):
    return y0 * np.exp(-75 * t)

def backward_euler(f, a, b, y0, h):
    t = a
    y = y0
    Ts = [t]
    Ys = [y]

    while t <= b:
        t += h
        y = y / (1 + 75 * h)  # Backward Euler update
        Ts.append(t)
        Ys.append(y)

    return Ts, Ys

# Parameters
a = 0   # start of interval
b = 1 # end of interval
y0 = 1  # initial value
h1 = 0.01 # step size
h2 = 0.02 # step size

# Calculate solutions
Ts1, Ys1 = backward_euler(f, a, b, y0, h1)
Ts2, Ys2 = backward_euler(f, a, b, y0, h2)
T_true = np.linspace(a, b, 100)
Y_true = y_true(T_true, y0)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(Ts1, Ys1, 'b.-', label='Backward Euler Approximation')
plt.plot(Ts2, Ys2, 'g.-', label='Backward Euler Approximation')
plt.plot(T_true, Y_true, 'r-', label='True Solution')
plt.title('Comparison of Backward Euler Method and True Solution')
plt.xlabel('t')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
