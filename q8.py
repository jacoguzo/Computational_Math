#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 14:31:08 2023

@author: jacoboguzowski
"""

import numpy as np

N = 1000
A = np.zeros((N, N))  # Correct function is np.zeros
h = 1 / (N + 1)

def f(x):
    # Handle the case when x is zero to avoid division by zero

    return 1 / (h**2 * (1 + 1/5 * np.sin(2 * np.pi * x * h)))
A[0, 0] = 2 * f(0)
A[N-1, N-1] = 2 * f(N-1)
for i in range(N):
    A[i, i] = 2 * f(i+1)
    if i + 1 < N:  # Ensure we don't go out of bounds
        A[i, i+1] = -1 * f(i+1)
    if i - 1 >= 0:  # Ensure we don't access negative index incorrectly
        A[i, i-1] = -1 * f(i+1)

print(A)

#from 6
import matplotlib.pyplot as plt
from numpy.linalg import inv

def normalize(x):
    fac = np.linalg.norm(x)
    x_n = x / fac
    return fac, x_n

x = np.random.rand(N)
x = x / np.linalg.norm(x)
a = A

lambda_previous = 0
lambda_1 = 1
eigenvalues = []

iteration = 0

a_inv = inv(a)  # Compute the inverse of a

while True:
    x = np.matmul(a_inv, x)
    lambda_previous = lambda_1
    lambda_1, x = normalize(x)
    
    if iteration >= 0:  # Avoid division by zero in the first iteration
        eigenvalues.append(1 / lambda_1)  # Store the reciprocal of lambda_1

    if iteration > 0 and abs(1 / lambda_1 - 1 / lambda_previous) < 0.0000005:
        break
    
    iteration += 1


print(eigenvalues)

from matplotlib.ticker import MultipleLocator

# ...

# Plotting
plt.plot(range(1, iteration + 1), eigenvalues[1:], marker='o')  # Start plotting from the second iteration
plt.xlabel('Iteration')
plt.ylabel('Minimum Eigenvalue')
plt.title('Convergence of the Inverse Power Method')
plt.grid(True)

plt.show()
plt.plot(x, marker='o')  # Start plotting from the second iteration
plt.xlabel('t')
plt.ylabel('Eigenfunction')
plt.title('Eigenfunction as a function of t')
plt.grid(True)



# Show the plot with the new tick marks

# Show the plot with the new tick marks
plt.show()
