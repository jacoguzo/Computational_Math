#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 18:15:48 2023

@author: jacoboguzowski
"""


import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-poster')

# Define the number of grid points
n = 199
h = 1.0 / (n + 1)

# Initialize the coefficient matrix A and right-hand side vector b
A = np.zeros((n, n))
b = np.zeros(n)

# Fill in the matrix A with the finite difference coefficients
for i in range(n):
    if i > 0:
        A[i, i-1] = (-1/h**2) - (1/(2*h))
    A[i, i] = (2/h**2) + 2
    if i < n - 1:
        A[i, i+1] = (-1/h**2) + (1/(2*h))

        
print(A)        

# Adjust the right-hand side vector b
x = np.linspace(h, 1 - h, n)  # internal points only
b = (1 - x)#

# Solve the linear equations
u = np.linalg.solve(A, b)

# Create a vector of grid points including the boundaries
t = np.linspace(0, 1, n + 2)



# Insert boundary conditions into the solution
u_full = np.insert(u, 0, 0)  # u(0) = 0
u_full = np.append(u_full, 0)  # u(1) = 0



#analytic solution

a = (np.exp(1) - 3*np.exp(3)) / (4*(np.exp(3) - 1))
b = (3 - np.exp(1)) / (4*(np.exp(3) - 1))

# True solution function
def u_true_function(x):
    u_g = a * np.exp(-x) + b * np.exp(2*x)
    u_p = 0.5 * ((1.5) - x)
  
    return u_g + u_p





# Plot the numerical solution
plt.figure(figsize=(10, 8))
plt.plot(t, u_full, label='Numerical Solution')
plt.plot(t, u_true_function(t), label='Analytic Solution',  linestyle='-.')
plt.plot(t,(u_full-u_true_function(t)), label="Error")
plt.xlabel('x')
plt.ylabel('u(x)')
plt.legend()
plt.show()









