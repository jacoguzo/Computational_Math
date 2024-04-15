#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 19:54:19 2023

@author: jacoboguzowski
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig
from scipy.sparse import diags

# Parameters
N = 100  # Number of internal points
q = 1.5  # An example value for q
h = 1 / (N + 1)  # Step size

# Discretize the interval (0, 1)
t = np.linspace(h, 1 - h, N)

# Construct the matrices A and B for the finite difference method
k = np.ones(N)
A = diags([-k, 2*k, -k], [-1, 0, 1], shape=(N, N)).toarray() / h**2
B = np.diag(1 + (1 / (q + 2)) * np.sin(2 * np.pi * t))

# Solve the generalized eigenvalue problem
eigenvalues, eigenvectors = eig(A, B)

# Since we want the smallest eigenvalue, we take the real part and sort them
# and then take the first one (smallest)
idx = np.argsort(np.real(eigenvalues))
smallest_eigenvalue = np.real(eigenvalues[idx[0]])
eigenfunction = np.real(eigenvectors[:, idx[0]])

# Normalize the eigenfunction
eigenfunction = eigenfunction / np.linalg.norm(eigenfunction)

# Plot the eigenfunction
plt.plot(t, eigenfunction, label=f'Eigenvalue: {smallest_eigenvalue:.3f}')
plt.title('Eigenfunction corresponding to the smallest eigenvalue')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.legend()
plt.grid(True)
plt.show()

# Output the smallest eigenvalue
smallest_eigenvalue
