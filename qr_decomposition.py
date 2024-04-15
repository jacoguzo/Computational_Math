#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 18:49:13 2023

@author: jacoboguzowski
"""

import matplotlib.pyplot as plt
import numpy as np

def qr_iteration_with_diagonals(A, num_iterations=10):
    diagonals = [np.diag(A)]  # Include initial diagonal as the first element
    for _ in range(num_iterations):
        Q, R = np.linalg.qr(A)
        A = np.matmul(R,Q)
        diagonals.append(np.diag(A))
    return diagonals

# Example square matrix
A = np.array([[6, 2, 2], [2, 8, -3], [2, -3, 12]])

# Perform QR Iteration and record diagonals, including the initial diagonal
diagonals_each_iteration = qr_iteration_with_diagonals(A, num_iterations=10)

# Plotting
plt.figure(figsize=(10, 6))
for i in range(A.shape[0]):  # Loop over each diagonal entry
    plt.plot(range(len(diagonals_each_iteration)), [diag[i] for diag in diagonals_each_iteration], label=f"Diagonal {i+1}")
plt.xlabel('Iteration')
plt.ylabel('Diagonal Entry Value')
plt.title('Convergence of Diagonal Entries in QR Algorithm (Including Initial Diagonal)')
plt.legend()
plt.grid(True)
plt.show()
