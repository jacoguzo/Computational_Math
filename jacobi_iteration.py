#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 18:09:48 2023

@author: jacoboguzowski
"""
"""
from pprint import pprint
from numpy import array, zeros, diag, diagflat, dot


import numpy as np
    
    

def jacobi(A,b,N=25,x=None):
    #Solves the equation Ax=b via the Jacobi iterative method.
    # Create an initial guess if needed                                                                                                                                                            
    if x is None:
        x = zeros(len(A[0]))

    # Create a vector of the diagonal elements of A                                                                                                                                                
    # and subtract them from A                                                                                                                                                                     
    D = diag(A)
    R = A - diagflat(D)

    # Iterate for N times                                                                                                                                                                          
    for i in range(N):
        x = (b - dot(R,x)) / D
    return x

A = array([[2.0,1.0],[5.0,7.0]])
b = array([11.0,13.0])
guess = array([1.0,1.0])

sol = jacobi(A,b,N=25,x=guess)

print("A:")
print(A)

print("b:")
print(b)

print("x:")
print(sol)
    

"""
import numpy as np
import matplotlib.pyplot as plt
def jacobi(A, b, n):
    
    x = (np.linalg.inv(A)).dot(b)
    print('x')
    print(x)
    x_new = np.array([1.0] * n)
    e = 0.0001
    x_0 = np.array([0.0] * n)

    for f in range(len(x)) : 
        
        x_0[f] = x[f] + (np.sin(f * np.pi / 3) + 0.1 * np.sin(f * np.pi / 20))

    print('x_0')
    print(x_0)
    r_0 = np.linalg.norm(b - np.dot(A, x_0))
    relerr = 2 * e
    
    res = np.zeros(n)  # Array to store the residual ratios
    err = np.zeros(n)  # Array to store the error ratios
    
    iter = 0

    while (relerr > e and iter < n):
        for i in range(1, n):
            s = 0
            for j in range(1, i - 1):
                s = s + A[i][j] * x[j]
            for j in range(i, n):
                s = s + A[i][j] * x[j]
                
            x_new[i] = (b[i] - s) / A[i][i]
        x = x_new
        
        relerr = np.linalg.norm(b - A.dot(x)) / r_0
        
        res[iter] =(np.linalg.norm(b - np.dot(A, x)))/ np.linalg.norm(b)
        err[iter] = np.linalg.norm(x - x_0) / np.linalg.norm(x)
    
        iter +=1
        
        

    
    return res, err

        
        
n = 64 # Change 'n' to the desired size of the matrix
matrix = 2 * np.eye(n)
matrix[0][0] = 1

for i in range(n):
    if i > 0:
        matrix[i, i - 1] = -1  # Element to the left of the diagonal
    if i < n - 1:
        matrix[i, i + 1] = -1  # Element to the right of the diagonal



A = matrix 

print(A)
array = np.ones((n, 1))


b = np.array([1.0]*n)
for i in range(n):
    b[i] = n**-2

print(b)

print(jacobi(A,b,64))





residuals, errors = jacobi(A, b, 64)

# Create a plot of errors against iteration
plt.figure(figsize=(8, 6))
plt.plot(range(64), errors, marker='o', linestyle='-', color='r')
plt.xlabel('Iteration (k)')
plt.ylabel('Error Norm')
plt.title('Errors vs. Iteration')
plt.grid(True)
plt.show()



# Create a plot of errors against iteration
plt.figure(figsize=(8, 6))
plt.plot(range(64), residuals, marker='o', linestyle='-', color='b')
plt.xlabel('Iteration (k)')
plt.ylabel('Residual Norm')
plt.title('Residuals vs. Iteration')
plt.grid(True)
plt.show()

"""


def jacobi(A, b, n_max, e):
    
    iter = 0
    x = (np.linalg.inv(A)).dot(b)
    
    x_0 = np.array([1.0]*n)
    for j in range(len(x)): 
        x_0[j] = x[j] +(np.sin(j * np.pi / 3) + 0.1 * np.sin(j * np.pi / 20))

    print('x_0')
    print(x_0)
    r_0 = np.linalg.norm(b - np.dot(A, x_0))
    relerr = 2 * e

    
    while (relerr > e and iter < n_max):
        x_new = np.array([0.0]*n)
        for i in range(n):
            s = 0
            for j in range(n):
                if i != j:
                    s += A[i][j] * x[j]
            x_new[i] = (b[i] - s) / A[i][i]
        
        x = x_new
        relerr = np.linalg.norm(b - np.dot(A, x)) / r_0
        iter += 1
    
    return 
"""






    