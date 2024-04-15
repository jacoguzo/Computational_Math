#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 20:05:39 2023

@author: jacoboguzowski
"""
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from numpy import array, zeros, diag, diagflat, dot

import pandas as pd

from mpl_toolkits.mplot3d import Axes3D

def jacobi(A,b, n_max):
    #n_max is k in this case. keeps track of max amount of iterations
    #n_max = k is 50 in this example
    iter = 0 #keeps track of iteration #
    x = np.matmul((np.linalg.inv(A)),(b)) #original solution to system Ax= b
    x_initial = np.matmul((np.linalg.inv(A)),(b))#saves the true solution, to be referenced w err and res
    x_new =np.zeros(64) #will change with every subsequent iteration
    res =np.zeros(50) #initializes original residuals array
    err = np.zeros(50)#initializes original errors array
    x_0 =  np.zeros(64) #initial guess, each input of nx1 vector is "touched" differently by guess
    
   
    x_minus_xk = np.zeros((50, 64))
    for f in range(64) : 
        
        x_0[f] = x[f] + (np.sin(f * np.pi / 3) + 0.1 * np.sin(f * np.pi / 20))
        

    r_0 = np.linalg.norm(b - np.dot(A, x_0))
    

    x = np.copy(x_0)
    #print("x")
    #print(x)


    while (iter < 50):
       #computing jacobi for non diagonal entries
        #print(x[1])
        for i in range(64):
            s=0
            for j in range(i):
                s = s + A[i][j] * x[j]
                
                #print(s)
            for j in range(i+1, 64):
                s = s + A[i][j] * x[j]
            # soltn component (b[i]) - (s), divided by diagonal value corresponding to each row      
            x_new[i] = np.divide((b[i] - s),A[i][i])

            print(x_new[i])

        x = np.copy(x_new) #sets up next iteration w/ new values
        #print(x)
        print("x-x_0")
        #print(x-x_0)
     
        res[iter] =(np.linalg.norm(b - np.dot(A, x)))/ np.linalg.norm(b)#calculates residual for iteration
        #uses x_initial to refer to true solution
        err[iter] = (np.linalg.norm(x_initial- x)) /(np.linalg.norm(x_initial)) #calculates error for iteration
        
        #creates a 2D array of x- xk values, each row is an iteration, 
        #each collumn is the values corresponding to each index
        for p in range(64):
            x_minus_xk[iter][p] = (x_initial- x)[p]
        

        iter +=1
    print("err")
    print(err)
    print("res")
    print(res)
    
    print(x_minus_xk)
    return res, err,x_minus_xk
        
            
    
    
    

n = 64 # Change 'n' to the desired size of the matrix
matrix = 2 * np.eye(n)
matrix[0][0] = 1

for i in range(n):
    if i > 0:
        matrix[i, i - 1] = -1  # Element to the left of the diagonal
    if i < n - 1:
        matrix[i, i + 1] = -1  # Element to the right of the diagonal



A = matrix 
print('a')
print(A)
array = np.ones((n, 1))


b = np.array([1.0]*n)
for i in range(n):
    b[i] = n**-2

print('b')
print(b)



residuals, errors, diff_array = jacobi(A, b, 50)

# Create a plot of errors against iteration
plt.figure(figsize=(8, 6))
plt.plot(range(50), errors, marker='o', linestyle='-', color='r')
plt.xlabel('Iteration (k)')
plt.ylabel('Error Norm')
plt.title('Errors vs. Iteration (Jacobi, n =64)')
plt.grid(True)
plt.show()



# Create a plot of errors against iteration
plt.figure(figsize=(8, 6))
plt.plot(range(50), residuals, marker='o', linestyle='-', color='b')
plt.xlabel('Iteration (k)')
plt.ylabel('Residual Norm')
plt.title('Residuals vs. Iteration (Jacobi, n =64)')
plt.grid(True)
plt.show()


fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111, projection='3d')

# Create a grid of iterations and indices
iterations = np.arange(50)
indices = np.arange(64)
X, Y = np.meshgrid(iterations, indices)

# Plot the surface
surf = ax.plot_surface(X, Y, diff_array.T, cmap='viridis')

# Add labels and a color bar
ax.set_xlabel('Iteration (k)')
ax.set_ylabel('Size (n)')
ax.set_zlabel('x - xk')
plt.title('X- Xk Surface Plot (Jacobi, n =64)')
cbar = fig.colorbar(surf,shrink=0.7)


# Show the plot
plt.show()
