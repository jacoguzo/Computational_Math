#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 15:51:08 2023

@author: jacoboguzowski
"""

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
    iter = 0 #keeps track of iteration #
    x = np.matmul((np.linalg.inv(A)),(b)) #original solution to system Ax= b
    x_initial = np.matmul((np.linalg.inv(A)),(b))
    x_new =np.zeros(128)
    res =np.zeros(50) #initializes original residuals array
    err = np.zeros(50)#initializes original errors array
    x_0 =  np.zeros(128) #initial guess, each input of nx1 vector is "touched" differently by guess
    
   
    x_minus_xk = np.zeros((50, 128))
    for f in range(128) : 
        
        x_0[f] = x[f] + (np.sin(f * np.pi / 3) + 0.1 * np.sin(f * np.pi / 20))
        

    r_0 = np.linalg.norm(b - np.dot(A, x_0))
    

    x = np.copy(x_0)
    #print("x")
    #print(x)


    while (iter < 50):
       #computing jacobi for non diagonal entries
        #print(x[1])
        for i in range(128):
            s=0
            for j in range(i):
                s = s + A[i][j] * x[j]
                
                #print(s)
            for j in range(i+1, 128):
                s = s + A[i][j] * x[j]
                
            x_new[i] = np.divide((b[i] - s),A[i][i])

            print(x_new[i])

        x = np.copy(x_new) #sets up next iteration w/ new values
        #print(x)
        print("x-x_0")
        #print(x-x_0)
     
        res[iter] =(np.linalg.norm(b - np.dot(A, x)))/ np.linalg.norm(b)#calculates residual for iteration
        #print(res[iter])
        err[iter] = (np.linalg.norm(x_initial- x)) /(np.linalg.norm(x_initial)) #calculates error for iteration
        
        
        for p in range(128):
            x_minus_xk[iter][p] = (x_initial- x)[p]
        

        iter +=1
    print("err")
    print(err)
    print("res")
    print(res)
    
    print(x_minus_xk)
    return res, err,x_minus_xk
        
            
    
    
    

n = 128 # Change 'n' to the desired size of the matrix
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
plt.plot(range(50), errors, marker='o', linestyle='-', color='pink')
plt.xlabel('Iteration (k)')
plt.ylabel('Error Norm')
plt.title('Errors vs. Iteration (Jacobi, n =128)')
plt.grid(True)
plt.show()



# Create a plot of errors against iteration
plt.figure(figsize=(8, 6))
plt.plot(range(50), residuals, marker='o', linestyle='-', color='orange')
plt.xlabel('Iteration (k)')
plt.ylabel('Residual Norm')
plt.title('Residuals vs. Iteration (Jacobi, n =128)')
plt.grid(True)
plt.show()


fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111, projection='3d')

# Create a grid of iterations and indices
iterations = np.arange(50)
indices = np.arange(128)
X, Y = np.meshgrid(iterations, indices)

# Plot the surface
surf = ax.plot_surface(X, Y, diff_array.T, cmap='plasma')

# Add labels and a color bar
plt.title('X- Xk Surface Plot ((Jacobi, n =128)')
ax.set_xlabel('Iteration (k)')
ax.set_ylabel('size (n)')
ax.set_zlabel('x - xk')
cbar = fig.colorbar(surf,shrink=0.7)


# Show the plot
plt.show()
