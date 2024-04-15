#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 14:01:45 2023

@author: jacoboguzowski
"""

from numpy import linalg as LA
import numpy as np
import scipy 


#1a
A = np.array([[1,0,1],
              [-1,1,0],
              [-1,-2,3]])

A1_j = np.array([[1,0,0],
                 [0,1,0],
                 [0,0,3]])


A1_gs = np.array([[1,0,0],
                  [-1,1,0],
                  [-1,-2,3]])

A2_j = A - A1_j
print(A2_j)
A2_gs = A - A1_gs
print(A2_gs)


inverse1 = np.linalg.inv(A1_j)
print(inverse1)
B_j = np.matmul(-inverse1,A2_j)


inverse2 = np.linalg.inv(A1_gs)
print(inverse2)
B_gs = np.matmul(-inverse2,A2_gs)


print("jacobi case:")
print("")
print(B_j)
eigenvalues_j, x = LA.eig(B_j)
print(eigenvalues_j)

print("GS case:")
print("")
print(B_gs)
eigenvalues_gs, y = LA.eig(B_gs)
print(eigenvalues_gs)

print("")
print("")
print("")

#1b

A = np.array([[1,0.5,0.5],
              [0.5,1,0.5],
              [0.5,0.5,1]])

A1_j = np.array([[1,0,0],
                 [0,1,0],
                 [0,0,1]])


A1_gs = np.array([[1,0,0],
                  [0.5,1,0],
                  [0.5,0.5,1]])

A2_j = A - A1_j
print(A2_j)
A2_gs = A - A1_gs
print(A2_gs)


inverse1 = np.linalg.inv(A1_j)
print(inverse1)
B_j = np.matmul(-inverse1,A2_j)


inverse2 = np.linalg.inv(A1_gs)
print(inverse2)
B_gs = np.matmul(-inverse2,A2_gs)


print("jacobi case:")
print("")
print(B_j)
eigenvalues_j, x = LA.eig(B_j)
print(eigenvalues_j)

print("GS case:")
print("")
print(B_gs)
eigenvalues_gs, y = LA.eig(B_gs)
print(eigenvalues_gs)


print("")
print("")
print("")

#2c
import numpy as np
spectral_rad= 1
optimal_omega=np.nan
for omega in np.linspace(1,2,1000): #takes in interval and how many split ups to do
    B = np.matrix([[(1-omega),(omega/2)],[(omega - omega**2)/2,(1- omega +((omega**2)/4))]])#construct matrix
    spectral_rad_updated= np.max(np.abs(np.linalg.eigvals(B)))
    if spectral_rad_updated < spectral_rad:
        spectral_rad = spectral_rad_updated
        optimal_omega = omega
print("approx optimal omega is ", optimal_omega, "spectral radius is",spectral_rad )
        
    
    
