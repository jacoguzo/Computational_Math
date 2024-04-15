#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 18:26:16 2023

@author: jacoboguzowski
"""


def gauss_seidel(n_max, e, x_0):
    
    iter = 0
    r_0 = abs(b - A*(x_0))
    relerr = 2*e 
    x = x_0
    
    while (relerr > e and iter < n_max):
        for i in range(1,n):
            s=0
            for j in range(1,i-1):
                s = s + A[i][j]*x_new[j]
            for j in range(1,n):
                s = s + A[i][j]*x[j]
            x_new[i] = (b[i]-s)/A[i][j]
        x = x_new
        relerr = abs(b-Ax)/r_0
        iter +=1