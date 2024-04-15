#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 15:04:45 2023

@author: jacoboguzowski
"""
#from sympy import symbols, diff, exp, log
from sympy import *
x = symbols('x')  # Define the variable
import math
"""
Pseudocode
1: Set iter = 0, x = x0, err 1 = 2ε1, err 2 = 2ε2
2: while err 1 > ε1 & err 2 > ε2 & iter ¡nmax do
3: xnew = x − f (x)/f ′(x)
4: err 1 = |xnew − x|
5: x = xnew
6: err 2 = |f (x)|
7: iter=iter+1
8: end while
"""
f = x - exp(-x)
def Newtons(n_max, x0):
        
    iter = 0
    x_curr = x0
    x_initial = x0
    err_1 = 2*e_1
    err_2 = 2*e_2
    while iter != n_max:
        
        f_prime =diff(f, x)

        x_new = x_curr - f.subs(x, x_curr)/f_prime.subs(x, x_curr) #newtons method formula
        err_1 = abs(x_new - x_curr)
        x_initial = x_curr #stepping up x_initial
        x_curr = x_new #stepping up x_curr
    
        err_2 = abs(f.subs(x, x_curr))
           
        #addition for part 2, taking x_5 = 0.567143290409784
        if iter >=0 and iter <4 : 
           
            x_5 = 0.567143290409784
            
            beta_k = log(abs(x_curr - x_5),abs(x_initial - x_5))
            print("Bk = ", beta_k)

        iter = iter +1
        print("iteration number: ", iter)
        print("current guess: ")
        print(x_curr)
        print("")
    return x_curr+

#4ii
result = Newtons(5, 1.25)
print("Approximate root:", result)
