#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 15:21:22 2023

@author: jacoboguzowski
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 15:04:45 2023

@author: jacoboguzowski
"""
from sympy import *
x = symbols('x')  # Define the variable
import math
import numpy as np
import matplotlib.pyplot as plt



"""
1: Set iter = 0, x = x0, err 1 = 2ε1, err 2 = 2ε2
2: while err 1 > ε1 & err 2 > ε2 & iter ¡nmax do
3: xnew = x − f (x)/f ′(x)
4: err 1 = |xnew − x|
5: x = xnew
6: err 2 = |f (x)|
7: iter=iter+1
8: end while
"""

#changing f
f = x**2+x- sin(x)

#5i
x_vector = np.zeros(20)
b_vector = np.zeros(19)


def Newtons(n_max, x0):
        
    iter = 0
    x_curr = x0
    x_initial = x0
    err_1 = 2*e_1
    err_2 = 2*e_2
    while iter != n_max:
        
        f_prime =diff(f, x)

        
        
        
        x_new = (x_curr - f.subs(x, x_curr)/f_prime.subs(x, x_curr)).evalf() #newtons method formula
        #x_new = (x_curr - 2*f.subs(x, x_curr)/f_prime.subs(x, x_curr)).evalf() #newtons method formula
        
        #err_1 = abs(x_new - x_curr)
        x_initial = x_curr #stepping up x_initial
        x_curr = x_new #stepping up x_curr
    
        #err_2 = abs(f.subs(x, x_curr))
        x_vector[iter] = x_curr
        
        
        
        
        
        #addition for part 2, taking x_5 = 0.567143290409784
        if iter >0 and iter <(n_max) : 
           
            x_5 = 0.567143290409784
            
            beta_k = log(abs(x_curr),abs(x_initial))
            print("Bk = ", beta_k)
            b_vector[iter-1] = beta_k
    
        
       
        iter = iter +1
        print("iteration number: ", iter)
        print("current guess: ")
        print(x_curr)
        print("")
    return x_curr, x_vector, b_vector
        
# Example usage:
n_max = 100  # Maximum number of iterations
x0 = 1.0  # Initial guess
e_1 = 1e-6  # Tolerance for |x_new - x_curr|
e_2 = 1e-6  # Tolerance for |f(x)|


#5i
result, roots, beta = Newtons(20,1)
#5ii
#result, roots, beta = Newtons(4,2.5)
print("Approximate root:", result)


#creating plots

#plotting roots v iteeration
plt.figure(figsize=(8, 6))


#plt.plot(range(1,21), roots, marker='o', linestyle='-', color='r')
plt.plot(range(1,21), roots, marker='o', linestyle='-', color='r')

plt.xlabel('Iteration (k)')
plt.ylabel('Root estimate')
plt.title('Root estimate vs. Iteration')
plt.grid(True)
plt.show()

#plotting beta v iteration
plt.figure(figsize=(8, 6))


#plt.plot(range(19), beta, marker='o', linestyle='-', color='b')

plt.plot(range(2,21), beta, marker='o', linestyle='-', color='b')

plt.xlabel('Iteration (k)')
plt.ylabel('Beta_k')
plt.title('Beta vs. Iteration')
plt.grid(True)
plt.show()



















