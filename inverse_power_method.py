import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

def normalize(x):
    fac = np.linalg.norm(x)
    x_n = x / fac
    return fac, x_n

x = np.array([5, 4, 5])
a = np.array([[6, 2, 2], 
              [2, 8, -3],
              [2, -3, 12]])

lambda_previous = 0
lambda_1 = 0
eigenvalues = []

iteration = 0

a_inv = inv(a)  # Compute the inverse of a

while True:
    x = np.matmul(a_inv, x)
    lambda_previous = lambda_1
    lambda_1, x = normalize(x)
    
    if iteration >= 0:  # Avoid division by zero in the first iteration
        eigenvalues.append(1 / lambda_1)  # Store the reciprocal of lambda_1

    if iteration > 0 and abs(1 / lambda_1 - 1 / lambda_previous) < 0.005:
        break
    
    iteration += 1


print(eigenvalues)

from matplotlib.ticker import MultipleLocator

# ...

# Plotting
plt.plot(range(1, iteration + 1), eigenvalues[1:], marker='o')  # Start plotting from the second iteration
plt.xlabel('Iteration')
plt.ylabel('Minimum Eigenvalue')
plt.title('Convergence of the Inverse Power Method')
plt.grid(True)

# Set the tick interval to 1 for both axes
plt.gca().xaxis.set_major_locator(MultipleLocator(1))
plt.gca().yaxis.set_major_locator(MultipleLocator(1))

# Show the plot with the new tick marks
plt.show()
