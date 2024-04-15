import numpy as np
import matplotlib.pyplot as plt

def normalize(x):
    fac = abs(x).max()
    x_n = x / fac
    return fac, x_n

def max_eigenvalue(a, x, iterations):
    lambda_1 = 0
    for i in range(iterations):
        x = np.dot(a, x)
        lambda_1, x = normalize(x)
    return lambda_1, x

# Initial vector and matrix
x = np.array([0, 1, 0])
a = np.array([[1, -1, 0], 
              [-1, 2, -1],
              [0, -1, 2]])

# Compute eigenvalues over iterations
iterations = 10
eigenvalues = []
for i in range(iterations):
    lambda_1, x = max_eigenvalue(a, x, 1)
    eigenvalues.append(lambda_1)

# Plotting
plt.figure(figsize=(10, 8))
plt.plot(range(iterations), eigenvalues, label='Max Eigenvalue')
plt.xlabel('Iteration')
plt.ylabel('Eigenvalue')
plt.legend()
plt.show()

print(eigenvalues)
