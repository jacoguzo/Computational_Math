import numpy as np
import matplotlib.pyplot as plt

def normalize(x):
    # Normalize the vector x by its maximum absolute value.
    fac = np.linalg.norm(x)
    x_n = x / fac
    return fac, x_n

# Initial guess for the eigenvector.
x = np.array([5, 4, 5])

# Matrix for which we are finding the eigenvalue and eigenvector.
a = np.array( [[0,0,1],[0.5,0,0],[0.5,1,0]])

# Initialize variables to store the previous and current eigenvalue estimates.
lambda_previous = 0
lambda_1 = 0

# List to store the eigenvalues from each iteration for plotting.
eigenvalues = []

# Counter for the number of iterations.
iteration = 0

# Begin the power method loop.
while True:
    # Multiply the current vector x by the matrix a.
    x = np.dot(a, x)

    # Store the previous eigenvalue and compute the new one.
    lambda_previous = lambda_1
    lambda_1, x = normalize(x)

    # Append the current eigenvalue to the list.
    eigenvalues.append(lambda_1)

    # Check if the difference between the current and previous eigenvalues is small enough to break the loop.
    if abs(lambda_1 - lambda_previous) < 0.005:
        break
    
    # Increment the iteration counter.
    iteration += 1

# Output the final eigenvalue and eigenvector.
print('Eigenvalue:', lambda_1)
print('Eigenvector:', x)
print(eigenvalues)
# Plotting the eigenvalues over iterations.
plt.plot(range(1, iteration + 1), eigenvalues[1:], marker='o')
plt.xlabel('Iteration')
plt.ylabel('Maximum Eigenvalue')
plt.title('Convergence of the Power Method')
plt.grid(True)
plt.show()
