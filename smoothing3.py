import numpy as np
import matplotlib.pyplot as plt
# Correct the matrix definition
A = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0],
    [-4.9, 10.8, -4.9, 0, 0, 0, 0, 0],
    [0, -4.9, 10.8, -4.9, 0, 0, 0, 0],
    [0, 0, -4.9, 10.8, -4.9, 0, 0, 0],
    [0, 0, 0, -4.9, 10.8, -4.9, 0, 0],
    [0, 0, 0, 0, -4.9, 10.8, -4.9, 0],
    [0, 0, 0, 0, 0, -4.9, 10.8, -4.9],
    [0, 0, 0, 0, 0, 0, 0, 1]
])

# Get the inverse of the matrix
A_inv = np.linalg.inv(A)

# Your vector of scores
B = np.array([1, 1, 1, 0.95, 1, 0.975, 17/19, 0.9])

# Solve the system Ax = B to find the vector x
u = np.dot(A_inv, B)

# Print the results
print("The inverse of matrix A is:")
print(A_inv)
print("The vector B is:")
print(B)
print("The solution vector u is:")
print(u)

# Plot the original scores (vector B) and the smoothed values (vector u)
plt.figure(figsize=(10, 5))
plt.plot(range(1,9),B, 'o-', label='Original Scores (f)')
plt.plot(range(1,9),u, 's-', label='Smoothed Scores (u)')
plt.xlabel('Index of Score')
plt.ylabel('Score Value')
plt.title('Comparison of Original Scores (B) and Smoothed Scores (u)')
plt.legend()
plt.grid(True)
plt.show()
