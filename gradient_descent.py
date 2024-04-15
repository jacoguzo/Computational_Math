import matplotlib.pyplot as plt

import numpy as np

# Redefine the function and gradient
def f(x, y):
    return (4-2.1*x**2 + x**4/3) * x**2 + x*y + 4*(y**2-1)*y**2

def gradient(x, y):
    return np.array([8*x-8.4*x**3+2*x**5+y, x+16*y**3-8*y])

# Gradient descent parameters
alpha = 0.01
iterations = 100
epsilon = 1e-6

# Initial guess
x, y = 0.375, 0

# Arrays to store iteration data
x_values, y_values, f_values = [], [], []

# Gradient descent algorithm with data recording
for i in range(iterations):
    grad = gradient(x, y)
    new_x = x - alpha * grad[0]
    new_y = y - alpha * grad[1]

    # Store iteration data
    x_values.append(new_x)
    y_values.append(new_y)
    f_values.append(f(new_x, new_y))
    
    if np.sqrt((f(new_x, new_y)- (-1.0316))**2) < epsilon:
        break

    # Update x and y
    x, y = new_x, new_y
    
print(x,y)  
print(f(x,y))  

# Plotting
plt.figure(figsize=(12, 5))

# Plot for (x, y) sequence
plt.subplot(1, 2, 1)
plt.plot(x_values, y_values, 'o-', color='blue')
plt.title('Sequence of points (x_n, y_n)')
plt.xlabel('x')
plt.ylabel('y')

# Plot for function values
plt.subplot(1, 2, 2)
plt.plot(f_values, 'o-', color='red')
plt.title('Function values f(x_n, y_n)')
plt.xlabel('Iteration')
plt.ylabel('Function value')

plt.tight_layout()
plt.show()
