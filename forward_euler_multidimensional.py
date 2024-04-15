import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function definitions

def forward_euler_system(x_prime, y_prime, z_prime, x0, y0, z0, t0, tf, h):
    
    """
    Solve the system of ODEs using the classical 4-stage Runge-Kutta method.

    Parameters:
    x_prime, y_prime, z_prime: functions
        The derivative functions for x, y, z respectively. Each must take three arguments (x, y, z).
    x0, y0, z0: float
        Initial conditions for x, y, z.
    t0, tf: float
        Initial and final time.
    h: float
        The step size.

    Returns:
    t: array
        Array of time points where the solution was computed.
    x, y, z: arrays
        Arrays containing the solutions corresponding to each time point in t.
    """
    t = np.arange(t0, tf+h, h)  # Time array
    x = np.zeros(len(t))
    y = np.zeros(len(t))
    z = np.zeros(len(t))

    x[0], y[0], z[0] = x0, y0, z0

    for i in range(len(t)-1):
        x[i+1] = x[i] + h * x_prime(x[i], y[i], z[i])
        y[i+1] = y[i] + h * y_prime(x[i], y[i], z[i])
        z[i+1] = z[i] + h * z_prime(x[i], y[i], z[i])

    return t, x, y, z

# Definitions of the ODEs
def x_prime(x, y, z):
    return 10 * (y - x)

def y_prime(x, y, z):
    return (28 - z) * x - y

def z_prime(x, y, z):
    return x * y - (8/3) * z

# Initial conditions and time span
x0, y0, z0 = -4, -8, 3
t0, tf = 0, 100  # Start and end time
h = 0.01  # Step size

# Solve the ODE using Forward Euler
t, x, y, z = forward_euler_system(x_prime, y_prime, z_prime, x0, y0, z0, t0, tf, h)

# 3D plot for the trajectory
plt.figure(figsize=(10, 8))
ax = plt.axes(projection='3d')
ax.plot3D(x, y, z, 'red')
ax.set_title('3D Trajectory using Forward Euler')
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
plt.show()





# Initial conditions and time span
x0, y0, z0 = 4, 3, 8
t0, tf = 0, 100  # Start and end time
h = 0.01  # Step size

# Solve the ODE using Forward Euler
t, x, y, z = forward_euler_system(x_prime, y_prime, z_prime, x0, y0, z0, t0, tf, h)

# 3D plot for the trajectory
plt.figure(figsize=(10, 8))
ax = plt.axes(projection='3d')
ax.plot3D(x, y, z, 'green')
ax.set_title('3D Trajectory using Forward Euler')
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
plt.show()



