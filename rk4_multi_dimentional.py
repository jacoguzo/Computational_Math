#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 15:41:28 2023

@author: jacoboguzowski
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 12})

# Define the classical 4-stage Runge-Kutta method for a system of ODEs
def runge_kutta_4(x_prime, y_prime, z_prime, x0, y0, z0, t0, tf, h):
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
    # Calculate the number of steps (ensure it's an integer)
    n = int((tf - t0) / h)

    # Initialize arrays for time points and solution values
    t = np.linspace(t0, tf, n + 1)
    x = np.zeros(n + 1)
    y = np.zeros(n + 1)
    z = np.zeros(n + 1)

    # Set initial conditions
    x[0], y[0], z[0] = x0, y0, z0

    # Runge-Kutta 4th order method loop
    for i in range(n):
        k1_x = x_prime(x[i], y[i], z[i])
        k1_y = y_prime(x[i], y[i], z[i])
        k1_z = z_prime(x[i], y[i], z[i])

        k2_x = x_prime(x[i] + h/2 * k1_x, y[i] + h/2 * k1_y, z[i] + h/2 * k1_z)
        k2_y = y_prime(x[i] + h/2 * k1_x, y[i] + h/2 * k1_y, z[i] + h/2 * k1_z)
        k2_z = z_prime(x[i] + h/2 * k1_x, y[i] + h/2 * k1_y, z[i] + h/2 * k1_z)

        k3_x = x_prime(x[i] + h/2 * k2_x, y[i] + h/2 * k2_y, z[i] + h/2 * k2_z)
        k3_y = y_prime(x[i] + h/2 * k2_x, y[i] + h/2 * k2_y, z[i] + h/2 * k2_z)
        k3_z = z_prime(x[i] + h/2 * k2_x, y[i] + h/2 * k2_y, z[i] + h/2 * k2_z)

        k4_x = x_prime(x[i] + h * k3_x, y[i] + h * k3_y, z[i] + h * k3_z)
        k4_y = y_prime(x[i] + h * k3_x, y[i] + h * k3_y, z[i] + h * k3_z)
        k4_z = z_prime(x[i] + h * k3_x, y[i] + h * k3_y, z[i] + h * k3_z)

        x[i + 1] = x[i] + (h / 6) * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)
        y[i + 1] = y[i] + (h / 6) * (k1_y + 2 * k2_y + 2 * k3_y + k4_y)
        z[i + 1] = z[i] + (h / 6) * (k1_z + 2 * k2_z + 2 * k3_z + k4_z)

    return t, x, y, z

# Definitions of the ODEs

def x_prime(x,y,z):
    return 10*(y-x)

def y_prime(x,y,z):
    return (28-z)*x-y
def z_prime(x,y,z):
    return x*y -(8/3)*z



# Initial conditions and time span
x0, y0, z0 = -4, -8, 3
t0, tf = 0, 100  # Start and end time
h = 0.01  # Step size

# Solve the ODE
t, x, y, z = runge_kutta_4(x_prime, y_prime, z_prime, x0, y0, z0, t0, tf, h)

# 3D plot for the trajectory
plt.figure(figsize=(10, 8))
ax = plt.axes(projection='3d')
ax.plot3D(x, y, z, 'blue')
ax.set_title('3D Trajectory using RK4')
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
plt.show()



# Initial conditions and time span
x0, y0, z0 = 4, 3, 8
t0, tf = 0, 100  # Start and end time
h = 0.01  # Step size

# Solve the ODE
t, x, y, z = runge_kutta_4(x_prime, y_prime, z_prime, x0, y0, z0, t0, tf, h)

# 3D plot for the trajectory
plt.figure(figsize=(10, 8))
ax = plt.axes(projection='3d')
ax.plot3D(x, y, z, 'orange')
ax.set_title('3D Trajectory using RK4')
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
plt.show()

