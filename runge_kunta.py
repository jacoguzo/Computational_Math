import numpy as np

import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 12})
# Define the classical 4-stage Runge-Kutta method
def runge_kutta_4(f, y0, t0, tf, h):
    """
    Solve the ODE y' = f(t, y) using the classical 4-stage Runge-Kutta method.

    Parameters:
    f: function
        The derivative function. Must take two arguments (t, y).
    y0: float
        Initial condition for y.
    t0: float
        Initial time.
    tf: float
        Final time.
    h: float
        The step size.

    Returns:
    t: array
        Array of time points where the solution was computed.
    y: array
        Array containing the solution corresponding to each time point in t.
    """
    # Calculate the number of steps (ensure it's an integer)
    n = int((tf - t0) / h)

    # Initialize arrays to hold time points and solution values
    #np.linspace creates an evenly spread array of numbers given start, end, amount of numbers
    t = np.linspace(t0, tf, n + 1)
    y = np.zeros(n + 1)
    y[0] = y0

    y_err = np.zeros(n+1)

   
    # Runge-Kutta 4th order method loop
    for i in range(n):
        k1 = f(t[i], y[i])
        k2 = f(t[i] + h / 2, y[i] + h / 2 * k1)
        k3 = f(t[i] + h / 2, y[i] + h / 2 * k2)
        k4 = f(t[i] + h, y[i] + h * k3)
        
        y[i + 1] = y[i] + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        y_err[i+1]= (y[i+1]-y_true(t[i+1]))/(h**4)
    
    return t, y, y_err
#definition of ODE
def f(t, y):
    return np.exp(t - y)
#definition of true solution
def y_true(t):
    return np.log(np.exp(t)+np.exp(1)-1)

# Initial conditions and time span
y0 = 1  # y(t0)
t0 = 0  # Start time
tf = 4  # End time
h = 0.5  # Step size
n1 = int((tf - t0) / h)



# Solve the ODE
t, y, y_err = runge_kutta_4(f, y0, t0, tf, h)
print(y)
print(t)
print(y_err)

# Print the final value as an example
print(t[-1], y[-1])

T_true = np.linspace(t0, tf, n1)
Y_true = y_true(T_true)

plt.figure(figsize=(7,5))

plt.plot(T_true, Y_true,label="Analytical solution",color="red", lw=2)

plt.plot(t, y, label="Numerical solution:\nRunge-Kutta w/ h=0.5", dashes=(3,2), color="blue", lw=3)
#plt.plot(x_eu, y_eu, label="Numerical solution:\nEuler", dashes=(3,2), color="green", lw=3)

plt.legend(loc="best", fontsize=12)
plt.title(r"Solution to ODE: $\quad\frac{dy}{dt}=e^{(t-y)}$ using RK4 with h =0.5")
plt.xlabel("t", fontsize=12)
plt.ylabel("y", fontsize=12)
plt.show()











#plot 0,25



# Initial conditions and time span
y0 = 1  # y(t0)
t0 = 0  # Start time
tf = 4  # End time
h = 0.25  # Step size
n2 = int((tf - t0) / h)

# Solve the ODE
t1, y1,y_err1 = runge_kutta_4(f, y0, t0, tf, h)

# Print the final value as an example
print(t1[-1], y1[-1])


T_true = np.linspace(t0, tf, n2)
Y_true = y_true(T_true)

plt.figure(figsize=(7,5))

plt.plot(T_true, Y_true,label="Analytical solution",color="green", lw=2)

plt.plot(t1, y1, label="Numerical solution:\nRunge-Kutta w/ h=0.25", dashes=(3,2), color="purple", lw=3)
#plt.plot(x_eu, y_eu, label="Numerical solution:\nEuler", dashes=(3,2), color="green", lw=3)

plt.legend(loc="best", fontsize=12)
plt.title(r"Solution to ODE: $\quad\frac{dy}{dt}=e^{(t-y)}$ using RK4 with h =0.25")
plt.xlabel("t", fontsize=12)
plt.ylabel("y", fontsize=12)
plt.show()

#plotting errors


plt.figure(figsize=(7,5))

plt.plot(t, y_err,label="error: h =0.5",color="orange", lw=2)

plt.plot(t1, y_err1, label="error: h =0.25", dashes=(3,2), color="pink", lw=3)
#plt.plot(x_eu, y_eu, label="Numerical solution:\nEuler", dashes=(3,2), color="green", lw=3)

plt.legend(loc="best", fontsize=12)
plt.title(r"Error comparison for h=0.5 and h=0.25")
plt.xlabel("t", fontsize=12)
plt.ylabel("y", fontsize=12)
plt.show()














