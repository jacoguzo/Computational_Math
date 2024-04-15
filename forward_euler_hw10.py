import numpy as np
import matplotlib.pyplot as plt

# Function definitions
def f(t, y):
    return 10*np.sin(y)

def y_true(t, y0):
    return y0 * np.exp(-75 * t)

def forward_euler(f, a, b, y0, h):
    t = a
    y = y0
    Ts = [t]
    Ys = [y]

    while t <= b:
        t += h
        y += h * f(t, y)
        Ts.append(t)
        Ys.append(y)

    return Ts, Ys

# Parameters
a = 0   # start of interval
b = 4.5 # end of interval
y0 = 3  # initial value
h1 = 0.22 # step size


# Calculate solutions
Ts1, Ys1 = forward_euler(f, a, b, y0, h1)

T_true = np.linspace(a, b, 100)
Y_true = y_true(T_true, y0)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(Ts1, Ys1, 'b.-', label='Forward Euler Approximation')
plt.plot(Ts1, np.cos(Ys1), 'g.-', label='cos(y)')

plt.title('Comparison of Forward Euler Method and True Solution')
plt.xlabel('t')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()



