import sympy as sym
import numpy as np
import time
import matplotlib.pyplot as plt
from Diff_Dynamic_Prog import *

# # Mass Spring system properties.
mass = 1
K = 1

# Cart Pendulum
M = 0.5
m = 0.2
b = 0.1
I = 0.006
g = 9.8
l = 0.3
denominator = I * (M + m) + M * m * pow(l, 2)

# # Mass spring dynamics
A = np.array([[0, 1],
              [-K / mass, 0]])
B = np.array([[0],
              [1 / m]])

## Inverted pendulum dynamics
#
# A = np.array([
#     [0, 1, 0, 0],
#     [0, -(I + m * pow(l, 2)) * b / denominator, pow(m, 2) * g * pow(l, 2) / denominator, 0],
#     [0, 0, 0, 1],
#     [0, -m * l * b / denominator, m * g * l * (M + m) / denominator, 0]
# ])
# B = np.array([
#     [0],
#     [(I + m * pow(l, 2) / denominator)],
#     [0],
#     [m * l / denominator]])

Nx = A.shape[0]  # No of states
Nu = B.shape[1]

u = sym.symarray('u', (Nu, 1))
x = sym.symarray('x', (Nx, 1))

F = A @ x + B @ u
dt = 0.01  # Time step
x_t1 = x + F * dt

if A.shape[0] == 2:
    Q = np.array([[50, 0],
                  [0, 0.005]])
else:
    Q = np.array([[5.0, 0, 0, 0],
                  [0, 0.1, 0, 0],
                  [0, 0, 5, 0],
                  [0, 0, 0, 0.1]])

R = np.array([[0.05]])

# Begin Process
time_horizon = 501
t = np.arange(0.0, time_horizon) * dt

# Initial Conditions
x_initial = np.ones((Nx, 1)) * 10
uInit = np.array([[100]])

# Initial Nominal Trajectories
x_bar = x_initial @ np.ones((1, time_horizon))
u_bar = uInit @ np.ones((1, time_horizon))

alpha = 1
miu = 0
max_iter = 20
convergence_threshold = 0.001
V_old = 0.0
V_total = []

for i in range(max_iter):
    tic = time.time()

    # 2) Backward Pass
    V_new, Vx, Vxx, k, K = backprop(x, u, F, Q, R, time_horizon, x_bar, u_bar, dt)
    V_new = np.float64(V_new)
    # 3) Forward Pass
    x_hat, u_hat = forward(x, u, x_initial, x_bar, u_bar, F, time_horizon, k, K, alpha, dt)

    x_bar = x_hat
    u_bar = u_hat
    V_total.append(V_new.reshape(-1, ))
    toc = 1000 * (time.time() - tic)
    print(toc, "ms")

    if (V_new - V_old) <= convergence_threshold:
        break
    else:
        V_old = V_new

# Plots
# plt.title('Pole - Cart Inverted Pendulum')
# plt.plot(t, u_hat.T, label='Control input')
# plt.plot(t, x_hat[0].T, label='Pole Angle ')
# plt.plot(t, x_hat[1].T, label='Pole Angular Velocity')
# plt.plot(t, x_hat[2].T, label='Cart Position')
# plt.plot(t, x_hat[3].T, label='Cart Velocity')
plt.title('Mass Spring')
plt.plot(t, u_hat.T, label='Control input')
plt.plot(t, x_hat[0].T, label='Position')
plt.plot(t, x_hat[1].T, label='Velocity')
plt.legend()
plt.show()


# plt.figure(2)
# plt.plot(np.arange(i + 1), np.array(V_total))
# plt.show()


