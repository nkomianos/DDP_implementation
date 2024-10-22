import numpy as np
import sympy as sym


# f: dynamics function R^d -> R^d, hence require \nabla(f) = jacobian of f

def jacobian(x, u, F):
    F = sym.Matrix(F)
    Fx = F.jacobian(x).T
    Fu = F.jacobian(u).T

    return Fx, Fu


def GradientHessian(x, u, args):
    functions = list()
    for fun in args:
        fx, fu = jacobian(x, u, fun)
        functions.append(fx)
        functions.append(fu)
        fxx, fxu = jacobian(x, u, fx)
        functions.append(fxx)
        functions.append(fxu)
        _, fuu = jacobian(x, u, fu)
        functions.append(fuu)

    return functions


def terminal_cost(x, Q=None, target=0):
    return (x-target).T @ Q @ (x-target)


def running_cost(x, u, Q, R):
    return x.T @ Q @ x + u.T @ R @ u


def backprop(x, u, F, Q, R, time_horizon, x_bar, u_bar, dt):

    #Initialize feedback gains k and K to zeros
    k = np.zeros((u_bar.shape[0], time_horizon))
    K = np.zeros((u_bar.shape[0], x_bar.shape[0], time_horizon))

    #Compute the terminal cost Phi and the running cost L0.

    Phi = terminal_cost(x, Q)
    L0 = running_cost(x, u, Q, R)

    #Compute the first and second-order derivatives of the cost functions using GradientHessian.
    functions = GradientHessian(x, u, [L0, Phi])

    #Unpack Derivatives

    Lx, Lu, Lxx, Lxu, Luu, Phix, Phiu, Phixx, Phixu, Phiuu = tuple(functions)

    #Convert symbolic expressions of the terminal cost and its derivatives to numerical functions

    Phi_fun = sym.lambdify([x], sym.Matrix(Phi), "numpy")
    Phix_fun = sym.lambdify([x], sym.Matrix(Phix), "numpy")
    Phixx_fun = sym.lambdify([x], sym.Matrix(Phixx), "numpy")

    #Convert symbolic expressions of the running cost and its derivatives to numerical functions.

    L0_fun = sym.lambdify([x, u], sym.Matrix(L0), "numpy")
    Lx_fun = sym.lambdify([x, u], sym.Matrix(Lx), "numpy")
    Lu_fun = sym.lambdify([x, u], sym.Matrix(Lu), "numpy")
    Lxx_fun = sym.lambdify([x, u], sym.Matrix(Lxx), "numpy")
    Lxu_fun = sym.lambdify([x, u], sym.Matrix(Lxu), "numpy")
    Luu_fun = sym.lambdify([x, u], sym.Matrix(Luu), "numpy")

    # Value function at last time step is equal to terminal cost

    #????
    V = Phi_fun(x_bar[:, [-1]])
    Vx = Phix_fun(x_bar[:, [-1]])
    Vxx = Phixx_fun(x_bar[:, [-1]])

    for t in range(time_horizon - 1, -1, -1):

        #compute next state using system dynamics and jacobians
        x_t1 = x + F * dt
        fx = sym.Matrix(x_t1).jacobian(x)
        fu = sym.Matrix(x_t1).jacobian(u)

        #compute quadratic approx. terms for the cost function

        Q0 = L0_fun(x_bar[:, [t]], u_bar[:, [t]]) + V
        Qx = Lx_fun(x_bar[:, [t]], u_bar[:, [t]]) + fx.T @ Vx
        Qu = Lu_fun(x_bar[:, [t]], u_bar[:, [t]]) + fu.T @ Vx
        Qxx = Lxx_fun(x_bar[:, [t]], u_bar[:, [t]]) + fx.T @ Vxx @ fx
        Qxu = Lxu_fun(x_bar[:, [t]], u_bar[:, [t]]) + fu.T @ Vxx @ fx
        Quu = Luu_fun(x_bar[:, [t]], u_bar[:, [t]]) + fu.T @ Vxx @ fu
        # check eigen values for positive semi-definiteness 
        eig_val_Quu = np.linalg.eig(np.array(Quu, dtype=float))
        if min(eig_val_Quu) < 0:
            print('Quu is not PSD')
            break

        #compute inverse of Quu

        invQuu = np.linalg.inv(np.array(Quu, dtype='float64'))

        #compute gains using inverse

        K[:, :, t] = -invQuu @ Qxu
        k[:, t] = -invQuu @ Qu

        #update value function and its derivatives page 11 of technical report
        #-dV/dt, -dVx/dt, -dVxx/dt

        V = Q0 - (1 / 2) * (Qu @ invQuu) @ Qu.T
        Vx = Qx - Qxu.T @ invQuu @ Qu
        Vxx = Qxx - Qxu.T @ invQuu @ Qxu
        # check eigen values to ensure positive semi-definiteness for Vxx
        eig_val_Vxx, _ = np.linalg.eig(np.array(Vxx, dtype=float))
        if min(eig_val_Vxx) < 0:
            print('Vxx is not PSD')
            break

    return V, Vx, Vxx, k, K


def forward(x, u, x_initial, x_bar, u_bar, F, time_horizon, k, K, alpha, dt):
    #alpha step size param for the the update

    #init predicted state and control trajectories
    x_hat = np.zeros((x_bar.shape[0], time_horizon))
    u_hat = np.zeros((u_bar.shape[0], time_horizon))

    #compute next state and convert to function
    x_t1 = x + F * dt
    x_t1 = sym.lambdify((x, u), sym.Matrix(x_t1), "numpy")

    #init first column  with the initial state

    x_hat[:, [0]] = x_initial

    for t in range(time_horizon - 1):

        #update nominal control input

        #??? why dif from pseudo code

        u_hat[:, [t]] = u_bar[:, [t]] + alpha * k[:, t] + K[:, :, t] @ (x_hat[:, [t]] - x_bar[:, [t]])

        #update state

        x_hat[:, [t + 1]] = x_t1(x_hat[:, [t]], u_hat[:, [t]])

    return x_hat, u_hat
