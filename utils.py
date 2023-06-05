import numpy as np
from numpy.random import multivariate_normal as mvn

N = 100
dt = 0.1
q1, q2 = 1, 1
s1, s2 = 0.5, 0.5

Q = np.array([[(q1 * dt**3) / 3, 0, (q1 * dt**2) / 2, 0],
            [ 0, (q2 * dt**3) / 3, 0, (q2 * dt**2) / 2],
            [(q1 * dt**2) / 2, 0, q1 * dt, 0],
            [0, (q2 * dt**2) / 2, 0, q2 * dt]])

R = np.array([[s1, 0],
            [0, s2]])

A = np.array([[1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]])

H = np.array([[1, 0, 0, 0],
            [0, 1, 0, 0]])

# Nonlinear state transition function
def f_linear(x):
    return A @ x

def f_nonlinear(x):
    return np.array([0.1 * np.sin(x[0]), 0.1 * np.sin(x[1]), 0, 0])

def f(x):
    return f_linear(x) + f_nonlinear(x)

# Nonlinear observation function
def h_linear(x):
    return H @ x

def h_nonlinear(x):
    return np.array([1.0 * np.sin(x[1]), -1.0 * np.cos(x[0])])

def h(x):
    return h_linear(x) + h_nonlinear(x)

# Jacobians of the state transition and observation functions
def F_jacobian(x):
    return np.array([[1 + 0.1 * np.cos(x[0]), 0, dt, 0],
                    [0, 1 + 0.1 * np.cos(x[1]), 0, dt],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])

def H_jacobian(x):
    return np.array([[1, 1.0 * np.cos(x[1]), 0, 0],
                     [1.0 * np.sin(x[0]), 1, 0, 0]])

def generate_trajectory_nonlinear():
    # Generate data with nonlinear true path and nonlinear observations
    x = np.array([0, 0, 1, 1])
    true_trajectory = [x]
    noisy_observations = [h(x) + mvn([0, 0], R)]

    for _ in range(N):
        x = f(x) + mvn([0, 0, 0, 0], Q)
        y = h(x) + mvn([0, 0], R)

        true_trajectory.append(x)
        noisy_observations.append(y)

    true_trajectory = np.array(true_trajectory)
    noisy_observations = np.array(noisy_observations)
    return true_trajectory, noisy_observations
    
def generate_trajectory_linear():
    # Generate data
    x = np.array([0, 0, 1, 1])
    true_trajectory = [x]
    noisy_observations = [x[:2]]

    for _ in range(N):
        x = A @ x + mvn([0, 0, 0, 0], Q)
        y = H @ x + mvn([0, 0], R)

        true_trajectory.append(x)
        noisy_observations.append(y)

    true_trajectory = np.array(true_trajectory)
    noisy_observations = np.array(noisy_observations)
    return true_trajectory, noisy_observations