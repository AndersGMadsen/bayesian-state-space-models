import numpy as np
from numpy.random import multivariate_normal as mvn

class CarTrajectoryLinear:
    def __init__(self, N = 100, dt = 0.1, q1 = 1, q2 = 1, s1 = 0.5, s2 = 0.5):
        self.N = N
        self.dt = dt
        self.q1 = q1
        self.q2 = q2
        self.s1 = s1
        self.s2 = s2
        
        self.Q = np.array([[(q1 * dt**3) / 3, 0, (q1 * dt**2) / 2, 0],
                        [ 0, (q2 * dt**3) / 3, 0, (q2 * dt**2) / 2],
                        [(q1 * dt**2) / 2, 0, q1 * dt, 0],
                        [0, (q2 * dt**2) / 2, 0, q2 * dt]])
        
        self.R = np.array([[s1, 0],
                        [0, s2]])
        
        self.A = np.array([[1, 0, dt, 0],
                        [0, 1, 0, dt],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
        
        self.H = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0]])
        
        self.dim_m = 4
        self.dim_y = 2
    
    def get_data(self):
        x = np.zeros((self.N, 4))
        y = np.zeros((self.N, 2))
        
        x[0] = mvn(mean = np.array([0, 0, 0, 0]), cov = self.Q)
        y[0] = mvn(mean = self.H @ x[0], cov = self.R)
        
        for i in range(1, self.N):
            x[i] = mvn(mean = self.A @ x[i - 1], cov = self.Q)
            y[i] = mvn(mean = self.H @ x[i], cov = self.R)
        
        return x, y
    
class CarTrajectoryNonLinear:
    def __init__(self, N = 100, dt = 0.1, q1 = 1, q2 = 1, s1 = 0.5, s2 = 0.5):
        self.N = N
        self.dt = dt
        self.q1 = q1
        self.q2 = q2
        self.s1 = s1
        self.s2 = s2
        
        self.Q = np.array([[(q1 * dt**3) / 3, 0, (q1 * dt**2) / 2, 0],
                        [ 0, (q2 * dt**3) / 3, 0, (q2 * dt**2) / 2],
                        [(q1 * dt**2) / 2, 0, q1 * dt, 0],
                        [0, (q2 * dt**2) / 2, 0, q2 * dt]])
        
        self.R = np.array([[s1, 0],
                        [0, s2]])
        
        self.A = np.array([[1, 0, dt, 0],
                        [0, 1, 0, dt],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
        
        self.H = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0]])
        
        self.dim_m = 4
        self.dim_y = 2
        
    def f_linear(self, x):
        return self.A @ x
    
    def f_nonlinear(self, x):
        return np.array([0.1 * np.sin(x[0]), 0.1 * np.sin(x[1]), 0, 0])
    
    def f(self, x):
        return self.f_linear(x) + self.f_nonlinear(x)
    
    def h_linear(self, x):
        return self.H @ x
    
    def h_nonlinear(self, x):
        return np.array([1.0 * np.sin(x[1]), -1.0 * np.cos(x[0])])

    def h(self, x):
        return self.h_linear(x) + self.h_nonlinear(x)
    
    def F_jacobian(self, x):
        return np.array([[1 + 0.1 * np.cos(x[0]), 0, self.dt, 0],
                        [0, 1 + 0.1 * np.cos(x[1]), 0, self.dt],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])

    def H_jacobian(self, x):
        return np.array([[1, 1.0 * np.cos(x[1]), 0, 0],
                        [1.0 * np.sin(x[0]), 1, 0, 0]])
    
    def get_data(self):
        self.x = np.zeros((self.N, 4))
        self.y = np.zeros((self.N, 2))
        
        self.x[0] = np.array([0, 0, 0, 0])
        self.y[0] = self.h(self.x[0]) + mvn(np.zeros(2), self.R)
        
        for i in range(1, self.N):
            self.x[i] = self.f(self.x[i - 1]) + mvn(np.zeros(4), self.Q)
            self.y[i] = self.h(self.x[i]) + mvn(np.zeros(2), self.R)
        
        return self.x, self.y