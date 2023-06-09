import numpy as np
from scipy.stats import norm

class StateSpaceModel:
    def __init__(self, dt = 0.1, q1 = 1, q2 = 1, s1 = 0.5, s2 = 0.5):
        self.dt = dt
        self.q1 = q1
        self.q2 = q2
        self.s1 = s1
        self.s2 = s2
        
        # These dynamics come from Example 4.3 in Bayesian Smoothing and Filtering, Simo Sarkka (2013)
        
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
        
    def get_linear_model(self):
        return self.Q, self.R, self.A, self.H
    
    def get_linear_model_function(self):
        return self.Q, self.R, self.f_linear, self.h_linear
    
    def get_nonlinear_model(self):
        return self.Q, self.R, self.f, self.h
    
    def get_nonlinear_model_jacobian(self):
        return self.F_jacobian, self.H_jacobian
        
    def f_linear(self, x):
        return self.A @ x
    
    # nonlinear f and h contain sines and cosines which is simply an example
    # of nonlinearity - these incorporate bias into the model (maybe of drunk driving?)
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