import numpy as np
from scipy.stats import norm
import casadi

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
        
    def f_linear(self, x, ca):
        if ca:
            return casadi.mtimes(casadi.MX(self.A), x)
        else:
            return self.A @ x
    
    # nonlinear f and h contain sines and cosines which is simply an example
    # of nonlinearity - these incorporate bias into the model (maybe of drunk driving?)
    def f_nonlinear(self, x, freq, amp, ca):
        if ca:
            #return casadi.vertcat(amp * casadi.sin(freq * x[0, :]), amp * casadi.sin(freq * x[1, :]), casadi.MX.zeros(1, x.shape[1]), casadi.MX.zeros(1, x.shape[1]))
            return casadi.vertcat(amp * casadi.sin(x[0, :]), amp * casadi.sin(x[1, :]), casadi.MX.zeros(1, x.shape[1]), casadi.MX.zeros(1, x.shape[1]))
        else:
            #return np.array([amp * np.sin(freq * x[0]), amp * np.sin(freq * x[1]), 0, 0])
            return np.array([amp * np.sin(x[0]), amp * np.sin(x[1]), 0, 0])
    
    def f(self, x, freq=np.pi/2, amp=0.2, ca=False):
        return self.f_linear(x, ca) + self.f_nonlinear(x, freq, amp, ca)
    
    def h_linear(self, x, ca):
        if ca:
            return casadi.mtimes(casadi.MX(self.H), x)
        else:
            return self.H @ x
    
    def h_nonlinear(self, x, freq, ca=False):
        if ca:
            return casadi.vertcat(1.0 * casadi.sin(x[1, :]), -1.0 * casadi.cos(x[0, :]))
        else:
            return np.array([1.0 * np.sin(freq * x[1]), -1.0 * np.cos(freq * x[0])])

    def h(self, x, freq=2*np.pi, ca=False):
        return self.h_linear(x, ca) + self.h_nonlinear(x, freq, ca)
    
    def F_jacobian(self, x):
        return np.array([[1 + 0.1 * np.cos(x[0]), 0, self.dt, 0],
                        [0, 1 + 0.1 * np.cos(x[1]), 0, self.dt],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])

    def H_jacobian(self, x):
        return np.array([[1, 1.0 * np.cos(x[1]), 0, 0],
                        [1.0 * np.sin(x[0]), 1, 0, 0]])