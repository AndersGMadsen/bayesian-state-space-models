import numpy as np

class KF:
    def __init__(self, A, Q, H, R, dim_m = 4, dim_y = 2):
        self.A = A
        self.Q = Q
        self.H = H
        self.R = R
        self.dim_m = dim_m
        self.dim_y = dim_y
        
        assert A.shape == (dim_m, dim_m)
        assert Q.shape == (dim_m, dim_m)
        assert H.shape == (dim_y, dim_m)
        assert R.shape == (dim_y, dim_y)
        
    def predict(self, m, P):
        m_pred = self.A @ m
        p_pred = self.A @ P @ self.A.T + self.Q
        return m_pred, p_pred
    
    def update(self, m_pred, P_pred, y):
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)
        m = m_pred + K @ (y - self.H @ m_pred)
        P = P_pred - K @ self.H @ P_pred
        return m, P
    
    # Kalman Filter
    def filter(self, measurements, m = None, P = None):
        n = len(measurements)
        if m is None: m = np.zeros(self.dim_m)
        if P is None: P = np.eye(self.dim_m)
        state_estimates = np.empty((n, self.dim_m))
        cov_estimates = np.empty((n, self.dim_m, self.dim_m))
        
        for i, y in enumerate(measurements):
            m_pred, P_pred = self.predict(m, P)
            m, P = self.update(m_pred, P_pred, y)
            state_estimates[i] = m
            cov_estimates[i] = P            
        
        return state_estimates, cov_estimates
    
    # Rauch-Tung-Striebel (RTS) Smoother
    def smoother(self, states_estimates, cov_estimates):
        n = len(states_estimates)
        states_estimates_smoothed = np.empty((n, self.dim_m))
        states_estimates_smoothed[-1] = states_estimates[-1]
        
        cov_estimates_smoothed = np.empty((n, self.dim_m, self.dim_m))
        cov_estimates_smoothed[-1] = cov_estimates[-1]

        for k in range(n- 2, -1, -1):
            m_pred = self.A @ states_estimates[k]
            P_pred = self.A @ cov_estimates[k] @ self.A.T + self.Q
            G = cov_estimates[k] @ self.A.T @ np.linalg.inv(P_pred)
            
            states_estimates_smoothed[k] = states_estimates[k] + G @ (states_estimates_smoothed[k + 1] - m_pred)
            cov_estimates_smoothed[k] = cov_estimates[k] + G @ (cov_estimates_smoothed[k + 1] - P_pred) @ G.T
            
        return states_estimates_smoothed, cov_estimates_smoothed
    
class EKF:
    def __init__(self, f, F_jacobian, h, H_jacobian, Q, R, dim_m = 4, dim_y = 2):
        self.f = f
        self.F_jacobian = F_jacobian
        self.h = h
        self.H_jacobian = H_jacobian
        self.Q = Q
        self.R = R
        self.dim_m = dim_m
        self.dim_y = dim_y

    def predict(self, m, P):
        F = self.F_jacobian(m)
        m_pred = self.f(m)
        P_pred = F @ P @ F.T + self.Q
        return m_pred, P_pred

    def update(self, m_pred, P_pred, y):
        H = self.H_jacobian(m_pred)
        S = H @ P_pred @ H.T + self.R
        K = P_pred @ H.T @ np.linalg.inv(S)
        
        m = m_pred + K @ (y - self.h(m_pred))
        P = P_pred - K @ S @ K.T
        
        return m, P

    # Extended Kalman Filtering
    def filter(self, measurements, m = None, P = None):
        n = len(measurements)
        if m is None: m = np.zeros(self.dim_m)
        if P is None: P = np.eye(self.dim_m)
        state_estimates = np.empty((n, self.dim_m))
        cov_estimates = np.empty((n, self.dim_m, self.dim_m))
        
        for i, y in enumerate(measurements):
            m_pred, P_pred = self.predict(m, P)
            m, P = self.update(m_pred, P_pred, y)
            state_estimates[i] = m
            cov_estimates[i] = P
            
        return state_estimates, cov_estimates
    
    # Rauch-Tung-Striebel (RTS) Smoother
    def smoother(self, states_estimates, cov_estimates):
        n = len(states_estimates)
        states_estimates_smoothed = np.empty((n, self.dim_m))
        states_estimates_smoothed[-1] = states_estimates[-1]
        
        cov_estimates_smoothed = np.empty((n, self.dim_m, self.dim_m))
        cov_estimates_smoothed[-1] = cov_estimates[-1]

        for k in range(n-2, -1, -1):
            F = self.F_jacobian(states_estimates[k])
            m_pred = self.f(states_estimates[k])
            P_pred = F @ cov_estimates[k] @ F.T + self.Q
            G = cov_estimates[k] @ F.T @ np.linalg.inv(P_pred)
            states_estimates_smoothed[k] = states_estimates[k] + G @ (states_estimates_smoothed[k+1] - m_pred)
            cov_estimates_smoothed[k] = cov_estimates[k] + G @ (cov_estimates_smoothed[k+1] - P_pred) @ G.T
            
        return states_estimates_smoothed, cov_estimates_smoothed