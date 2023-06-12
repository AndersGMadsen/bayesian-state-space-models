import numpy as np

class KF:
    def __init__(self, A, Q, H, R, dim_x = 4, dim_y = 2):
        self.A = A
        self.Q = Q
        self.H = H
        self.R = R
        self.dim_x = dim_x
        self.dim_y = dim_y
        
        assert A.shape == (dim_x, dim_x)
        assert Q.shape == (dim_x, dim_x)
        assert H.shape == (dim_y, dim_x)
        assert R.shape == (dim_y, dim_y)
        
    def predict(self, x, P):
        m_pred = self.A @ x
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
        if m is None: m = np.zeros(self.dim_x)
        if P is None: P = np.eye(self.dim_x)
        state_estimates = np.empty((n, self.dim_x))
        cov_estimates = np.empty((n, self.dim_x, self.dim_x))
        
        for i, y in enumerate(measurements):
            m_pred, P_pred = self.predict(m, P)
            m, P = self.update(m_pred, P_pred, y)
            state_estimates[i] = m
            cov_estimates[i] = P            
        
        return state_estimates, cov_estimates
    
    # Rauch-Tung-Striebel (RTS) Smoother
    def smoother(self, states_estimates, cov_estimates):
        n = len(states_estimates)
        states_estimates_smoothed = np.empty((n, self.dim_x))
        states_estimates_smoothed[-1] = states_estimates[-1]
        
        cov_estimates_smoothed = np.empty((n, self.dim_x, self.dim_x))
        cov_estimates_smoothed[-1] = cov_estimates[-1]

        for k in range(n- 2, -1, -1):
            m_pred = self.A @ states_estimates[k]
            P_pred = self.A @ cov_estimates[k] @ self.A.T + self.Q
            G = cov_estimates[k] @ self.A.T @ np.linalg.inv(P_pred)
            
            states_estimates_smoothed[k] = states_estimates[k] + G @ (states_estimates_smoothed[k + 1] - m_pred)
            cov_estimates_smoothed[k] = cov_estimates[k] + G @ (cov_estimates_smoothed[k + 1] - P_pred) @ G.T
            
        return states_estimates_smoothed, cov_estimates_smoothed
    
    
    