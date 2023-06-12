import numpy as np
from scipy.linalg import cholesky

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
    def smoother(self, state_estimates, cov_estimates):
        n = len(state_estimates)
        state_estimates_smoothed = np.empty((n, self.dim_m))
        state_estimates_smoothed[-1] = state_estimates[-1]
        
        cov_estimates_smoothed = np.empty((n, self.dim_m, self.dim_m))
        cov_estimates_smoothed[-1] = cov_estimates[-1]

        for k in range(n- 2, -1, -1):
            m_pred = self.A @ state_estimates[k]
            P_pred = self.A @ cov_estimates[k] @ self.A.T + self.Q
            G = cov_estimates[k] @ self.A.T @ np.linalg.inv(P_pred)
            
            state_estimates_smoothed[k] = state_estimates[k] + G @ (state_estimates_smoothed[k + 1] - m_pred)
            cov_estimates_smoothed[k] = cov_estimates[k] + G @ (cov_estimates_smoothed[k + 1] - P_pred) @ G.T
            
        return state_estimates_smoothed, cov_estimates_smoothed
    
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

    # Extended Kalman Filtering (EFK)
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
    
    # Extended Rauch-Tung-Striebel (ERTS) Smoother
    def smoother(self, state_estimates, cov_estimates):
        n = len(state_estimates)

        state_estimates_smoothed = np.empty((n, self.dim_m))
        state_estimates_smoothed[-1] = state_estimates[-1]
        
        cov_estimates_smoothed = np.empty((n, self.dim_m, self.dim_m))
        cov_estimates_smoothed[-1] = cov_estimates[-1]

        for k in range(n-2, -1, -1):
            F = self.F_jacobian(state_estimates[k])
            m_pred = self.f(state_estimates[k])
            P_pred = F @ cov_estimates[k] @ F.T + self.Q
            G = cov_estimates[k] @ F.T @ np.linalg.inv(P_pred)
            state_estimates_smoothed[k] = state_estimates[k] + G @ (state_estimates_smoothed[k+1] - m_pred)
            cov_estimates_smoothed[k] = cov_estimates[k] + G @ (cov_estimates_smoothed[k+1] - P_pred) @ G.T
            
        return state_estimates_smoothed, cov_estimates_smoothed
    
# Unscented Kalman Filter (merwe)
class UKF:
    def __init__(self, f, Q, h, R, dim_m = 4, dim_y = 2, alpha=0.5, beta=2, kappa=0):
        self.f = f
        self.Q = Q
        self.h = h
        self.R = R
        self.dim_m = dim_m
        self.dim_y = dim_y
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        
        assert Q.shape == (dim_m, dim_m)
        assert R.shape == (dim_y, dim_y)

    def weights(self, n):
            
        lambda_ = self.alpha**2 * (n + self.kappa) - n

        Wc = np.full(2*n + 1, 1 / (2*(n + lambda_)))
        Wm = Wc.copy()

        Wc[0] = lambda_ / (n + lambda_) + (1 - self.alpha**2 + self.beta)
        Wm[0] = lambda_ / (n + lambda_)
        
        return Wm, Wc

    def sigma_points(self, n, m, P):

        sigma_points = np.empty((2*n + 1, n))

        lambda_ = self.alpha**2 * (n + self.kappa) - n
        U = cholesky((n + lambda_) * P)

        sigma_points[0] = m
        for i in range(n):
            sigma_points[i + 1] = m + U[i]
            sigma_points[n + i + 1] = m - U[i]

        return sigma_points

    def predict(self, m, P, Wm, Wc):
        
        sigma_points = self.sigma_points(self.dim_m, m, P)
        sigma_points_transformed = np.array([self.f(sigma) for sigma in sigma_points])
        
        m_pred = np.sum(Wm[:, None] * sigma_points_transformed, axis=0)
        P_pred = np.sum([Wc[i] * np.outer(sigma_points_transformed[i] - m_pred, sigma_points_transformed[i] - m_pred) for i in range(len(Wc))], axis=0) + self.Q

        return m_pred, P_pred

    def update(self, m_pred, P_pred, y, Wm, Wc):

        sigma_points = self.sigma_points(self.dim_m, m_pred, P_pred)
        sigma_points_transformed = np.array([self.h(sigma) for sigma in sigma_points])
        
        y_pred = np.sum(Wm[:, None] * sigma_points_transformed, axis=0)
        S = np.sum([Wc[i] * np.outer(sigma_points_transformed[i] - y_pred, sigma_points_transformed[i] - y_pred) for i in range(len(Wc))], axis=0) + self.R
        C = np.sum([Wc[i] * np.outer(sigma_points[i] - m_pred, sigma_points_transformed[i] - y_pred) for i in range(len(Wc))], axis=0)

        K = C @ np.linalg.inv(S)
        
        m = m_pred + K @ (y - y_pred)
        P = P_pred - K @ S @ K.T

        return m, P

    # Unscented Kalman Filtering (UKF)
    def filter(self, measurements, m = None, P = None):
        n_measurements = len(measurements)

        if m is None: m = np.zeros(self.dim_m)
        if P is None: P = np.eye(self.dim_m)

        state_estimates = np.empty((n_measurements, self.dim_m))
        cov_estimates = np.empty((n_measurements, self.dim_m, self.dim_m))

        Wm, Wc = self.weights(self.dim_m)
        
        for i, y in enumerate(measurements):
            m_pred, P_pred = self.predict(m, P, Wm, Wc)
            m, P = self.update(m_pred, P_pred, y, Wm, Wc)
            state_estimates[i] = m
            cov_estimates[i] = P            
        
        return state_estimates, cov_estimates

    # Unscented Rauch-Tung-Striebel (URTS) Smoother
    def smoother(self, state_estimates, cov_estimates):
        n = len(state_estimates)

        state_estimates_smoothed = np.empty((n, self.dim_m))
        state_estimates_smoothed[-1] = state_estimates[-1]
        
        cov_estimates_smoothed = np.empty((n, self.dim_m, self.dim_m))
        cov_estimates_smoothed[-1] = cov_estimates[-1]

        Wm, Wc = self.weights(self.dim_m)

        for k in range(n - 2, -1, -1):
            m = state_estimates[k]
            P = cov_estimates[k]

            sigma_points = self.sigma_points(self.dim_m, m, P)
            sigma_points_transformed = np.array([self.f(sigma) for sigma in sigma_points])

            m_pred = np.sum(Wm[:, None] * sigma_points_transformed, axis=0)
            P_pred = np.sum([Wc[i] * np.outer(sigma_points_transformed[i] - m_pred, sigma_points_transformed[i] - m_pred) for i in range(len(Wc))], axis=0) + self.Q
            D = np.sum([Wc[i] * np.outer(sigma_points[i] - m, sigma_points_transformed[i] - m_pred) for i in range(len(Wc))], axis=0)
            G = D @ np.linalg.inv(P_pred)

            state_estimates_smoothed[k] = m + G @ (state_estimates_smoothed[k + 1] - m_pred)
            cov_estimates_smoothed[k] = P + G @ (cov_estimates_smoothed[k + 1] - P_pred) @ G.T

        return state_estimates_smoothed, cov_estimates_smoothed