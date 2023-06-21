# Implementation by Anders Gjølbye Madsen

import numpy as np
from scipy.linalg import cholesky
from utils.methods import systematic_resampling, residual_resampling, stratified_resampling
from scipy.stats import multivariate_normal as mvn
from tqdm.auto import tqdm

class KF:
    """
    Kalman Filter (KF) class. Provides methods for prediction, 
    update, filtering, and smoothing of a linear system.
    """
    
    def __init__(self, A, Q, H, R, dim_m = 4, dim_y = 2):
        """
        Initialize KF object with given parameters.

        Args:
        A (numpy.ndarray): State transition matrix.
        Q (numpy.ndarray): Process noise covariance matrix.
        H (numpy.ndarray): Measurement matrix.
        R (numpy.ndarray): Measurement noise covariance matrix.
        dim_m (int): Dimension of the state. Default is 4.
        dim_y (int): Dimension of the output. Default is 2.
        """
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
        """
        Perform prediction step in Kalman Filter.

        Args:
        m (numpy.ndarray): State estimate vector.
        P (numpy.ndarray): State covariance matrix.

        Returns:
        m_pred (numpy.ndarray): Predicted state estimate vector.
        p_pred (numpy.ndarray): Predicted state covariance matrix.
        """
        m_pred = self.A @ m
        p_pred = self.A @ P @ self.A.T + self.Q
        return m_pred, p_pred
    
    def update(self, m_pred, P_pred, y):
        """
        Perform update step in Kalman Filter.

        Args:
        m_pred (numpy.ndarray): Predicted state estimate vector from predict step.
        P_pred (numpy.ndarray): Predicted state covariance matrix from predict step.
        y (numpy.ndarray): Current measurement vector.

        Returns:
        m (numpy.ndarray): Updated state estimate vector.
        P (numpy.ndarray): Updated state covariance matrix.
        """
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)
        m = m_pred + K @ (y - self.H @ m_pred)
        P = P_pred - K @ self.H @ P_pred
        return m, P
    
    def filter(self, measurements, m = None, P = None):
        """
        Perform Kalman Filtering on a sequence of measurements.

        Args:
        measurements (numpy.ndarray): Sequence of measurement vectors.
        m (numpy.ndarray): Initial state estimate vector. If None, initialized to zero vector.
        P (numpy.ndarray): Initial state covariance matrix. If None, initialized to identity matrix.

        Returns:
        state_estimates (numpy.ndarray): Sequence of state estimate vectors.
        cov_estimates (numpy.ndarray): Sequence of state covariance matrices.
        """
        n = len(measurements)
        if m is None: m = np.zeros(self.dim_m)
        if P is None: P = np.eye(self.dim_m)
        state_estimates = np.empty((n, self.dim_m))
        cov_estimates = np.empty((n, self.dim_m, self.dim_m))
        
        # Perform filtering for each measurement
        for i, y in enumerate(measurements):
            m_pred, P_pred = self.predict(m, P)
            m, P = self.update(m_pred, P_pred, y)
            state_estimates[i] = m
            cov_estimates[i] = P            
        
        return state_estimates, cov_estimates
    
    def smoother(self, state_estimates, cov_estimates):
        """
        Perform Rauch-Tung-Striebel (RTS) Smoothing on a sequence of state estimates and covariance estimates.

        Args:
        state_estimates (numpy.ndarray): Sequence of state estimate vectors from filter step.
        cov_estimates (numpy.ndarray): Sequence of state covariance matrices from filter step.

        Returns:
        state_estimates_smoothed (numpy.ndarray): Sequence of smoothed state estimate vectors.
        cov_estimates_smoothed (numpy.ndarray): Sequence of smoothed state covariance matrices.
        """
        n = len(state_estimates)
        state_estimates_smoothed = np.empty((n, self.dim_m))
        state_estimates_smoothed[-1] = state_estimates[-1]
        
        cov_estimates_smoothed = np.empty((n, self.dim_m, self.dim_m))
        cov_estimates_smoothed[-1] = cov_estimates[-1]

        # Perform smoothing for each state estimate (backwards from end)
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
    
# Unscented Kalman Filter (UKF)
class UKF:
    def __init__(self, f, h, Q, R, dim_m = 4, dim_y = 2, alpha=0.5, beta=2, kappa=0, method='merwe'):
        self.f = f
        self.Q = Q
        self.h = h
        self.R = R
        self.dim_m = dim_m
        self.dim_y = dim_y
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.method = method
        
        assert Q.shape == (dim_m, dim_m)
        assert R.shape == (dim_y, dim_y)

    def weights(self, n):

        if self.method == 'merwe':
            
            lambda_ = self.alpha**2 * (n + self.kappa) - n

            Wc = np.full(2*n + 1, 1 / (2*(n + lambda_)))
            Wm = Wc.copy()

            Wc[0] = lambda_ / (n + lambda_) + (1 - self.alpha**2 + self.beta)
            Wm[0] = lambda_ / (n + lambda_)

        elif self.method == 'julier':

            Wm = np.full(2*n + 1, 1 / (2*(n + self.kappa)))
            Wm[0] = self.kappa / (n + self.kappa)

            Wc = Wm.copy()
        
        return Wm, Wc

    def sigma_points(self, n, m, P):

        sigma_points = np.empty((2*n + 1, n))

        if self.method == 'merwe':

            lambda_ = self.alpha**2 * (n + self.kappa) - n
            U = cholesky((n + lambda_) * P)

        elif self.method == 'julier':
                
            U = cholesky((n + self.kappa) * P)

        sigma_points[0] = m
        for i in range(n):
            sigma_points[i + 1] = m + U[i]
            sigma_points[n + i + 1] = m - U[i]

        return sigma_points

    def predict(self, m, P, Wm, Wc):
        
        sigma_points = self.sigma_points(self.dim_m, m, P)
        sigma_points_transformed = np.array([self.f(sigma) for sigma in sigma_points])
        
        m_pred = np.sum(Wm[:, None] * sigma_points_transformed, axis=0)
        P_pred = np.sum([Wc[i] * np.outer(sigma_points_transformed[i] - m_pred, sigma_points_transformed[i] - m_pred)
                         for i in range(len(Wc))], axis=0) + self.Q

        return m_pred, P_pred

    def update(self, m_pred, P_pred, y, Wm, Wc):

        sigma_points = self.sigma_points(self.dim_m, m_pred, P_pred)
        sigma_points_transformed = np.array([self.h(sigma) for sigma in sigma_points])
        
        y_pred = np.sum(Wm[:, None] * sigma_points_transformed, axis=0)
        S = np.sum([Wc[i] * np.outer(sigma_points_transformed[i] - y_pred, sigma_points_transformed[i] - y_pred)
                    for i in range(len(Wc))], axis=0) + self.R
        C = np.sum([Wc[i] * np.outer(sigma_points[i] - m_pred, sigma_points_transformed[i] - y_pred)
                    for i in range(len(Wc))], axis=0)

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
            P_pred = np.sum([Wc[i] * np.outer(sigma_points_transformed[i] - m_pred, sigma_points_transformed[i] - m_pred)
                             for i in range(len(Wc))], axis=0) + self.Q
            D = np.sum([Wc[i] * np.outer(sigma_points[i] - m, sigma_points_transformed[i] - m_pred)
                        for i in range(len(Wc))], axis=0)
            G = D @ np.linalg.inv(P_pred)

            state_estimates_smoothed[k] = m + G @ (state_estimates_smoothed[k + 1] - m_pred)
            cov_estimates_smoothed[k] = P + G @ (cov_estimates_smoothed[k + 1] - P_pred) @ G.T

        return state_estimates_smoothed, cov_estimates_smoothed
    
class PF:
    def __init__(self, f, h, Q, R, dim_m = 4, dim_y = 2, N=500):
        self.f = f
        self.h = h
        self.Q = Q
        self.R = R
        self.dim_m = dim_m
        self.dim_y = dim_y
        self.N = N
        
        assert dim_m == len(Q)
        assert dim_y == len(R)
        
        
    def resample(self, particles, weights, method = 'systematic'):
        if 1. / np.sum(np.square(weights)) < self.N / 2:
            if method == 'systematic':
                indexes = systematic_resampling(weights)
            elif method == 'residual':
                indexes = residual_resampling(weights)
            elif method == 'stratified':
                indexes = stratified_resampling(weights)
            else:
                raise ValueError('Unknown resampling method')
            
            return particles[indexes], np.ones(self.N) / self.N
        else:
            return particles, weights
        
    def predict(self, particles, weights):
        for i, particle in enumerate(particles):
            particles[i] = mvn(self.f(particle), self.Q).rvs()
            
        return particles, weights
    
    def update(self, y, particles, weights):
        for i, (weight, particle) in enumerate(zip(weights, particles)):
            weights[i] *= mvn(self.h(particle), self.R).pdf(y)
        
        weights /= np.sum(weights)
        
        return particles, weights
    
    def filter(self, measurements, m = None, P = None, resampling_method = 'systematic', verbose = True):
        if m is None: m = np.zeros(self.dim_m)
        if P is None: P = np.eye(self.dim_m)
        
        n = len(measurements)
        state_estimates = np.empty((n, self.dim_m))
        cov_estimates = np.empty((n, self.dim_m, self.dim_m))
        particle_history = np.empty((n, self.N, self.dim_m))
        weights_history = np.empty((n, self.N))

        # Draw N samples from the prior
        particles = mvn(m, P).rvs(self.N)
        weights = np.ones(self.N) / self.N
        
        if verbose:
            iterator = tqdm(enumerate(measurements), total=n)
        else:
            iterator = enumerate(measurements)
        
        for k, y in iterator:
            particles, weights = self.predict(particles, weights)
            particles, weights = self.update(y, particles, weights)
            
            m = np.average(particles, weights=weights, axis=0)
            P = np.sum([weights[i] * np.outer(particles[i] - m, particles[i] - m)
                        for i in range(self.N)], axis=0)
            
            state_estimates[k] = m
            cov_estimates[k] = P
            particle_history[k] = particles
            weights_history[k] = weights
            
            particles, weights = self.resample(particles, weights, resampling_method)
            
        return state_estimates, cov_estimates, particle_history, weights_history
    
    # Particle Rauch-Tung-Striebel (URTS) Smoother
    def smoother(self, state_estimates, cov_estimates, particle_history, weights_history, verbose = True):
        n = len(state_estimates)
        state_estimates_smoothed = np.empty((n, self.dim_m))
        state_estimates_smoothed[-1] = state_estimates[-1]
        cov_estimates_smoothed = np.empty((n, self.dim_m, self.dim_m))
        cov_estimates_smoothed[-1] = cov_estimates[-1]
        
        if verbose:
            iterator = tqdm(range(n - 2, -1, -1), initial=1, total=n)
        else:
            iterator = range(n - 2, -1, -1)

        for k in iterator:
            m = state_estimates[k]
            P = cov_estimates[k]
            
            weights = weights_history[k]
            particles = particle_history[k]

            particles_transformed = np.empty_like(particles)
            for i, particle in enumerate(particles):
                particles_transformed[i] = self.f(particle)
            
            m_pred = np.average(particles_transformed, weights=weights, axis=0)
            P_pred = np.sum([weights[i] * np.outer(particles_transformed[i] - m_pred, particles_transformed[i] - m_pred)
                             for i in range(self.N)], axis=0) + self.Q
            
            D = np.sum([weights[i] * np.outer(particles[i] - m, particles_transformed[i] - m_pred)
                        for i in range(self.N)], axis=0)
            G = D @ np.linalg.inv(P_pred)
            
            state_estimates_smoothed[k] = m + G @ (state_estimates_smoothed[k + 1] - m_pred)
            cov_estimates_smoothed[k] = P + G @ (cov_estimates_smoothed[k + 1] - P_pred) @ G.T
            
        return state_estimates_smoothed, cov_estimates_smoothed