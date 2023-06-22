# Implementation by Anders Gjølbye Madsen

import numpy as np
from scipy.linalg import cholesky
from utils.methods import systematic_resampling, residual_resampling, stratified_resampling, sample_from_mixture
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
    """
    Extended Kalman Filter (EKF) class. Provides methods for prediction, 
    update, filtering, and smoothing of a nonlinear system.
    """
    
    def __init__(self, f, F_jacobian, h, H_jacobian, Q, R, dim_m = 4, dim_y = 2):
        """
        Initialize EKF object with given parameters.

        Args:
        f (function): Function for state transition.
        F_jacobian (function): Function to compute the Jacobian of f.
        h (function): Function for measurement equation.
        H_jacobian (function): Function to compute the Jacobian of h.
        Q (numpy.ndarray): Process noise covariance matrix.
        R (numpy.ndarray): Measurement noise covariance matrix.
        dim_m (int): Dimension of the state. Default is 4.
        dim_y (int): Dimension of the output. Default is 2.
        """
        self.f = f
        self.F_jacobian = F_jacobian
        self.h = h
        self.H_jacobian = H_jacobian
        self.Q = Q
        self.R = R
        self.dim_m = dim_m
        self.dim_y = dim_y

    def predict(self, m, P):
        """
        Perform prediction step in Extended Kalman Filter.

        Args:
        m (numpy.ndarray): State estimate vector.
        P (numpy.ndarray): State covariance matrix.

        Returns:
        m_pred (numpy.ndarray): Predicted state estimate vector.
        P_pred (numpy.ndarray): Predicted state covariance matrix.
        """
        F = self.F_jacobian(m)
        m_pred = self.f(m)
        P_pred = F @ P @ F.T + self.Q
        return m_pred, P_pred

    def update(self, m_pred, P_pred, y):
        """
        Perform update step in Extended Kalman Filter.

        Args:
        m_pred (numpy.ndarray): Predicted state estimate vector from predict step.
        P_pred (numpy.ndarray): Predicted state covariance matrix from predict step.
        y (numpy.ndarray): Current measurement vector.

        Returns:
        m (numpy.ndarray): Updated state estimate vector.
        P (numpy.ndarray): Updated state covariance matrix.
        """
        H = self.H_jacobian(m_pred)
        S = H @ P_pred @ H.T + self.R
        K = P_pred @ H.T @ np.linalg.inv(S)
        
        m = m_pred + K @ (y - self.h(m_pred))
        P = P_pred - K @ S @ K.T
        
        return m, P

    def filter(self, measurements, m = None, P = None):
        """
        Perform Extended Kalman Filtering on a sequence of measurements.

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
        
        for i, y in enumerate(measurements):
            m_pred, P_pred = self.predict(m, P)
            m, P = self.update(m_pred, P_pred, y)
            state_estimates[i] = m
            cov_estimates[i] = P
            
        return state_estimates, cov_estimates
    
    def smoother(self, state_estimates, cov_estimates):
        """
        Perform Extended Rauch-Tung-Striebel (ERTS) Smoothing on a sequence of state estimates and covariance estimates.

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

        for k in range(n-2, -1, -1):
            F = self.F_jacobian(state_estimates[k])
            m_pred = self.f(state_estimates[k])
            P_pred = F @ cov_estimates[k] @ F.T + self.Q
            G = cov_estimates[k] @ F.T @ np.linalg.inv(P_pred)
            state_estimates_smoothed[k] = state_estimates[k] + G @ (state_estimates_smoothed[k+1] - m_pred)
            cov_estimates_smoothed[k] = cov_estimates[k] + G @ (cov_estimates_smoothed[k+1] - P_pred) @ G.T
            
        return state_estimates_smoothed, cov_estimates_smoothed
    
class UKF:
    """
    Unscented Kalman Filter (UKF) class. Provides methods for prediction, 
    update, filtering, and smoothing of a non-linear system using the Unscented 
    Transform for state and covariance estimation.
    """
    
    def __init__(self, f, h, Q, R, dim_m = 4, dim_y = 2, alpha=0.5, beta=2, kappa=0, method='merwe'):
        """
        Initialize UKF object with given parameters.

        Args:
        f (function): State transition function.
        h (function): Measurement function.
        Q (numpy.ndarray): Process noise covariance matrix.
        R (numpy.ndarray): Measurement noise covariance matrix.
        dim_m (int): Dimension of the state. Default is 4.
        dim_y (int): Dimension of the output. Default is 2.
        alpha (float): Scaling parameter for the sigma points. Default is 0.5.
        beta (float): Parameter which incorporates prior knowledge about the distribution of the state. Default is 2.
        kappa (float): Secondary scaling parameter. Default is 0.
        method (str): Method for calculating sigma points and weights. Default is 'merwe'.

        """
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
        """
        Calculate weights for sigma points.

        Args:
        n (int): Dimension of the state.

        Returns:
        Wm (numpy.ndarray): Weights for the mean.
        Wc (numpy.ndarray): Weights for the covariance.
        """
        # choose method for weight calculation
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
        """
        Generate sigma points.

        Args:
        n (int): Dimension of the state.
        m (numpy.ndarray): State estimate vector.
        P (numpy.ndarray): State covariance matrix.

        Returns:
        sigma_points (numpy.ndarray): Generated sigma points.
        """
        sigma_points = np.empty((2*n + 1, n))

        # choose method for sigma point calculation
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
        """
        Perform prediction step of the UKF.

        Args:
        m (numpy.ndarray): State estimate vector.
        P (numpy.ndarray): State covariance matrix.
        Wm (numpy.ndarray): Weights for the mean.
        Wc (numpy.ndarray): Weights for the covariance.

        Returns:
        m_pred (numpy.ndarray): Predicted state estimate vector.
        P_pred (numpy.ndarray): Predicted state covariance matrix.
        """
        sigma_points = self.sigma_points(self.dim_m, m, P)
        sigma_points_transformed = np.array([self.f(sigma) for sigma in sigma_points])
        
        m_pred = np.sum(Wm[:, None] * sigma_points_transformed, axis=0)
        P_pred = np.sum([Wc[i] * np.outer(sigma_points_transformed[i] - m_pred, sigma_points_transformed[i] - m_pred)
                         for i in range(len(Wc))], axis=0) + self.Q

        return m_pred, P_pred

    def update(self, m_pred, P_pred, y, Wm, Wc):
        """
        Perform update step of the UKF.

        Args:
        m_pred (numpy.ndarray): Predicted state estimate vector.
        P_pred (numpy.ndarray): Predicted state covariance matrix.
        y (numpy.ndarray): Measurement vector.
        Wm (numpy.ndarray): Weights for the mean.
        Wc (numpy.ndarray): Weights for the covariance.

        Returns:
        m (numpy.ndarray): Updated state estimate vector.
        P (numpy.ndarray): Updated state covariance matrix.
        """
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
        """
        Perform UKF on a sequence of measurements.

        Args:
        measurements (numpy.ndarray): Sequence of measurement vectors.
        m (numpy.ndarray): Initial state estimate vector. Default is zero vector.
        P (numpy.ndarray): Initial state covariance matrix. Default is identity matrix.

        Returns:
        state_estimates (numpy.ndarray): Sequence of state estimate vectors.
        cov_estimates (numpy.ndarray): Sequence of state covariance matrices.
        """
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
        """
        Perform Unscented Rauch-Tung-Striebel (URTS) smoothing on a sequence of state estimates.

        Args:
        state_estimates (numpy.ndarray): Sequence of state estimate vectors.
        cov_estimates (numpy.ndarray): Sequence of state covariance matrices.

        Returns:
        state_estimates_smoothed (numpy.ndarray): Sequence of smoothed state estimate vectors.
        cov_estimates_smoothed (numpy.ndarray): Sequence of smoothed state covariance matrices.
        """
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
    """
    Particle Filter (PF) class. Provides methods for prediction, update, filtering, resampling and smoothing of a non-linear system.
    """

    def __init__(self, f, h, Q, R, dim_m=4, dim_y=2, N=500):
        """
        Initialize PF object with given parameters.

        Args:
        f (function): Function for state transition.
        h (function): Function for measurement.
        Q (numpy.ndarray): Process noise covariance matrix.
        R (numpy.ndarray): Measurement noise covariance matrix.
        dim_m (int): Dimension of the state. Default is 4.
        dim_y (int): Dimension of the output. Default is 2.
        N (int): Number of particles. Default is 500.
        """
        self.f = f
        self.h = h
        self.Q = Q
        self.R = R
        self.dim_m = dim_m
        self.dim_y = dim_y
        self.N = N

        assert dim_m == len(Q)
        assert dim_y == len(R)

    def resample(self, particles, weights, method='systematic'):
        """
        Resample particles and weights if the effective number of particles is below N/2.

        Args:
        particles (numpy.ndarray): Current particle states.
        weights (numpy.ndarray): Current weights.
        method (str): Method for resampling. Default is 'systematic'. Other options are 'residual' and 'stratified'.

        Returns:
        particles (numpy.ndarray): Resampled particles.
        weights (numpy.ndarray): Resampled weights.
        """
        # Compute effective number of particles
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
        """
        Perform prediction step in Particle Filter.

        Args:
        particles (numpy.ndarray): Current particle states.
        weights (numpy.ndarray): Current weights.

        Returns:
        particles (numpy.ndarray): Predicted particles.
        weights (numpy.ndarray): Weights (unchanged).
        """
        for i, particle in enumerate(particles):
            particles[i] = mvn(self.f(particle), self.Q).rvs()

        return particles, weights

    def update(self, y, particles, weights):
        """
        Perform update step in Particle Filter.

        Args:
        y (numpy.ndarray): Current measurement vector.
        particles (numpy.ndarray): Predicted particle states from predict step.
        weights (numpy.ndarray): Weights from predict step.

        Returns:
        particles (numpy.ndarray): Particles (unchanged).
        weights (numpy.ndarray): Updated weights.
        """
        for i, (weight, particle) in enumerate(zip(weights, particles)):
            weights[i] *= mvn(self.h(particle), self.R).pdf(y)

        weights /= np.sum(weights)

        return particles, weights

    def filter(self, measurements, m=None, P=None, resampling_method='systematic', verbose=True):
        """
        Perform Particle Filtering on a sequence of measurements.

        Args:
        measurements (numpy.ndarray): Sequence of measurement vectors.
        m (numpy.ndarray): Initial state estimate vector. If None, initialized to zero vector. Default is None.
        P (numpy.ndarray): Initial state covariance matrix. If None, initialized to identity matrix. Default is None.
        resampling_method (str): Method for resampling. Default is 'systematic'.
        verbose (bool): If True, shows progress bar. Default is True.

        Returns:
        state_estimates (numpy.ndarray): Sequence of state estimate vectors.
        cov_estimates (numpy.ndarray): Sequence of state covariance matrices.
        particle_history (numpy.ndarray): Sequence of particle states for each time step.
        weights_history (numpy.ndarray): Sequence of weights for each time step.
        """
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

    def smoother(self, state_estimates, cov_estimates, particle_history, weights_history, verbose=True):
        """
        Perform Unscented Rauch-Tung-Striebel (URTS) Smoothing on a sequence of state estimates, covariance estimates, particle states, and weights.

        Args:
        state_estimates (numpy.ndarray): Sequence of state estimate vectors from filter step.
        cov_estimates (numpy.ndarray): Sequence of state covariance matrices from filter step.
        particle_history (numpy.ndarray): Sequence of particle states from filter step.
        weights_history (numpy.ndarray): Sequence of weights from filter step.
        verbose (bool): If True, shows progress bar. Default is True.

        Returns:
        state_estimates_smoothed (numpy.ndarray): Sequence of smoothed state estimate vectors.
        cov_estimates_smoothed (numpy.ndarray): Sequence of smoothed state covariance matrices.
        """
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
    

class PPF:
    """
    Parzen Particle Filter (PPF) class. Provides methods for prediction, update, filtering, resampling and smoothing of a non-linear system.
    """

    def __init__(self, f, h, F_jacobian, Q, R, dim_m=4, dim_y=2, N=50):
        """
        Initialize PPF object with given parameters.

        Args:
        f (function): Function for state transition.
        F_jacobian (function): Function to compute the Jacobian of f.
        h (function): Function for measurement.
        dim_m (int): Dimension of the state. Default is 4.
        dim_y (int): Dimension of the output. Default is 2.
        N (int): Number of particles. Default is 50.
        """
        self.f = f
        self.h = h
        self.F_jacobian = F_jacobian
        self.Q = Q
        self.R = R
        self.dim_m = dim_m
        self.dim_y = dim_y
        self.N = N

    def resample(self, weights, particles, particle_covs, dim_m = 4):

        new_particles = sample_from_mixture(weights, particles, particle_covs, len(weights))

        
        new_particle_covs = np.tile(np.eye(dim_m) / 10, (len(weights), 1, 1))

        new_weights = np.ones(len(weights)) / len(weights)

        return new_weights, new_particles, new_particle_covs

    def predict(self, particles, particle_covs, weights):
        """
        Perform prediction step in Parzen Particle Filter.

        Args:
        particles (numpy.ndarray): Current particle state_estimates.
        particle_covs (numpy.ndarray): Current particle covariance matrices.
        weights (numpy.ndarray): Current weights.

        Returns:
        particles (numpy.ndarray): Predicted particle state_estimates.
        particle_covs (numpy.ndarray): Predicted particle covariance matrices.
        weights (numpy.ndarray): Predicted weights.
        """
        for i, particle in enumerate(particles):
            #particles[i] = mvn(self.f(particle), self.Q).rvs() 
            particles[i] = mvn(self.f(particle), particle_covs[i]).rvs() 
            #particles[i] = self.f(particle)
            particle_covs[i] = self.F_jacobian(particle) @ particle_covs[i] @ self.F_jacobian(particle).T # (7.b) from Parzen paper

        return particles, particle_covs, weights

    def update(self, y, particles, particle_covs, weights):
        """
        Perform update step in Parzen Particle Filter.

        Args:
        y (numpy.ndarray): Measurement vector.
        particles (numpy.ndarray): Current particle state_estimates.
        particle_covs (numpy.ndarray): Current particle covariance matrices.
        weights (numpy.ndarray): Current weights.

        Returns:
        particles (numpy.ndarray): Updated particle state_estimates.
        particle_covs (numpy.ndarray): Updated particle covariance matrices.
        weights (numpy.ndarray): Updated weights.
        """
        for i, (weight, particle) in enumerate(zip(weights, particles)):
            weights[i] *= mvn(self.h(particle), self.R).pdf(y) * (np.linalg.det(self.F_jacobian(particle)) ** -1)

        weights /= np.sum(weights)

        return particles, particle_covs, weights
    
    def filter(self, measurements, m=None, P=None, verbose=True):#resampling_method='systematic', ):
        """
        Perform Parzen Particle Filtering on a sequence of measurements.
        """
        if m is None: m = np.zeros(self.dim_m)
        if P is None: P = np.eye(self.dim_m) / 10

        n = len(measurements)
        state_estimates = np.empty((n, self.dim_m))
        particle_history = np.empty((n, self.N, self.dim_m))
        particle_cov_history = np.empty((n, self.N, self.dim_m, self.dim_m))
        weights_history = np.empty((n, self.N))

        # Draw N samples from the prior
        particles = mvn(m, P).rvs(self.N)
        particle_covs = np.tile(P, (self.N, 1, 1))
        weights = np.ones(self.N) / self.N

        if verbose:
            iterator = tqdm(enumerate(measurements), total=n)
        else:
            iterator = enumerate(measurements)

        for k, y in iterator:
            particles, particle_covs, weights = self.predict(particles, particle_covs, weights)
            particles, particle_covs, weights = self.update(y, particles, particle_covs, weights)

            m = np.average(particles, weights=weights, axis=0)

            state_estimates[k] = m
            particle_history[k] = particles
            particle_cov_history[k] = particle_covs
            weights_history[k] = weights

            # Resample
            weights, particles, particle_covs  = self.resample(weights, particles, particle_covs)

        return state_estimates, particle_history, particle_cov_history, weights_history