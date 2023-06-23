import numpy as np
from numpy.random import multivariate_normal
from scipy.linalg import cholesky, sqrtm


def make_constraint(polygon):
    """
    Function to create a new function for checking points within a specific polygon.

    Args:
    polygon (list): List of tuples representing the vertices of the polygon in counter-clockwise order

    Returns:
    function: Function that takes a point (x, y) and returns True if the point is in the polygon, False otherwise
    """

    num_vertices = len(polygon)

    def point_in_polygon(state):
        x, y, _, _ = state
        """
        Function to determine if a point is inside the specific polygon using the Ray Casting algorithm.

        Args:
        x, y (float): Coordinates of the point to test

        Returns:
        bool: True if the point is in the polygon, False otherwise
        """

        inside = False
        p1x, p1y = polygon[0]
        for i in range(1, num_vertices + 1):
            p2x, p2y = polygon[i % num_vertices]
            if min(p1y, p2y) < y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        x_intersects = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= x_intersects:
                        inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    return point_in_polygon

def line_search(m, h, c):
    start = m
    end = h

    while np.linalg.norm(start - end) > 1e-6:
        mid = (start + end) / 2
        if c(mid):
            start = mid
        else:
            end = mid

    return end

def nearest_point(x0, y0, c, precision=0.01, max_dist=1000):
    radius = 0

    while radius < max_dist:
        for theta in np.linspace(0, 2 * np.pi, 100):
            x = x0 + radius * np.cos(theta)
            y = y0 + radius * np.sin(theta)
            if c([x, y, 0, 0]):
                # Get the unit vector from (x, y) to (x0, y0)
                unit_vector = np.array([x0 - x, y0 - y]) / np.linalg.norm(np.array([x0 - x, y0 - y]))
                x_tmp = x + unit_vector[0] * precision
                y_tmp = y + unit_vector[1] * precision                
                
                return line_search(np.array([x_tmp, y_tmp, 0, 0]), np.array([x, y, 0, 0]), c)[:2]
                
        radius += precision


### Unscented Transform ###

def get_weights(n, alpha=0.5, beta=2, kappa=0, method='merwe'):

    """ 
    Returns the weights for the unscented transform

    Parameters
    ----------
    n : int
        Dimension of the state space
    alpha : float
        Spread of the sigma points around the mean
    beta : float
        Incorporates prior knowledge of the distribution of the mean
    kappa : float   
        Secondary scaling parameter
    method : str
        Method to compute the weights (merwe, julier, simplex)

    Returns
    -------
    Wm : numpy array
        Weights for the mean
    Wc : numpy array
        Weights for the covariance
    """

    if method == 'merwe':
         
        lambda_ = alpha**2 * (n + kappa) - n

        Wc = np.full(2*n + 1, 1 / (2*(n + lambda_)))
        Wm = Wc.copy()

        Wc[0] = lambda_ / (n + lambda_) + (1 - alpha**2 + beta)
        Wm[0] = lambda_ / (n + lambda_)

    elif method == 'julier':
         
        Wm = np.full(2*n + 1, 1 / (2*(n + kappa)))
        Wm[0] = kappa / (n + kappa)

        Wc = Wm.copy()

    elif method == 'simplex':
             
        Wm = np.full(n + 1, 1 / (n + 1))
        Wc = Wm.copy()

    else:
        raise ValueError('Invalid method')
    
    return Wm, Wc


def get_sigmas(mean, cov, alpha, kappa, method='merwe'):

    """
    Returns the sigma points for the unscented transform

    Parameters
    ----------
    mean : numpy array
        Mean of the state
    cov : numpy array
        Covariance of the state
    alpha : float
        Spread of the sigma points around the mean
    kappa : float
        Secondary scaling parameter
    method : str
        Method to construct the sigma points (merwe, julier, simplex)

    Returns
    -------
    sigmas : numpy array
        Sigma points
    """

    def sqrt(A):
        try:
            return cholesky(A)
        except np.linalg.LinAlgError:
            return sqrtm(A)

    n = len(mean)

    if method in ['merwe', 'julier']:

        sigmas = np.empty((2*n + 1, n))

        if method == 'merwe':

            lambda_ = alpha**2 * (n + kappa) - n
            U = sqrt((n + lambda_) * cov)

        elif method == 'julier':
                 
            U = sqrt((n + kappa) * cov)

        sigmas[0] = mean
        for i in range(n):
            sigmas[i + 1] = mean + U[i]
            sigmas[n + i + 1] = mean - U[i]

    elif method == 'simplex':

        lambda_ = n / (n + 1)

        U = sqrt(cov)

        Istar = np.array([[-1/np.sqrt(2*lambda_), 1/np.sqrt(2*lambda_)]])

        for d in range(2, n+1):
            row = np.ones((1, Istar.shape[1] + 1)) * 1. / np.sqrt(lambda_*d*(d + 1))
            row[0, -1] = -d / np.sqrt(lambda_ * d * (d + 1))
            Istar = np.r_[np.c_[Istar, np.zeros((Istar.shape[0]))], row]

        I = np.sqrt(n)*Istar
        scaled_unitary = U.dot(I)

        mean = np.array(mean).reshape(n, 1) 
        sigmas = np.subtract(mean, -scaled_unitary).T   

    else:
        raise ValueError('Unknown method')      

    return sigmas


def unscented_transform(mean, cov, f, alpha=.5, beta=2, kappa=0, method='merwe'):

    """
    Returns the unscented transform of a function

    Parameters
    ----------
    mean : numpy array
        Mean of the state
    cov : numpy array
        Covariance of the state
    f : function
        Non-linear transformation function
    alpha : float
        Spread of the sigma points around the mean
    beta : float
        Incorporates prior knowledge of the distribution of the mean
    kappa : float
        Secondary scaling parameter
    method : str
        Method to construct the sigma points (merwe, julier, simplex)

    Returns
    -------
    mean : numpy array
        Mean of the transformed function
    cov : numpy array
        Covariance of the transformed function
    sigmas : numpy array
        Sigma points
    sigmas_f : numpy array
        Sigma points transformed
    """

    n = len(mean)

    Wm, Wc = get_weights(n, alpha, beta, kappa, method)
    sigmas = get_sigmas(mean, cov, alpha, kappa, method)
    
    n_sigmas = len(sigmas)
    sigmas_f = np.empty((n_sigmas, n))

    for i in range(n_sigmas):
        sigmas_f[i] = f(sigmas[i, 0], sigmas[i, 1])
    
    mean = np.sum(Wm * sigmas_f.T, axis=1)

    cov = np.zeros((n, n))
    for i in range(n_sigmas):
        y = sigmas_f[i] - mean
        cov += Wc[i] * np.outer(y, y)

    return mean, cov, sigmas, sigmas_f