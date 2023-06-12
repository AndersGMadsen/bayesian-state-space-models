import scipy.stats as stats
import numpy as np
from numpy.random import multivariate_normal
import matplotlib.pyplot as plt

from scipy.stats.mstats import mquantiles
import matplotlib.patches as patches

from scipy.linalg import eigh, cholesky, sqrtm
from scipy.stats import chi2


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


def illustrate_unscented_transform(ut_dict, xs, ys, xs_nl, ys_nl, title='title'):

    """
    Illustrates the unscented transform

    Parameters
    ----------
    ut_dict : dict
        Dictionary containing the unscented transform
    xs : numpy array
        x coordinates of the points
    ys : numpy array
        y coordinates of the points
    xs_nl : numpy array
        x coordinates of the transformed points
    ys_nl : numpy array
        y coordinates of the transformed points
    title : str
        Title of the plot
    """

    def get_ellipsis(cov):

        eigvals, eigvecs = np.linalg.eig(cov)
        eigvals = 2 * np.sqrt(eigvals)

        theta = np.linspace(0, 2*np.pi, 1000);
        ellipsis = (eigvals[None,:] * eigvecs) @ [np.sin(theta), np.cos(theta)]

        return ellipsis

    methods = ['merwe', 'julier', 'simplex']

    fig, ax = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)

    alpha = .4

    for i, method in enumerate(methods):
        mean = ut_dict[method]['mean']
        cov = ut_dict[method]['cov']
        sigmas = ut_dict[method]['sigmas']
        sigmas_f = ut_dict[method]['sigmas_f']

        ax[i].scatter(xs_nl, ys_nl, c='g', s=30, alpha=alpha, label='Transformed Points')
        ax[i].scatter(xs, ys, c='b', s=30, alpha=alpha, label='Original Points')

        # transformed points
        ax[i].scatter(xs_nl.mean(), ys_nl.mean(), c='purple', s=50, label='Mean of Transformed Points', marker='x')

        ellipsis = get_ellipsis(np.cov([xs_nl, ys_nl]))
        ax[i].plot(xs_nl.mean() + ellipsis[0,:], ys_nl.mean() + ellipsis[1,:], c='purple', label='Covariance of Transformed Points')

        # sigma points
        ax[i].scatter(sigmas[:,0], sigmas[:,1], c='r', s=50, label='Sigma Points')
        ax[i].scatter(sigmas_f[:,0], sigmas_f[:,1], c='k', s=50, label='Transformed Sigma Points')

        # UT estimate
        ax[i].scatter(mean[0], mean[1], c='y', s=100, label='UT Mean', marker='*')

        ellipsis = get_ellipsis(cov)
        ax[i].plot(mean[0] + ellipsis[0,:], mean[1] + ellipsis[1,:], c='y', label='UT Covariance')

        ax[i].set_title(method, fontsize=16)
        ax[i].set_xlabel('x')
        ax[i].set_ylabel('y')

    ax[0].legend(fontsize=12)

    fig.suptitle(title, fontsize=24)

    plt.tight_layout()
    plt.show()