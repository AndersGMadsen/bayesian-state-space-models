# Implementation by Anders Gjølbye Madsen


import numpy as np

def systematic_resampling(weights):
    """Systematic Resampling
    
    Parameters:
    weights (np.array): Array of weights, they should be normalized.

    Returns:
    indexes (np.array): Array of index values for resampled particles.
    """
    
    # Get the number of particles
    N = len(weights)
    
    # Generate random positions within each subinterval of [0, 1)
    positions = (np.arange(N) + np.random.random()) / N

    # Array to hold resampled indexes
    indexes = np.zeros(N, 'i')

    # Compute cumulative sum of weights
    cumulative_sum = np.cumsum(weights)

    # Initialize pointers
    i, j = 0, 0

    # Resampling loop
    while i < N:
        # If position is less than cumulative sum, assign index
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        # If position is larger, move to next particle
        else:
            j += 1

    return indexes


def residual_resampling(weights):
    """Residual Resampling
    
    Parameters:
    weights (np.array): Array of weights, they should be normalized.

    Returns:
    indexes (np.array): Array of index values for resampled particles.
    """
    
    # Get the number of particles
    N = len(weights)

    # Array to hold resampled indexes
    indexes = np.zeros(N, 'i')

    # Step 1: deterministic resampling
    # take integer part of weights
    num_copies = (N * weights).astype(int)
    k = 0
    for i in range(N):
        for _ in range(num_copies[i]): # make n copies
            indexes[k] = i
            k += 1

    # Step 2: stochastic universal resampling on residuals
    # get fractional part
    residual = weights - num_copies
    
    # normalize residuals
    residual /= np.sum(residual)        

    # Compute cumulative sum of residuals
    cumulative_sum = np.cumsum(residual)
    cumulative_sum[-1] = 1.  # avoid round-off error

    # draw sample from uniform distribution
    u = np.random.uniform(0, 1/N)

    # Generate uniform random numbers within each subinterval of [0, 1)
    u = (np.arange(N) + u) / N

    # Initialize pointers
    i, j = 0, 0

    # Resampling loop
    while (i < N) and (j < N):
        while u[i] > cumulative_sum[j]:
            j += 1
        while i < N and u[i] <= cumulative_sum[j]:
            indexes[i] = j
            i += 1
            
    return indexes


def stratified_resampling(weights):
    """Stratified Resampling
    
    Parameters:
    weights (np.array): Array of weights, they should be normalized.

    Returns:
    indexes (np.array): Array of index values for resampled particles.
    """
    
    # Get the number of particles
    N = len(weights)

    # Generate random positions within each subinterval of [0, 1)
    positions = (np.random.random(N) + np.arange(N)) / N

    # Array to hold resampled indexes
    indexes = np.zeros(N, 'i')

    # Compute cumulative sum of weights
    cumulative_sum = np.cumsum(weights)

    # Initialize pointers
    i, j = 0, 0

    # Resampling loop
    while i < N:
        # If position is less than cumulative sum, assign index
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        # If position is larger, move to next particle
        else:
            j += 1
            
    return indexes

def sample_from_mixture(weights, means, covariances, num_samples):
    """
    Generate samples from a mixture of Gaussians.

    Parameters:
        means (list of numpy arrays): Mean vectors of the Gaussians.
        covariances (list of numpy arrays): Covariance matrices of the Gaussians.
        weights (list of floats): Mixing weights of the Gaussians.
        num_samples (int): Number of samples to generate.

    Returns:
        numpy array: Generated samples.
    """
    num_gaussians = len(means)
    samples = np.zeros((num_samples, len(means[0])))

    for i in range(num_samples):
        # Select a Gaussian component based on the mixing weights
        gaussian_index = np.random.choice(range(num_gaussians), p=weights)

        # Generate a sample from the selected Gaussian component
        sample = np.random.multivariate_normal(means[gaussian_index], covariances[gaussian_index])
        samples[i] = sample

    return samples