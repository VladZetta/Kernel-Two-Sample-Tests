# src/mmd.py

import numpy as np
from scipy.spatial.distance import pdist
# No need to import permutation_test here until we actually use it.

def gaussian_rbf_kernel(X, Y, sigma):
    """
    Compute the Gaussian RBF kernel matrix between samples in X and Y.
    """
    # Ensure inputs are numpy arrays.
    X = np.asarray(X)
    Y = np.asarray(Y)
    
    # Compute squared norms.
    X_norm = np.sum(X ** 2, axis=1)
    Y_norm = np.sum(Y ** 2, axis=1)
    
    # Calculate squared Euclidean distances using broadcasting.
    distances = X_norm[:, None] + Y_norm[None, :] - 2 * np.dot(X, Y.T)
    # Compute the kernel matrix.
    K = np.exp(-distances / (2 * sigma ** 2))
    return K

def median_heuristic(X, Y):
    """
    Compute the median heuristic for bandwidth selection using a memory-efficient method.
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    Z = np.vstack([X, Y])
    # Use pdist for efficient computation.
    dists = pdist(Z, metric='euclidean')
    median_val = np.median(dists[dists > 0])
    return median_val

def compute_mmd_stat(X, Y, kernel='rbf', bandwidth='median', preprocess=False):
    """
    Helper function to compute the squared MMD statistic between X and Y.
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    
    if preprocess:
        # Include any additional preprocessing here if needed.
        pass

    if kernel == 'rbf':
        if bandwidth == 'median':
            sigma = median_heuristic(X, Y)
        elif isinstance(bandwidth, (float, int)):
            sigma = float(bandwidth)
        else:
            raise ValueError("Bandwidth must be 'median' or a numeric value.")
        K_xx = gaussian_rbf_kernel(X, X, sigma)
        K_yy = gaussian_rbf_kernel(Y, Y, sigma)
        K_xy = gaussian_rbf_kernel(X, Y, sigma)
    else:
        raise NotImplementedError("Currently, only the 'rbf' kernel is supported.")
    
    m = X.shape[0]
    n = Y.shape[0]
    # Compute unbiased estimators (remove self-similarities).
    sum_K_xx = (np.sum(K_xx) - np.sum(np.diag(K_xx))) / (m * (m - 1))
    sum_K_yy = (np.sum(K_yy) - np.sum(np.diag(K_yy))) / (n * (n - 1))
    sum_K_xy = np.sum(K_xy) / (m * n)
    
    mmd_squared = sum_K_xx + sum_K_yy - 2 * sum_K_xy
    return mmd_squared

def mmd_test(X, Y, kernel='rbf', bandwidth='median', preprocess=False, 
             return_p_value=False, num_permutations=1000):
    """
    Compute the MMD statistic between two datasets and, optionally, a permutation-based p-value.
    
    Returns:
        If return_p_value is False: a float representing the squared MMD statistic.
        If return_p_value is True: a tuple (mmd_squared, p_value).
    """
    # Calculate the observed MMD statistic.
    stat_observed = compute_mmd_stat(X, Y, kernel, bandwidth, preprocess)
    
    if not return_p_value:
        return stat_observed
    
    # Import the permutation test function.
    from src.permutation_test import permutation_test_statistic
    # Compute the p-value using the permutation test.
    p_value = permutation_test_statistic(
        X, Y, stat_fn=compute_mmd_stat, num_permutations=num_permutations,
        kernel=kernel, bandwidth=bandwidth, preprocess=preprocess
    )
    
    return stat_observed, p_value


# Example usage (for testing purposes):
if __name__ == '__main__':
    np.random.seed(0)
    # Create synthetic data samples.
    X_sample = np.random.normal(0, 1, (100, 10))
    Y_sample = np.random.normal(0.5, 1, (100, 10))
    
    stat, p_val = mmd_test(X_sample, Y_sample, kernel='rbf', bandwidth='median',
                           preprocess=False, return_p_value=True, num_permutations=500)
    print("Squared MMD Statistic:", stat)
    print("p-value:", p_val)