import numpy as np
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel, polynomial_kernel, laplacian_kernel
from scipy.spatial.distance import pdist, squareform
import numba
from .permutation_test_optimized import permutation_test_statistic

def median_heuristic(X, Y):
    """
    Compute the median heuristic for bandwidth selection efficiently.
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    Z = np.vstack([X, Y])
    # Use pdist for efficient pairwise distance computation
    dists = pdist(Z, metric='euclidean')
    median_val = np.median(dists[dists > 0])
    return median_val

def compute_mmd_stat(X, Y, kernel='rbf', bandwidth='median', preprocess=False, **kernel_params):
    """
    Helper function to compute the squared MMD statistic between X and Y using scikit-learn kernels.
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    
    if preprocess:
        # Include any additional preprocessing here if needed
        pass

    m = X.shape[0]
    n = Y.shape[0]

    if kernel == 'rbf':
        if bandwidth == 'median':
            sigma = median_heuristic(X, Y)
        elif isinstance(bandwidth, (float, int)):
            sigma = float(bandwidth)
        else:
            raise ValueError("Bandwidth must be 'median' or a numeric value.")
        
        # Convert sigma to gamma parameter for scikit-learn (gamma = 1/(2*sigma²))
        gamma = 1.0 / (2 * sigma**2)
        K_xx = rbf_kernel(X, X, gamma=gamma)
        K_yy = rbf_kernel(Y, Y, gamma=gamma)
        K_xy = rbf_kernel(X, Y, gamma=gamma)

    elif kernel == 'linear':
        K_xx = linear_kernel(X, X)
        K_yy = linear_kernel(Y, Y)
        K_xy = linear_kernel(X, Y)

    elif kernel == 'poly':
        degree = kernel_params.get('degree', 3)
        coef0 = kernel_params.get('coef0', 1)
        K_xx = polynomial_kernel(X, X, degree=degree, coef0=coef0)
        K_yy = polynomial_kernel(Y, Y, degree=degree, coef0=coef0)
        K_xy = polynomial_kernel(X, Y, degree=degree, coef0=coef0)

    elif kernel == 'laplace':
        # gamma either numeric or use 1/median pairwise distance
        gamma = (1 / median_heuristic(X, Y)) if bandwidth == 'median' else float(bandwidth)
        K_xx = laplacian_kernel(X, X, gamma=gamma)
        K_yy = laplacian_kernel(Y, Y, gamma=gamma)
        K_xy = laplacian_kernel(X, Y, gamma=gamma)

    else:
        raise NotImplementedError(f"Kernel '{kernel}' not supported.")
    
    # Compute unbiased estimators (remove self-similarities)
    # Vectorized operations with element-wise multiplication are faster
    mask_xx = np.ones((m, m), dtype=bool)
    np.fill_diagonal(mask_xx, False)
    sum_K_xx = np.sum(K_xx[mask_xx]) / (m * (m - 1))
    
    mask_yy = np.ones((n, n), dtype=bool)
    np.fill_diagonal(mask_yy, False)
    sum_K_yy = np.sum(K_yy[mask_yy]) / (n * (n - 1))
    
    sum_K_xy = np.sum(K_xy) / (m * n)
    
    mmd_squared = sum_K_xx + sum_K_yy - 2 * sum_K_xy
    return mmd_squared

def mmd_test(X, Y, kernel='rbf', bandwidth='median', preprocess=False, 
             return_p_value=False, num_permutations=1000, n_jobs=-1, **kernel_params):
    """
    Compute the MMD statistic between two datasets and, optionally, a permutation-based p-value.
    
    Parameters:
        X, Y: Input datasets
        kernel: Kernel type ('rbf', 'linear', 'poly', 'laplace')
        bandwidth: Bandwidth parameter for the kernel
        preprocess: Whether to preprocess the data
        return_p_value: Whether to return the p-value
        num_permutations: Number of permutations for p-value computation
        n_jobs: Number of parallel jobs
        **kernel_params: Additional parameters for the kernel (e.g., degree for poly kernel)
    
    Returns:
        If return_p_value is False: a float representing the squared MMD statistic.
        If return_p_value is True: a tuple (mmd_squared, p_value).
    """
    # Calculate the observed MMD statistic
    stat_observed = compute_mmd_stat(X, Y, kernel, bandwidth, preprocess, **kernel_params)
    
    if not return_p_value:
        return stat_observed
    
    # Import the permutation test function
    
    
    # Compute the p-value using the parallel permutation test
    p_value = permutation_test_statistic(
        X, Y, stat_fn=compute_mmd_stat, num_permutations=num_permutations, n_jobs=n_jobs,
        kernel=kernel, bandwidth=bandwidth, preprocess=preprocess, **kernel_params
    )
    
    return stat_observed, p_value