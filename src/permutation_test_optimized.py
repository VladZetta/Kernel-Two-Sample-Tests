import numpy as np
from joblib import Parallel, delayed

def _single_permutation(combined, m, n, stat_fn, stat_observed, **kwargs):
    """Compute a single permutation for parallel execution"""
    # Permute the indices of the combined dataset
    perm_indices = np.random.permutation(m + n)
    permuted = combined[perm_indices]
    # Split back into two groups
    X_perm = permuted[:m]
    Y_perm = permuted[m:]
    stat_perm = stat_fn(X_perm, Y_perm, **kwargs)
    # Return 1 if permuted statistic is >= observed, 0 otherwise
    return int(stat_perm >= stat_observed)

def permutation_test_statistic(X, Y, stat_fn, num_permutations=1000, n_jobs=-1, **kwargs):
    """
    Perform a permutation test for a given test statistic using parallel processing.
    
    Parameters:
        X: array-like of shape (m, d)
            First group of samples.
        Y: array-like of shape (n, d)
            Second group of samples.
        stat_fn: function
            A function that takes (X, Y, **kwargs) and returns a test statistic.
        num_permutations: int, optional
            Number of permutations to perform (default is 1000).
        n_jobs: int, optional
            Number of jobs to run in parallel (-1 uses all cores).
        **kwargs: dict
            Additional keyword arguments that will be passed to stat_fn.
    
    Returns:
        p_value: float
            The p-value of the test computed as the proportion of permutations where
            the permuted statistic is greater than or equal to the observed statistic.
    """
    # Convert inputs to NumPy arrays
    X = np.asarray(X)
    Y = np.asarray(Y)
    
    m = X.shape[0]
    n = Y.shape[0]
    # Combine datasets for permutation
    combined = np.vstack((X, Y))
    # Compute the observed statistic
    stat_observed = stat_fn(X, Y, **kwargs)
    
    # Run permutations in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(_single_permutation)(combined, m, n, stat_fn, stat_observed, **kwargs)
        for _ in range(num_permutations)
    )
    
    # Count how many permutations had stat >= observed
    count = sum(results)
    
    p_value = (count + 1) / (num_permutations + 1)
    return p_value