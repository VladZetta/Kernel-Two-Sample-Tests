
import numpy as np

def permutation_test_statistic(X, Y, stat_fn, num_permutations=1000, **kwargs):
    """
    Perform a permutation test for a given test statistic.
    
    Parameters:
        X: array-like of shape (m, d)
            First group of samples.
        Y: array-like of shape (n, d)
            Second group of samples.
        stat_fn: function
            A function that takes (X, Y, **kwargs) and returns a test statistic.
        num_permutations: int, optional
            Number of permutations to perform (default is 1000).
        **kwargs: dict
            Additional keyword arguments that will be passed to stat_fn.
    
    Returns:
        p_value: float
            The p-value of the test computed as the proportion of permutations where
            the permuted statistic is greater than or equal to the observed statistic.
    """
    # Convert inputs to NumPy arrays.
    X = np.asarray(X)
    Y = np.asarray(Y)
    
    m = X.shape[0]
    n = Y.shape[0]
    # Combine datasets for permutation.
    combined = np.vstack((X, Y))
    # Compute the observed statistic.
    stat_observed = stat_fn(X, Y, **kwargs)
    count = 0
    
    for _ in range(num_permutations):
        # Permute the indices of the combined dataset.
        perm_indices = np.random.permutation(m + n)
        permuted = combined[perm_indices]
        # Split back into two groups.
        X_perm = permuted[:m]
        Y_perm = permuted[m:]
        stat_perm = stat_fn(X_perm, Y_perm, **kwargs)
        if stat_perm >= stat_observed:
            count += 1
    
    p_value = (count + 1) / (num_permutations + 1)
    return p_value
