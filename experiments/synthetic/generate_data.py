import numpy as np

def generate_gaussian_shift(n, d, delta):
    """
    Generate two synthetic datasets X and Y drawn from multivariate normals:
    X ~ N(0, I_d), Y ~ N(delta * 1, I_d), both of shape (n, d).
    """
    X = np.random.normal(loc=0.0, scale=1.0, size=(n, d))
    Y = np.random.normal(loc=delta, scale=1.0, size=(n, d))
    return X, Y
