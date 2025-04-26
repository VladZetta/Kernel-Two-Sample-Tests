import numpy as np
import scipy.spatial



def gaussian_rbf_kernel(X, Y, sigma=1.0):
    dists = scipy.spatial.distance.cdist(X, Y, 'sqeuclidean')
    return np.exp(-dists / (2 * sigma ** 2))

def laplacian_kernel(X, Y, sigma=1.0):
    dists = scipy.spatial.distance.cdist(X, Y, 'euclidean')
    return np.exp(-dists / sigma)

def matern_kernel(X, Y, nu=1.5, lengthscale=1.0):
    dists = scipy.spatial.distance.cdist(X, Y, 'euclidean')
    if nu == 0.5:
        return np.exp(-dists / lengthscale)
    elif nu == 1.5:
        sqrt3 = np.sqrt(3)
        return (1.0 + sqrt3 * dists / lengthscale) * np.exp(-sqrt3 * dists / lengthscale)
    elif nu == 2.5:
        sqrt5 = np.sqrt(5)
        return (1.0 + sqrt5 * dists / lengthscale + 5 * dists ** 2 / (3 * lengthscale ** 2)) * np.exp(-sqrt5 * dists / lengthscale)
    else:
        raise ValueError("Unsupported nu value for Matérn kernel.")

def polynomial_kernel(X, Y, degree=3, coef0=1.0):
    return (np.dot(X, Y.T) + coef0) ** degree
