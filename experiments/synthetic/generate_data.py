import numpy as np

def generate_gaussian_shift(n, d, delta):
    """
    Generate two synthetic datasets X and Y drawn from multivariate normals:
    X ~ N(0, I_d), Y ~ N(delta * 1, I_d), both of shape (n, d).
    """
    X = np.random.normal(loc=0.0, scale=1.0, size=(n, d))
    Y = np.random.normal(loc=delta, scale=1.0, size=(n, d))
    return X, Y

def generate_gaussian_mixture(n, d, num_components=2, weights=None, mean_shift=1.0, 
                             component_variance=0.3, mixture_difference='components'):
    """
    Generate two synthetic datasets X and Y from Gaussian mixtures.
    
    Parameters:
        n (int): Number of samples per dataset
        d (int): Dimensionality of the data
        num_components (int): Number of mixture components
        weights (array-like): Component weights (if None, equal weights are used)
        mean_shift (float): Distance between component means
        component_variance (float): Variance of each component
        mixture_difference (str): Type of difference between mixtures:
            - 'components': X and Y have different component means
            - 'weights': X and Y have same components but different weights
            - 'variance': X and Y have same components but different variances
    
    Returns:
        X, Y: Two datasets from different Gaussian mixtures
    """
    # For weights experiment, use non-uniform weights if not specified
    if weights is None:
        if mixture_difference == 'weights':
            # Use non-uniform weights to make the experiment meaningful
            weights = np.array([0.6, 0.3, 0.1][:num_components])
            # Normalize in case num_components is not 3
            weights = weights / np.sum(weights)
        else:
            weights = np.ones(num_components) / num_components
    else:
        weights = np.array(weights) / np.sum(weights)
    
    # Generate component assignments for dataset X
    component_indices_X = np.random.choice(num_components, size=n, p=weights)
    
    # Create more distinctive component means in 2D space
    component_means = np.zeros((num_components, d))
    
    # Arrange components in a grid or circle pattern for better visualization
    if num_components <= 4:
        # Place in corners of a square for up to 4 components
        positions = [(-1, -1), (1, -1), (-1, 1), (1, 1)]
        for i in range(min(num_components, 4)):
            if d >= 2:
                component_means[i, 0] = positions[i][0] * mean_shift * 2
                component_means[i, 1] = positions[i][1] * mean_shift * 2
            else:
                # If only 1D, place along the line
                component_means[i, 0] = (i - (num_components-1)/2) * mean_shift * 3
    else:
        # For more components, arrange in a circle
        for i in range(num_components):
            angle = 2 * np.pi * i / num_components
            if d >= 2:
                component_means[i, 0] = np.cos(angle) * mean_shift * 3
                component_means[i, 1] = np.sin(angle) * mean_shift * 3
            else:
                component_means[i, 0] = (i - (num_components-1)/2) * mean_shift * 2
    
    # Generate X from components
    X = np.zeros((n, d))
    for i in range(n):
        comp_idx = component_indices_X[i]
        X[i] = np.random.normal(loc=component_means[comp_idx], 
                               scale=np.sqrt(component_variance), size=d)
    
    # Generate Y based on the specified difference
    if mixture_difference == 'components':
        # Y has components with different means (shifted by mean_shift/2 in each dimension)
        Y_means = component_means.copy()
        # Apply a consistent shift to all components
        shift_vector = np.ones(d) * mean_shift / 2
        for i in range(num_components):
            Y_means[i] += shift_vector
        
        component_indices_Y = np.random.choice(num_components, size=n, p=weights)
        Y = np.zeros((n, d))
        for i in range(n):
            comp_idx = component_indices_Y[i]
            Y[i] = np.random.normal(loc=Y_means[comp_idx], 
                                  scale=np.sqrt(component_variance), size=d)
            
    elif mixture_difference == 'weights':
        # Y has same components but different mixing weights
        y_weights = weights.copy()
        # Modify weights to create a different mixture - make the change more dramatic
        if num_components >= 2:
            # Create distinctly different weights based on magnitude parameter
            # Increase the difference with larger mean_shift
            weight_scale = min(0.8, mean_shift/2)  # Cap at 0.8 to keep valid probabilities
            
            # Reverse dominant components - most common becomes least common
            y_weights = weights[::-1]
            
            # Scale the difference even more to ensure it's detectable
            if num_components >= 3:
                # Make largest weight smaller and smallest weight larger
                max_idx = np.argmax(weights)
                min_idx = np.argmin(weights)
                
                # Apply modifications based on mean_shift (difference magnitude)
                y_weights[max_idx] = max(0.1, weights[max_idx] - weight_scale)
                y_weights[min_idx] = min(0.9, weights[min_idx] + weight_scale)
                
                # Renormalize
                y_weights = y_weights / np.sum(y_weights)
        else:
            y_weights = weights  # No change possible with 1 component
        
        component_indices_Y = np.random.choice(num_components, size=n, p=y_weights)
        Y = np.zeros((n, d))
        for i in range(n):
            comp_idx = component_indices_Y[i]
            Y[i] = np.random.normal(loc=component_means[comp_idx], 
                                  scale=np.sqrt(component_variance), size=d)
            
    elif mixture_difference == 'variance':
        # Y has same components but different variance - make the difference larger
        component_indices_Y = np.random.choice(num_components, size=n, p=weights)
        Y = np.zeros((n, d))
        for i in range(n):
            comp_idx = component_indices_Y[i]
            # Increase variance for Y more dramatically
            Y[i] = np.random.normal(loc=component_means[comp_idx], 
                                  scale=np.sqrt(component_variance * 4), size=d)
    else:
        raise ValueError("mixture_difference must be 'components', 'weights', or 'variance'")
    
    return X, Y
