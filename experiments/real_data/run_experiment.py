# run_experiment.py
import numpy as np
from experiments.real_data.preprocess_mnist import load_mnist
from src.mmd import mmd_test  # Import MMD function


def subsample_data(data, labels, digit, max_samples=1000):
    # Filter the data for the specific digit.
    digit_data = data[labels == digit]
    # If there are more samples than max_samples, randomly choose a subset.
    if digit_data.shape[0] > max_samples:
        indices = np.random.choice(digit_data.shape[0], max_samples, replace=False)
        digit_data = digit_data[indices]
    return digit_data


def main():
    data, labels = load_mnist()
    
    # Choose two subgroups based on digit labels (e.g., 3 vs 8)
    subgroup_digit1 = 3
    subgroup_digit2 = 3
    
    group1 = subsample_data(data, labels, subgroup_digit1, max_samples=1000)
    group2 = subsample_data(data, labels, subgroup_digit2, max_samples=1000)
    
    # Print basic information for verification
    print(f"Group for digit {subgroup_digit1} has {group1.shape[0]} samples.")
    print(f"Group for digit {subgroup_digit2} has {group2.shape[0]} samples.")
    
    # Run the MMD test.
    statistic, p_value = mmd_test(group1, group2, kernel='rbf', bandwidth='median', preprocess=False, return_p_value=True, num_permutations=100)
    
    print("MMD Statistic:", statistic)
    print("p-value:", p_value)

if __name__ == '__main__':
    main()
