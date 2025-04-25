# run_experiment.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from experiments.real_data.preprocess_mnist import load_mnist
from src.mmd import mmd_test

KERNEL_EXPERIMENTS = [
    {'kernel':'rbf',    'bandwidth':'median'},
    {'kernel':'rbf',    'bandwidth':1.0},
    {'kernel':'linear', 'bandwidth':None},
    {'kernel':'poly',   'bandwidth':None, 'degree':2, 'coef0':0},
    {'kernel':'poly',   'bandwidth':None, 'degree':3, 'coef0':1},
    {'kernel':'laplace','bandwidth':'median'},
    {'kernel':'laplace','bandwidth':0.5},
]


def subsample_data(data, labels, digit, max_samples=1000):
    # Filter the data for the specific digit.
    digit_data = data[labels == digit]
    # If there are more samples than max_samples, randomly choose a subset.
    if digit_data.shape[0] > max_samples:
        indices = np.random.choice(digit_data.shape[0], max_samples, replace=False)
        digit_data = digit_data[indices]
    return digit_data

FIG_DIR = "figures"
import time
def main():
    time_start = time.time()
    print("Starting MMD experiment...")

    # ensure output directory exists
    os.makedirs(FIG_DIR, exist_ok=True)

    #load and subsample
    data, labels = load_mnist()
    group1 = subsample_data(data, labels, digit=3, max_samples=1000)
    group2 = subsample_data(data, labels, digit=8, max_samples=1000)

    print(f"Group1 (digit=3): {group1.shape[0]} samples")
    print(f"Group2 (digit=8): {group2.shape[0]} samples")

    #un MMD tests over all kernels
    results = []
    for cfg in KERNEL_EXPERIMENTS:
        params = cfg.copy()
        name   = params.pop('kernel')
        bw     = params.pop('bandwidth')
        stat, pval = mmd_test(
            group1, group2,
            kernel=name,
            bandwidth=bw,
            return_p_value=True,
            num_permutations=10,
            **params
        )
        print(f"{name:7s} | bw={str(bw):6s} → MMD²={stat:.4e}, p={pval:.4f}")
        results.append({
            'kernel':    name,
            'bandwidth': bw,
            'mmd2':       stat,
            'p_value':   pval
        })
    end_time = time.time()  
    print(f"Total time: {end_time - time_start:.2f} seconds")
    
    # 3) summarize in DataFrame
    df = pd.DataFrame(results)
    print("\nSummary:\n", df)

    # 4) bar plot of MMD²
    plt.figure(figsize=(8,4))
    plt.bar(df['kernel'], df['mmd2'])
    plt.ylabel("MMD² statistic")
    plt.title("MMD² by kernel")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "mmd2_by_kernel.png"))
    plt.close()

    # 5) bar plot of p-value
    plt.figure(figsize=(8,4))
    plt.bar(df['kernel'], df['p_value'])
    plt.ylabel("permutation p-value")
    plt.title("Significance by kernel")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "pvalue_by_kernel.png"))
    plt.close()

    print(f"Figures saved to '{FIG_DIR}/'")

if __name__ == '__main__':
    main()
