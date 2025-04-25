# run_experiment.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from experiments.real_data.preprocess_mnist import load_mnist
from src.mmd_optimized import mmd_test

from joblib import Parallel, delayed

# your kernel configs
KERNEL_EXPERIMENTS = [
    {'kernel':'rbf',    'bandwidth':'median'},
    {'kernel':'rbf',    'bandwidth':1.0},
    {'kernel':'linear', 'bandwidth':None},
    {'kernel':'poly',   'bandwidth':None, 'degree':2, 'coef0':0},
    {'kernel':'poly',   'bandwidth':None, 'degree':3, 'coef0':1},
    {'kernel':'laplace','bandwidth':'median'},
    {'kernel':'laplace','bandwidth':0.5},
]

# how many permutations to try
PERM_COUNTS = [10, 50, 100 , 200 ]

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

def subsample_data(data, labels, digit, max_samples=1000):
    idx = np.where(labels == digit)[0]
    if len(idx) > max_samples:
        idx = np.random.choice(idx, max_samples, replace=False)
    return data[idx]

def run_one(cfg, group1, group2, num_permutations):
    params = cfg.copy()
    name = params.pop('kernel')
    bw   = params.pop('bandwidth')
    stat, pval = mmd_test(
        group1, group2,
        kernel=name,
        bandwidth=bw,
        return_p_value=True,
        num_permutations=num_permutations,
        n_jobs=1,
        **params
    )
    return {
        'kernel': name,
        'bandwidth': bw,
        'num_permutations': num_permutations,
        'mmd2': stat,
        'p_value': pval
    }

def main():
    data, labels = load_mnist()
    g1 = subsample_data(data, labels, digit=3)
    g2 = subsample_data(data, labels, digit=8)

    records = []
    for B in PERM_COUNTS:
        # parallelize over kernels
        batch = Parallel(n_jobs=-1)(
            delayed(run_one)(cfg, g1, g2, B)
            for cfg in KERNEL_EXPERIMENTS
        )
        records.extend(batch)
        print(f"Done B={B}")

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(FIG_DIR, "mmd_sweep.csv"), index=False)
    print(df.head())

    # 2) Plot p-value vs #permutations for each kernel
    plt.figure(figsize=(8,5))
    for (k, bw), sub in df.groupby(['kernel','bandwidth']):
        plt.plot(sub['num_permutations'], sub['p_value'],
                 marker='o', label=f"{k} ({bw})")
    plt.xscale('log')
    plt.xlabel("Number of permutations (log scale)")
    plt.ylabel("permutation p-value")
    plt.title("p-value convergence vs #permutations")
    plt.legend(loc='best', fontsize='small')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "pvalue_vs_permutations.png"))
    plt.close()

    # 3) Plot MMD² vs #permutations
    plt.figure(figsize=(8,5))
    for (k, bw), sub in df.groupby(['kernel','bandwidth']):
        plt.plot(sub['num_permutations'], sub['mmd2'],
                 marker='o', label=f"{k} ({bw})")
    plt.xscale('log')
    plt.xlabel("Number of permutations (log scale)")
    plt.ylabel("MMD² statistic")
    plt.title("MMD² stability vs #permutations")
    plt.legend(loc='best', fontsize='small')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "mmd2_vs_permutations.png"))
    plt.close()

    print("Plots written to figures/")

if __name__ == "__main__":
    main()
