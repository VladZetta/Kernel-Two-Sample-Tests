# Kernel Two-Sample Tests Project

This project implements and analyzes **kernel-based two-sample tests** to detect differences between probability distributions.  
We focus on **Maximum Mean Discrepancy (MMD)** and its computational variants, and validate methods on **synthetic** and **real-world datasets** like MNIST and CIFAR-10.

## 📂 Project Structure

- `data/` — Datasets for training and evaluation (e.g., MNIST, GAN outputs).
- `experiments/` — Scripts for running synthetic and real-data experiments.
- `figs/` and `figures/` — Plots and visualizations of results.
- `notebooks/` — Jupyter notebooks for analysis and prototyping.
- `results/` — Saved results from experiments and sensitivity analyses.
- `src/` — Core source code, including MMD computations and kernel definitions.
- `tests/` — Unit tests for validating core functions.

## ⚙️ Methods Implemented

- Standard MMD two-sample test with permutation-based p-value estimation
- **B-test**: Block-wise variance reduction
- **Linear-time MMD**: Random pairing approximation
- **Deep-kernel MMD**: Learning adaptive kernels with neural networks
- Sensitivity analysis: kernel type, bandwidth, sample size, dimensionality

## 🚀 Key Contributions

- Detailed empirical comparison of MMD variants.
- Evaluation of GAN-generated data quality using kernel two-sample testing.
- Practical guidelines for choosing kernels and tuning parameters.

## 🛠 How to Run

1. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2. Run synthetic experiments:
    ```bash
    python experiments/run_synthetic.py
    ```
3. Run real-data experiments:

    ### Permutation Analysis
    ```bash
    python experiments/real_data/run_experiment_optimized_perm.py
    ```
    This script computes p-values for different permutations of the data.

    ### MNIST Experiments
    ```bash
    python experiments/real_data/run_experiment_optimized.py
    ```
    Run optimization experiments on the MNIST dataset.

    ```bash
    python experiments/real_data/run_two_different_digits.py
    ```
    Run experiments comparing two different digits from the MNIST dataset.

    ### GAN Experiments
    ```bash
    # Train GANs
    python experiments/real_data/train_gan.py  # Train GAN on MNIST
    python experiments/real_data/train_gan_CIFAR.py  # Train GAN on CIFAR

    # Test GANs
    python experiments/real_data/run_gan_exp.py  # Test GAN on MNIST
    python experiments/real_data/run_gan_CIFAR.py  # Test GAN on CIFAR
    ```

Results and figures will be saved automatically inside `results/` and `figs/` folders.

## 📌 Notes

- Permutation testing is used for robust, finite-sample p-value estimation.
- GAN training scripts are included under `data/MNIST/raw/`.
- All figures can be fully reproduced from the saved experiment outputs.

---
