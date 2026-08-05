"""
مرحله 6: واریانس توضیح داده شده (Explained Variance)
"""
import numpy as np
import matplotlib.pyplot as plt

from config import OUT

def run(eigenvalues, n, results):
    print()
    print("=" * 70)
    print("Step 6 - Explained Variance")
    print("=" * 70)

    explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
    cumulative_variance = np.cumsum(explained_variance_ratio)

    n_components_90 = np.argmax(cumulative_variance >= 0.90) + 1
    ratio_of_64 = n_components_90 / n * 100
    print(f"Components needed to retain 90% variance: {n_components_90}")
    print(f"{n_components_90} out of {n} is {ratio_of_64:.2f}% of all features")
    results['n_components_90'] = int(n_components_90)
    results['ratio_of_64'] = ratio_of_64

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(range(1, n + 1), cumulative_variance, marker='o', markersize=3, color='#2563eb', label='Cumulative variance')
    ax.axhline(0.90, color='red', linestyle='--', label='90% threshold')
    ax.axvline(n_components_90, color='green', linestyle=':', label=f'k = {n_components_90}')
    ax.set_xlabel("Number of principal components (k)")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_title("Cumulative Explained Variance vs. Number of Components")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT}/step6_cumulative_variance.png", dpi=150)
    plt.close()

    return cumulative_variance
