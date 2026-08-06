import matplotlib.pyplot as plt
from itertools import combinations

from config import OUT

def run(B, eigenvectors, y, results):
    print()
    print("=" * 70)
    print("Step 8 - Visualization in 2D space")
    print("=" * 70)

    W2 = eigenvectors[:, :2]
    T2 = B @ W2

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(T2[:, 0], T2[:, 1], c=y, cmap='tab10', s=15, alpha=0.8)
    ax.set_xlabel("Principal Component 1 (PC1)")
    ax.set_ylabel("Principal Component 2 (PC2)")
    ax.set_title("2D PCA Visualization of the Digits Dataset")
    cbar = plt.colorbar(scatter, ax=ax, ticks=range(10))
    cbar.set_label("Digit label")
    plt.tight_layout()
    plt.savefig(f"{OUT}/step8_scatter_2d.png", dpi=150)
    plt.close()

    return T2
