import numpy as np
import matplotlib.pyplot as plt

from config import OUT

def reconstruct(B_data, eigvecs, k_):
    Wk = eigvecs[:, :k_]
    Tk = B_data @ Wk
    B_rec = Tk @ Wk.T
    return B_rec


def mse_for_k(X, B, mu, eigenvectors, k_):
    B_rec = reconstruct(B, eigenvectors, k_)
    X_rec = B_rec + mu
    mse = np.mean((X - X_rec) ** 2)
    return mse


def run(X, B, mu, eigenvalues, eigenvectors, y, m, n, results):
    print()
    print("=" * 70)
    print("Step 9 - Data reconstruction and error analysis")
    print("=" * 70)

    ks = list(range(1, n + 1))
    mses = [mse_for_k(X, B, mu, eigenvectors, kk) for kk in ks]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ks, mses, color='#dc2626')
    ax.set_xlabel("Number of components used (k)")
    ax.set_ylabel("Reconstruction error (MSE)")
    ax.set_title("Reconstruction Error Decreasing with More Components")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT}/step9_mse_vs_k.png", dpi=150)
    plt.close()

    print(f"MSE with k=2  : {mse_for_k(X, B, mu, eigenvectors, 2):.5f}")
    print(f"MSE with k=10 : {mse_for_k(X, B, mu, eigenvectors, 10):.5f}")
    print(f"MSE with k=30 : {mse_for_k(X, B, mu, eigenvectors, 30):.5f}")
    print(f"MSE with k=64 (full reconstruction, should be ~0): {mse_for_k(X, B, mu, eigenvectors, 64):.2e}")
    results['mse_k2'] = mse_for_k(X, B, mu, eigenvectors, 2)
    results['mse_k10'] = mse_for_k(X, B, mu, eigenvectors, 10)
    results['mse_k30'] = mse_for_k(X, B, mu, eigenvectors, 30)
    results['mse_k64'] = mse_for_k(X, B, mu, eigenvectors, 64)

    sample_idx = 0
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
    axes[0].imshow(X[sample_idx].reshape(8, 8), cmap='gray')
    axes[0].set_title("Original image")
    for ax_, kk in zip(axes[1:], [2, 10, 30]):
        B_rec = reconstruct(B, eigenvectors, kk)
        X_rec = B_rec + mu
        ax_.imshow(X_rec[sample_idx].reshape(8, 8), cmap='gray')
        ax_.set_title(f"Reconstructed with k={kk}")
    for ax_ in axes:
        ax_.set_xticks([])
        ax_.set_yticks([])
    plt.suptitle(f"Reconstruction of a sample image (true label={y[sample_idx]}) for different k values")
    plt.tight_layout()
    plt.savefig(f"{OUT}/step9_reconstruction_comparison.png", dpi=150)
    plt.close()
