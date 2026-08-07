import numpy as np

def run(B, m, eigenvalues, eigenvectors, results):
    print()
    print("=" * 70)
    print("Optional section - PCA and SVD relationship")
    print("=" * 70)

    U, S, Vt = np.linalg.svd(B, full_matrices=False)
    V = Vt.T

    cos_sim = [np.abs(np.dot(V[:, i], eigenvectors[:, i])) for i in range(5)]
    print(f"Absolute cosine similarity between C's first 5 eigenvectors and V's first 5 columns (should be near 1): {np.round(cos_sim, 6)}")

    eigvals_from_svd = (S ** 2) / (m - 1)
    diff = np.max(np.abs(eigvals_from_svd[:10] - eigenvalues[:10]))
    print(f"Max difference between eigh eigenvalues and SVD-derived eigenvalues (first 10): {diff:.2e}")
    results['svd_eig_diff'] = diff
    results['cos_sim_svd'] = cos_sim
