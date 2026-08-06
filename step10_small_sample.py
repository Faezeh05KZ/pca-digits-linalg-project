import numpy as np


def run(X, n, results):
    print()
    print("=" * 70)
    print("Step 10 - The m < n case")
    print("=" * 70)

    indices = np.random.choice(len(X), 50, replace=False)
    X_small = X[indices]
    m_small = X_small.shape[0]

    mu_small = X_small.mean(axis=0)
    B_small = X_small - mu_small
    C_small = (B_small.T @ B_small) / (m_small - 1)

    eigvals_small, eigvecs_small = np.linalg.eigh(C_small)
    idx_small = np.argsort(eigvals_small)[::-1]
    eigvals_small = eigvals_small[idx_small]
    eigvecs_small = eigvecs_small[:, idx_small]

    tol_small = 1e-8 * eigvals_small[0]
    n_zero_eig = np.sum(np.abs(eigvals_small) < tol_small)
    rank_B_small = np.linalg.matrix_rank(B_small)

    print(f"Subset sample count: m = {m_small}, feature count: n = {n}")
    print(f"Number of zero or near-zero eigenvalues (relative tolerance): {n_zero_eig}")
    print(f"Actual rank of B_small (numpy.linalg.matrix_rank): {rank_B_small}")
    print(f"64 - rank = {n - rank_B_small}")
    results['m_small'] = int(m_small)
    results['n_zero_eig'] = int(n_zero_eig)
    results['rank_B_small'] = int(rank_B_small)
