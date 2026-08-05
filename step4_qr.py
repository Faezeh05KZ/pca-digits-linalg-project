import numpy as np

def one_qr_step(A):
    Q, R = np.linalg.qr(A, mode='reduced')
    A1 = R @ Q
    return A1, Q, R


def run(B, results):
    print()
    print("=" * 70)
    print("Step 4 - Numerical exercise: QR algorithm and rank of the data matrix")
    print("=" * 70)

    print("Testing the QR algorithm on a few random 4x4 symmetric matrices:")
    for trial in range(2):
        M = np.random.randn(4, 4)
        M = (M + M.T) / 2 
        A1, Q, R = one_qr_step(M)

        eig_before = np.sort(np.linalg.eigvalsh(M))
        eig_after = np.sort(np.linalg.eigvalsh(A1))
        same = np.allclose(eig_before, eig_after, atol=1e-8)
        print(f"  Trial {trial+1}: eigenvalues equal before/after the QR step? {same}")
        print(f"    Eigenvalues before: {np.round(eig_before, 4)}")
        print(f"    Eigenvalues after : {np.round(eig_after, 4)}")


    Q_B, R_B = np.linalg.qr(B, mode='reduced')
    print(f"\nQ shape (reduced mode on B): {Q_B.shape}")
    print(f"R shape (reduced mode on B): {R_B.shape}")

    diag_R = np.diag(R_B)
    tol = 1e-8 * np.max(np.abs(diag_R))
    rank_from_R = np.sum(np.abs(diag_R) > tol)
    print(f"Number of nonzero diagonal entries of R (rank of B estimated from R): {rank_from_R}")
    print(f"Actual rank of B via numpy.linalg.matrix_rank: {np.linalg.matrix_rank(B)}")
    results['rank_B_from_R'] = int(rank_from_R)
    results['rank_B_numpy'] = int(np.linalg.matrix_rank(B))

    QtQ = Q_B.T @ Q_B
    orthonormal_check = np.allclose(QtQ, np.eye(QtQ.shape[0]), atol=1e-8)
    print(f"Is Q^T Q = I? (confirms orthonormality of Q's columns) {orthonormal_check}")
    results['Q_orthonormal'] = orthonormal_check
