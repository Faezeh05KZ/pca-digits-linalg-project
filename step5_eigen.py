"""
مرحله 5: محاسبه مقادیر و بردارهای ویژه (eigh روی C)
"""
import numpy as np

def run(C, results):
    print()
    print("=" * 70)
    print("Step 5 - Computing eigenvalues and eigenvectors")
    print("=" * 70)

    eigenvalues, eigenvectors = np.linalg.eigh(C)

    # مرتب‌سازی نزولی (هماهنگ برای مقادیر و بردارها)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    print(f"Top 5 eigenvalues: {np.round(eigenvalues[:5], 3)}")
    print(f"Sum of all eigenvalues (should equal trace of C): {eigenvalues.sum():.4f}")
    print(f"Trace of C: {np.trace(C):.4f}")
    results['top5_eigenvalues'] = eigenvalues[:5].tolist()

    # بررسی تعامد بردارهای ویژه
    WtW = eigenvectors.T @ eigenvectors
    eigvec_orthonormal = np.allclose(WtW, np.eye(WtW.shape[0]), atol=1e-8)
    print(f"Are the eigenvectors orthonormal (W^T W = I)? {eigvec_orthonormal}")
    results['eigvec_orthonormal'] = eigvec_orthonormal

    return eigenvalues, eigenvectors
