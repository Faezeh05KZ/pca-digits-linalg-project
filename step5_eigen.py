import numpy as np

def run(C, results):
    print()
    print("=" * 70)
    print("Step 5 - Computing eigenvalues and eigenvectors")
    print("=" * 70)

    val, vec = np.linalg.eigh(C)

    sorted_indices = np.argsort(val)[::-1]
    
    eigenvalues = val[sorted_indices]
    eigenvectors = vec[:, sorted_indices]

    print(f"Top 5 eigenvalues: {np.round(eigenvalues[:5], 3)}")
    print(f"Sum of all eigenvalues (should equal trace of C): {eigenvalues.sum():.4f}")
    print(f"Trace of C: {np.trace(C):.4f}")
    
    results['top5_eigenvalues'] = eigenvalues[:5].tolist()


    reconstruction = eigenvectors.T @ eigenvectors
    is_orthonormal = np.allclose(reconstruction, np.eye(reconstruction.shape[0]) , atol=1e-8)

    
    print(f"Are the eigenvectors orthonormal (W^T W = I)? {is_orthonormal}")
    results['eigvec_orthonormal'] = is_orthonormal

    return eigenvalues, eigenvectors
