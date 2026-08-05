import numpy as np

def run(B, m, results):
    print()
    print("=" * 70)
    print("Step 3 - Building the covariance matrix")
    print("=" * 70)

    C = (B.T @ B) / (m - 1)     
    print(f"C shape: {C.shape}")
    is_symmetric = np.allclose(C, C.T)
    print(f"Is C symmetric? {is_symmetric}")
    results['C_shape'] = C.shape
    results['C_symmetric'] = is_symmetric

    return C
