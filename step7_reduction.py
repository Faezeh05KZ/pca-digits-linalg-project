"""
مرحله 7: کاهش بعد (تغییر پایه) با k=10
"""

def run(B, eigenvectors, results, k=10):
    print()
    print("=" * 70)
    print(f"Step 7 - Dimensionality reduction (change of basis) with k={k}")
    print("=" * 70)

    W = eigenvectors[:, :k]      # ماتریس 64 x k
    T = B @ W                     # ماتریس m x k
    print(f"W shape: {W.shape}")
    print(f"T shape (new representation of data in {k}-D space): {T.shape}")
    results['k_step7'] = k
    results['W_shape'] = W.shape
    results['T_shape'] = T.shape

    return W, T
