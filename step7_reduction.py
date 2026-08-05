def run(B, eigenvectors, results, k=10):
    print()
    print("=" * 70)
    print(f"Step 7 - Dimensionality reduction (change of basis) with k={k}")
    print("=" * 70)

    W = eigenvectors[:, :k]
    T = B @ W
    
    print(f"W shape: {W.shape}")
    print(f"T shape (new representation of data in {k}-D space): {T.shape}")
    
    results['k_step7'] = k
    results['W_shape'] = W.shape
    results['T_shape'] = T.shape

    return W, T
