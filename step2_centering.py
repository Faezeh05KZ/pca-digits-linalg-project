import numpy as np

def run(X, results):
    print()
    print("=" * 70)
    print("Step 2 - Centering the data")
    print("=" * 70)

    mu = X.mean(axis=0) 
    B = X - mu   

    mean_of_B_columns = B.mean(axis=0)
    max_abs_mean_B = np.max(np.abs(mean_of_B_columns))
    print(f"Max |mean of B's columns| (should be ~0): {max_abs_mean_B:.2e}")
    results['max_abs_mean_B'] = max_abs_mean_B

    return B, mu
