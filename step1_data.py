import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

from config import OUT

def run(results):
    print("=" * 70)
    print("Step 1 - Getting to know the data")
    print("=" * 70)

    digits = load_digits()
    X = digits.data    
    y = digits.target

    m, n = X.shape
    print(f"X matrix shape: {X.shape}  (m={m} samples, n={n} features)")
    results['m'] = m
    results['n'] = n

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(X[0].reshape(8, 8), cmap='gray')
    ax.set_title(f"Sample data — true label: {y[0]}")
    ax.set_xlabel("pixel column")
    ax.set_ylabel("pixel row")
    plt.tight_layout()
    plt.savefig(f"{OUT}/step1_sample_image.png", dpi=150)
    plt.close()

    rank_X_numeric = np.linalg.matrix_rank(X)
    print(f"Numeric rank of X (for conceptual inspection): {rank_X_numeric} out of max {min(m, n)}")
    results['rank_X'] = rank_X_numeric

    return X, y, m, n
