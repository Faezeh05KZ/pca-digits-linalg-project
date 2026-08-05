
# PCA From Scratch — Discovering the Essence of Data

> A Linear Algebra final project: implementing **Principal Component Analysis (PCA)** completely from scratch — no `sklearn.decomposition.PCA`, no shortcuts, just vectors, eigenvalues, and a lot of curiosity.

---

## What is this?

We took 1800 handwritten digit images (8×8 pixels = 64 features each) and asked one simple question:

> *Can we represent these images with way fewer numbers, without losing what makes a "5" look like a "5"?*

The answer is **PCA** — and instead of calling one magic function, we built every mathematical piece ourselves:

`center data → build covariance matrix → find eigenvectors → change of basis → reduce dimensions → reconstruct → analyze error`

---

## Project structure

```
pca_package/
├── config.py                  # shared settings: output folder + random seed
├── step1_data.py               # Step 1  — load data, view as vectors in R^64
├── step2_centering.py          # Step 2  — center the data (shift to origin)
├── step3_covariance.py         # Step 3  — build the covariance matrix
├── step4_qr.py                 # Step 4  — QR algorithm + rank exercise
├── step5_eigen.py               # Step 5  — eigenvalues & eigenvectors (eigh)
├── step6_variance.py           # Step 6  — explained variance / cumulative plot
├── step7_reduction.py          # Step 7  — dimensionality reduction (k=10)
├── step8_visualization.py      # Step 8  — 2D scatter plot (k=2)
├── step9_reconstruction.py     # Step 9  — reconstruct images & measure MSE
├── step10_small_sample.py      # Step 10 — the m < n case (50 samples)
├── optional_svd.py             # Bonus — PCA ↔ SVD relationship
├── main.py                     # Entry point — runs everything in order
├── requirements.txt
└── outputs/                    # all generated plots (.png) land here
```

---

## Quick start

```bash
# 1. Create a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the whole pipeline
python3 main.py
```

All plots (cumulative variance, 2D digit clusters, reconstructions at different k, MSE curve...) will be saved automatically inside `outputs/`. 

---

## Rules we followed (per course requirements)

- ❌ No `sklearn.decomposition.PCA` or any ready-made PCA function
- ✅ Eigen-decomposition done only via `numpy.linalg.eigh`
- ✅ `sklearn.datasets` used **only** to load the digits dataset
- ✅ All linear algebra (centering, covariance, projection, reconstruction) implemented by hand

---

## Key concepts covered

| Concept | Where |
|---|---|
| Vector spaces & dimension | Step 1 |
| Change of basis / translation | Step 2 |
| Symmetric matrices & positive semi-definiteness | Step 3 |
| Similarity & the QR algorithm | Step 4 |
| Eigenvalues / eigenvectors & orthogonality | Step 5 |
| Rank & variance | Step 6 |
| Linear transformations, image & kernel | Step 7 |
| Geometric intuition of dimensionality reduction | Step 8–9 |
| Rank–nullity theorem (m < n case) | Step 10 |
| PCA ↔ SVD connection | Bonus |

---

---

## Course

Linear Algebra Final Project · Summer 2026
Professor: Dr. MirHossein Dezfoulian

---
