import config 

import step1_data
import step2_centering
import step3_covariance
import step4_qr
import step5_eigen
import step6_variance
import step7_reduction
import step8_visualization
import step9_reconstruction
import step10_small_sample
import optional_svd


def main():
    results = {}

    X, y, m, n = step1_data.run(results)
    B, mu = step2_centering.run(X, results)
    C = step3_covariance.run(B, m, results)
    step4_qr.run(B, results)
    eigenvalues, eigenvectors = step5_eigen.run(C, results)
    step6_variance.run(eigenvalues, n, results)
    step7_reduction.run(B, eigenvectors, results, k=10)
    step8_visualization.run(B, eigenvectors, y, results)
    step9_reconstruction.run(X, B, mu, eigenvalues, eigenvectors, y, m, n, results)
    step10_small_sample.run(X, n, results)
    optional_svd.run(B, m, eigenvalues, eigenvectors, results)

    print()
    print("All steps executed successfully. Plots saved in the outputs folder.")

    return results


if __name__ == "__main__":
    main()
