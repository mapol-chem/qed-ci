#include <iostream>
#include <vector>
#include <cstring> // for memset
#include <Accelerate/Accelerate.h> // For BLAS and LAPACK

extern "C" {
    void dsyev_(char* jobz, char* uplo, int* n, double* A, int* lda, double* eig, double* work, int* lwork, int* info);
}

// Function to perform symmetric eigenvalue decomposition
void symmetric_eigenvalue_problem(double* A, int N, double* eig) {
    int n = N, lda = N, info;

    // Query the optimal workspace size
    int lwork = -1;
    double work_query;
    char jobz = 'V'; // Compute eigenvalues and eigenvectors
    char uplo = 'U'; // Upper triangle of A is stored

    dsyev_(&jobz, &uplo, &n, A, &lda, eig, &work_query, &lwork, &info);

    // Allocate the workspace
    lwork = static_cast<int>(work_query);
    std::vector<double> work(lwork);

    // Compute eigenvalues and eigenvectors
    // A will be overwritten with eigenvectors
    // eig will hold the eigenvalues
    dsyev_(&jobz, &uplo, &n, A, &lda, eig, work.data(), &lwork, &info);

    if (info != 0) {
        std::cerr << "LAPACK dsyev_ failed with info = " << info << std::endl;
        return;
    }
}

// Function to perform matrix-matrix multiplication using cblas_dgemm
void matrix_matrix_multiply(const double* A, const double* B, double* C, int M, int N, int K) {
    // C = A * B
    // A is M x K
    // B is K x N
    // C is M x N
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0, A, K, B, N, 0.0, C, N);
}

// Utility function to print a matrix
void printMatrix(const char* name, const double* matrix, int rows, int cols) {
    std::cout << name << ":\n";
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << matrix[i * cols + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

int main() {
    const int N = 3;

    // Example symmetric matrix for eigenvalue decomposition
    double A[N * N] = {4, 1, 1, 1, 2, 0, 1, 0, 3}; // Symmetric matrix
    double eig[N]; // Array to store eigenvalues

    // Perform eigenvalue decomposition
    symmetric_eigenvalue_problem(A, N, eig);

    std::cout << "Eigenvalues:\n";
    for (int i = 0; i < N; ++i) {
        std::cout << eig[i] << " ";
    }
    std::cout << "\n";

    std::cout << "Eigenvectors (columns):\n";
    printMatrix("Eigenvectors", A, N, N);

    // Example matrices for matrix-matrix multiplication
    double B[N * N] = {1, 2, 3, 4, 5, 6, 7, 8, 9}; // 3x3 matrix
    double C[N * N] = {0}; // Result matrix

    // Perform matrix-matrix multiplication: C = A * B
    matrix_matrix_multiply(A, B, C, N, N, N);

    // Print the result of matrix multiplication
    printMatrix("Matrix C (A * B)", C, N, N);

    return 0;
}
