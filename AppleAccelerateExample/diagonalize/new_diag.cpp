#include <iostream>
#include <vector>
#include <cstring> // for memset
#include <algorithm> // for std::swap

extern "C" {
    void dsyev_(char* jobz, char* uplo, int* n, double* A, int* lda, double* eig, double* work, int* lwork, int* info);
}

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

    // Eigenvalues are already in ascending order, so no need to sort them.
    // If you want to sort eigenvectors based on eigenvalues, you can do so here.
    // However, LAPACK already returns them in ascending order, so this is unnecessary.
}



int main() {
    const int N = 3;
    double A[N * N] = {4, 1, 1, 1, 2, 0, 1, 0, 3}; // Symmetric matrix
    double eig[N]; // Array to store eigenvalues

    symmetric_eigenvalue_problem(A, N, eig);

    std::cout << "Eigenvalues:\n";
    for (int i = 0; i < N; ++i) {
        std::cout << eig[i] << " ";
    }
    std::cout << "\n";

    std::cout << "Eigenvectors (columns):\n";
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << A[i * N + j] << " ";
        }
        std::cout << "\n";
    }

    return 0;
}
