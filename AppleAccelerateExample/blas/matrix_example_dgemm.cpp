#include <iostream>
#include <vector>
#include <Accelerate/Accelerate.h>

// Helper function to print a matrix
void printMatrix(const std::vector<double>& matrix, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            std::cout << matrix[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    // Matrix Multiplication Example
    size_t m = 3; // rows of A and C
    size_t n = 4; // cols of A and rows of B
    size_t p = 2; // cols of B and C

    std::vector<double> A = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}; // m x n
    std::vector<double> B = {1, 2, 3, 4, 5, 6, 7, 8};  // n x p
    std::vector<double> C(m * p, 0.0); // m x p, initialized to 0

    // Use cblas_dgemm for double-precision matrix multiplication
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, p, n, 1.0, A.data(), n, B.data(), p, 0.0, C.data(), p);

    std::cout << "Matrix Multiplication:" << std::endl;
    std::cout << "Matrix A:" << std::endl;
    printMatrix(A, m, n);
    std::cout << "Matrix B:" << std::endl;
    printMatrix(B, n, p);
    std::cout << "Result Matrix C:" << std::endl;
    printMatrix(C, m, p);


    // Add Matrix Diagonalization Example - Get eigenvalues and eigenvectors of A
    // and print them here!
    return 0;
}
