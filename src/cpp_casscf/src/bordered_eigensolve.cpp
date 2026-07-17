#include "casscf/bordered_eigensolve.hpp"

#include <Eigen/Eigenvalues>

namespace casscf {

BorderedEigenPairs bordered_eigensolve(const Vector& gradient, const Matrix& hessian, double alpha) {
    const int n = static_cast<int>(gradient.size());
    const int dim0 = n + 1;

    Matrix augmented = Matrix::Zero(dim0, dim0);
    augmented(0, 0) = alpha;
    augmented.block(0, 1, 1, n) = gradient.transpose();
    augmented.block(1, 0, n, 1) = gradient;
    augmented.block(1, 1, n, n) = hessian;

    Eigen::SelfAdjointEigenSolver<Matrix> solver(augmented);

    BorderedEigenPairs result;
    result.eigenvalues = solver.eigenvalues();
    result.eigenvectors = solver.eigenvectors();
    return result;
}

} // namespace casscf
