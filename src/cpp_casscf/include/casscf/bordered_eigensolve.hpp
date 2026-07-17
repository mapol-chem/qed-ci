#pragma once

#include "casscf/types.hpp"

namespace casscf {

// Result of diagonalizing the bordered/augmented matrix
//     [ alpha   g^T ]
//     [ g       H   ]
// Eigenvalues ascending (mu(0) <= mu(1) <= ...), eigenvectors as columns,
// each of length dim(g)+1 with the border component at index 0.
struct BorderedEigenPairs {
    Vector eigenvalues;
    Matrix eigenvectors;
};

// Port of projection_step2, helper_PFCI.py:16621-16643. Shared by
// LstrsSolver (dense, full-dimension bordered matrix) and
// DavidsonAugmentedHessianSolver (dense, but on the small guess/subspace
// projection of the bordered matrix) -- both ultimately reduce to
// diagonalizing a small bordered matrix and reading off the two lowest
// eigenpairs.
BorderedEigenPairs bordered_eigensolve(const Vector& gradient, const Matrix& hessian, double alpha);

} // namespace casscf
