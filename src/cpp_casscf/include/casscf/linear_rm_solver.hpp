#pragma once

#include "casscf/types.hpp"

#include <vector>

namespace casscf {

// Faithful port of LinearRMSolver (residual_minimization.py) -- a
// Krylov-subspace "Linear Residual Minimization" iterative solver for
// Ax + b = 0, based on the algorithm in BAGEL's linearRM.h: each outer
// iteration contributes a new (trial direction, matrix-vector product)
// pair to a small subspace, canonically orthogonalized (eigendecomposition
// of the trial-direction overlap matrix, discarding near-zero eigenvalues
// to remove linear dependence) and re-extrapolated to the best
// linear-combination residual every step -- NOT a plain unpreconditioned
// Krylov method. Stateful: the caller drives the outer loop (see
// linear_equation_solve.hpp), calling update_subspace_and_extrapolate()
// once per outer iteration with a fresh (trial, matvec) pair.
class LinearRMSolver {
public:
    LinearRMSolver(Vector b_vector, int max_subspace);

    // residual_minimization.py:60-124. Returns the new residual estimate
    // (b + s_optimal), matching the Python's own return value exactly.
    Vector update_subspace_and_extrapolate(const Vector& c_new, const Vector& s_new);

    // residual_minimization.py:126-131.
    Vector get_solution() const;

private:
    Vector b_;
    int max_subspace_;
    std::vector<Vector> c_vectors_;
    std::vector<Vector> s_vectors_;
};

} // namespace casscf
