#include "casscf/lstrs_solver.hpp"
#include "casscf/bordered_eigensolve.hpp"
#include "casscf/lstrs_bisection_core.hpp"

namespace casscf {

LstrsSolver::LstrsSolver(std::shared_ptr<const DenseHessianOperator> hessian, int max_iter)
    : hessian_(std::move(hessian)), max_iter_(max_iter) {}

TrustRegionResult LstrsSolver::solve(const Vector& gradient, double trust_radius) const {
    const Matrix& H = hessian_->matrix();

    EigenpairAtAlphaFn eigenpairs_at = [&](double alpha, double /*alpha_range*/, bool /*restart*/) {
        BorderedEigenPairs eig = bordered_eigensolve(gradient, H, alpha);
        TwoLowestEigenpairs result;
        result.mu0 = eig.eigenvalues(0);
        result.mu1 = eig.eigenvalues(1);
        result.w0 = eig.eigenvectors.col(0);
        result.w1 = eig.eigenvectors.col(1);
        return result;
    };

    ModelValueFn model_value = [&](const Vector& x) { return gradient.dot(x) + 0.5 * x.dot(H * x); };

    // Replaces scipy's iterative minres (helper_PFCI.py:7525-7527) with a
    // direct dense solve -- equivalent for the small dimensions this solver
    // targets, since the system is already fully materialized.
    // NOTE: the Vector return type forces eager evaluation here -- returning
    // the bare `H.ldlt().solve(...)` expression would dangle a reference to
    // the temporary LDLT decomposition once this lambda returns.
    HardCase2SolveFn hard_case2_solve = [&]() -> Vector { return H.ldlt().solve(-gradient); };

    return solve_lstrs_bisection(gradient, H.diagonal(), trust_radius, max_iter_,
                                  eigenpairs_at, model_value, hard_case2_solve);
}

} // namespace casscf
