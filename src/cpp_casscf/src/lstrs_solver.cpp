#include "casscf/lstrs_solver.hpp"
#include "casscf/bordered_eigensolve.hpp"
#include "casscf/lstrs_bisection_core.hpp"
#include "casscf/minres_solver.hpp"

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

    // Faithful port of scipy.sparse.linalg.minres(hessian_tilde_ai,
    // -gradient_tilde_ai, rtol=1e-5) (helper_PFCI.py:7547-7549). An earlier
    // version of this lambda used a direct dense solve instead, on the
    // assumption it would be numerically equivalent at these small
    // dimensions -- validating against real chemistry (see
    // cpp_casscf/validation/, README "Sweep findings") showed that's false
    // whenever hessian_tilde_ai is ill-conditioned, which is common exactly
    // because hard_case==2 triggers near a singular Hessian: scipy's MINRES
    // legitimately halts well short of full convergence under its own
    // Paige/Saunders stopping test, and a direct solve doesn't reproduce
    // that. See minres_solver.hpp's doc comment for detail.
    HardCase2SolveFn hard_case2_solve = [&]() -> Vector { return minres_solve(H, -gradient, 1e-5).x; };

    return solve_lstrs_bisection(gradient, H.diagonal(), trust_radius, max_iter_,
                                  eigenpairs_at, model_value, hard_case2_solve);
}

} // namespace casscf
