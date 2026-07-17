#include "casscf/davidson_driven_lstrs_solver.hpp"
#include "casscf/lstrs_bisection_core.hpp"
#include "casscf/pcg_trust_region_solver.hpp"

#include <stdexcept>

namespace casscf {

DavidsonDrivenLstrsSolver::DavidsonDrivenLstrsSolver(
    std::shared_ptr<const HessianOperator> hessian,
    std::shared_ptr<const HessianGuessProvider> guess_provider,
    Vector reduced_hessian_diagonal,
    DavidsonDrivenLstrsConfig config,
    DavidsonAugmentedHessianConfig davidson_config)
    : hessian_(std::move(hessian)),
      guess_provider_(std::move(guess_provider)),
      reduced_hessian_diagonal_(std::move(reduced_hessian_diagonal)),
      config_(config),
      davidson_config_(davidson_config) {}

TrustRegionResult DavidsonDrivenLstrsSolver::solve(const Vector& gradient, double trust_radius) const {
    const int H_dim = static_cast<int>(gradient.size()) + 1;
    DavidsonAugmentedHessianSolver davidson(hessian_, guess_provider_, davidson_config_);

    EigenpairAtAlphaFn eigenpairs_at = [&](double alpha, double alpha_range, bool restart) {
        DavidsonIterationResult davidson_result =
            davidson.solve(gradient, reduced_hessian_diagonal_, alpha, alpha_range, trust_radius, restart);
        // davidson.solve() always either converges or throws (matching the
        // Python's exit() on exhausting max_davidson_iterations), so
        // davidson_result.converged is guaranteed true here.

        TwoLowestEigenpairs result;
        if (davidson_result.eigenvalues.size() == 1) {
            // Easy case: only root 0 was needed. Fill in a "fake root 1"
            // exactly as the Python does (helper_PFCI.py:15430-15431:
            // aug_hessian_eigenvals[1] = 1e10; aug_hessian_eigenvecs[:,1] = 1e-14)
            // so the shared bisection core's aa2/bb2 test naturally treats
            // this second slot as negligible.
            result.mu0 = davidson_result.eigenvalues(0);
            result.w0 = davidson_result.eigenvectors.col(0);
            result.mu1 = 1e10;
            result.w1 = Vector::Constant(H_dim, 1e-14);
        } else {
            result.mu0 = davidson_result.eigenvalues(0);
            result.mu1 = davidson_result.eigenvalues(1);
            result.w0 = davidson_result.eigenvectors.col(0);
            result.w1 = davidson_result.eigenvectors.col(1);
        }
        return result;
    };

    ModelValueFn model_value = [&](const Vector& x) {
        return gradient.dot(x) + 0.5 * x.dot(hessian_->apply(x));
    };

    // See class doc comment, gap 2.
    HardCase2SolveFn hard_case2_solve = [&]() -> Vector {
        constexpr double effectively_unconstrained_radius = 1e10;
        PcgTrustRegionSolver cg(hessian_, reduced_hessian_diagonal_, config_.cg_tol, config_.cg_max_iter);
        return cg.solve(gradient, effectively_unconstrained_radius).step;
    };

    return solve_lstrs_bisection(gradient, reduced_hessian_diagonal_, trust_radius, config_.max_bisection_iter,
                                  eigenpairs_at, model_value, hard_case2_solve);
}

} // namespace casscf
