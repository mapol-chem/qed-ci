#include "casscf/linear_equation_solve.hpp"

#include "casscf/linear_rm_solver.hpp"
#include "casscf/orbital_sigma.hpp"

#include <random>

namespace casscf {

LinearEquationSolveResult linear_equation_solve(const std::function<Vector(const Vector&)>& apply,
                                                  const Vector& reduced_gradient, const Vector& denom, int max_iter,
                                                  double conv_thresh, unsigned int random_seed) {
    const int n = static_cast<int>(reduced_gradient.size());

    // helper_PFCI.py:16346-16358: random-probe initial residual -- see
    // this function's own header doc comment for why this is an inherent,
    // documented non-reproducibility against the Python, and for why
    // `random_seed` makes THIS side deterministic across repeated runs
    // (matching GltrTrustRegionSolver::solve()'s own fresh-RNG-per-call
    // pattern) even though it can't match Python's own specific draws.
    Vector residual = reduced_gradient;
    if (residual.norm() < 1e-3) {
        std::mt19937 rng(random_seed);
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        Vector trial0(n);
        for (int i = 0; i < n; ++i) trial0(i) = dist(rng);
        const double norm0 = trial0.norm();
        if (norm0 > 1e-12) trial0 /= norm0;
        residual = apply(trial0) + reduced_gradient;
    }

    LinearRMSolver solver(reduced_gradient, max_iter);

    for (int i = 0; i < max_iter; ++i) {
        const double residual_norm = residual.norm();
        if (residual_norm < conv_thresh) {
            return {solver.get_solution(), true};
        }

        // helper_PFCI.py:16378-16382: diagonal-preconditioned trial
        // direction.
        Vector trial_c = residual.cwiseQuotient(denom);
        const double norm = trial_c.norm();
        if (norm > 1e-12) trial_c /= norm;

        Vector sigma = apply(trial_c);
        Vector residual_old = residual;
        residual = solver.update_subspace_and_extrapolate(trial_c, sigma);

        // helper_PFCI.py:16389-16393: self-consistency convergence check.
        if (i > 0 && (residual - residual_old).norm() < 1e-8) {
            return {solver.get_solution(), true};
        }
    }
    return {solver.get_solution(), false};
}

LinearEquationSolveResult linear_equation_solve(const Matrix& U, const Matrix& A_tilde, const Tensor4& G,
                                                  const Vector& reduced_gradient, const Vector& denom, int max_iter,
                                                  double conv_thresh, const Dimensions& dims,
                                                  unsigned int random_seed) {
    // Fast path: build the sigma operator once (fixed U/A_tilde/G) and reuse
    // it across every iteration's Hessian-vector product.
    OrbitalSigmaOperator sigma_op(U, A_tilde, G, dims);
    auto apply = [&sigma_op](const Vector& v) { return sigma_op.apply(v); };
    return linear_equation_solve(apply, reduced_gradient, denom, max_iter, conv_thresh, random_seed);
}

} // namespace casscf
