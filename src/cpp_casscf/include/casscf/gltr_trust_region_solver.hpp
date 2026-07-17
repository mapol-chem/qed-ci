#pragma once

#include "casscf/trust_region_solver.hpp"
#include <memory>

namespace casscf {

// GLTR (Generalized Lanczos Trust-Region) solver.
//
// Port target: solve_gltr_with_operator (helper_PFCI.py:14783-15011).
// solve_gltr_trust_region (helper_PFCI.py:15015-15247) turns out to be a
// byte-for-byte duplicate of the same algorithm, differing only in how it
// computes the Hessian-vector product -- inline via `self.orbital_sigma3(...)`
// (helper_PFCI.py:15057-15059) instead of a generic `B_operator(v)` callable.
// That's exactly what HessianOperator::apply() already abstracts (see
// mv2/get_bfgs_mv in hessian_operator.hpp), so both Python entry points
// collapse into this one class.
//
// Algorithm: build a preconditioned Lanczos tridiagonalization of the
// (matrix-free) Hessian with full re-orthogonalization against every prior
// Lanczos vector; at each iteration diagonalize the growing tridiagonal
// model and test (a) whether the unconstrained Newton step in the Lanczos
// basis lands inside the trust radius (interior case), and if not, (b)
// solve the secular equation ||p(lambda)|| = radius by bisection on lambda,
// with an explicit hard-case branch (the secular function's value at the
// safeguarded lower bound on lambda is already below the radius -- no root
// exists -- handled by adding a scaled component of the lowest tridiagonal
// eigenvector to reach the boundary exactly).
//
// Unlike LstrsSolver/PcgTrustRegionSolver, this does NOT use diagonal
// structure to seed anything -- it's selected exactly when that structure
// stops being useful (n_negative == 0 in select_trs_strategy).
//
// NOISE: the Python seeds the very first Lanczos vector from `g + noise`
// with `noise = np.random.randn(n)` (unseeded, global RNG) to perturb away
// from exact orthogonality to hidden negative-curvature directions
// (helper_PFCI.py:14799-14802). For reproducible testing, this port uses a
// locally seeded generator (GltrConfig::random_seed) instead of relying on
// global RNG state, and lets the noise be disabled entirely
// (GltrConfig::add_noise = false) for deterministic validation runs.
struct GltrConfig {
    double tol = 1e-4;
    int max_iter = 100;
    bool add_noise = true;
    double noise_relative_scale = 1e-6; // helper_PFCI.py:14800
    unsigned int random_seed = 0;
};

class GltrTrustRegionSolver final : public TrustRegionSolver {
public:
    GltrTrustRegionSolver(std::shared_ptr<const HessianOperator> hessian,
                           Vector diag_preconditioner,
                           GltrConfig config = {});

    TrustRegionResult solve(const Vector& gradient, double trust_radius) const override;

private:
    std::shared_ptr<const HessianOperator> hessian_;
    Vector diag_preconditioner_;
    GltrConfig config_;
};

} // namespace casscf
