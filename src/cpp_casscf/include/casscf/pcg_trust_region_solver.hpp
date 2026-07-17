#pragma once

#include "casscf/trust_region_solver.hpp"
#include <memory>

namespace casscf {

// Port of solve_pcg_trust_region / solve_pcg_with_operator
// (helper_PFCI.py:16148-16264 and :16265+): diagonally-preconditioned
// Steihaug-Toint truncated CG. Matrix-free, so it is driven by whichever
// HessianOperator the caller supplies (exact reduced-Hessian sigma build or
// the L-BFGS approximation).
//
// Used once the augmented-Hessian diagonal is uniformly positive (Davidson's
// initial guess stops being reliable there) but the true Hessian can still
// have negative curvature -- handled directly via the negative-curvature
// branch below, without needing an augmented/bordered matrix.
class PcgTrustRegionSolver final : public TrustRegionSolver {
public:
    // diag_preconditioner: diagonal of M, same role as M_diag in the Python
    // signature. Values are floored at 1e-6 in absolute value, matching the
    // `epsilon` safeguard at helper_PFCI.py:16177-16178.
    PcgTrustRegionSolver(std::shared_ptr<const HessianOperator> hessian,
                          Vector diag_preconditioner,
                          double tol = 1e-8,
                          int max_iter = 1000);

    TrustRegionResult solve(const Vector& gradient, double trust_radius) const override;

private:
    std::shared_ptr<const HessianOperator> hessian_;
    Vector diag_preconditioner_;
    double tol_;
    int max_iter_;
};

} // namespace casscf
