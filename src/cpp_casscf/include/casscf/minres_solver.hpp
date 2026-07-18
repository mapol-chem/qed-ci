#pragma once

#include "casscf/types.hpp"

namespace casscf {

struct MinresResult {
    Vector x;
    // scipy's istop/info convention: 1-5 are various "converged" reasons,
    // 6 means the iteration limit was reached without meeting rtol, -1 is
    // the beta2==0 special case (Abar effectively a multiple of I).
    int istop = 0;
    int iterations = 0;
};

// Faithful port of scipy.sparse.linalg.minres (Paige/Saunders MINRES,
// https://web.stanford.edu/group/SOL/software/minres/ -- scipy's own
// docstring says it's a translation of the reference MATLAB
// implementation), specialized to this module's exact usage: a dense
// symmetric matrix A, starting guess x0 = 0, no preconditioner, no shift.
//
// Why this exists: internal_optimization3's hard_case==2 fallback
// (helper_PFCI.py:7545-7551) calls exactly
// `scipy.sparse.linalg.minres(hessian_tilde_ai, -gradient_tilde_ai, rtol=1e-5)`.
// LstrsSolver originally substituted an exact `H.ldlt().solve(-gradient)`
// for this, on the assumption the two would be numerically equivalent at
// the small dimensions this solver targets. Validating against real
// chemistry (see cpp_casscf/validation/, README "Sweep findings") showed
// that assumption is false whenever hessian_tilde_ai is ill-conditioned --
// which is common exactly because hard_case==2 triggers near a singular
// Hessian. scipy's MINRES uses its own multi-quantity Paige/Saunders
// stopping test (Acond, epsx, test1/test2 -- not a simple
// ||residual||/||b|| < rtol check) and can legitimately halt well short of
// the exact solution; a faithful port needs to reproduce that stopping
// behavior, not just call some other iterative solver with the same rtol
// (Eigen's built-in MINRES was tried and simply reconverges to the exact
// answer within a handful of iterations for these small systems, since
// Krylov methods hit exact convergence in at most n iterations -- it does
// not reproduce scipy's deliberately-early-stopped answer).
//
// One deliberate deviation, flagged rather than silently matched: scipy
// raises ValueError if the (M-preconditioned) residual inner product comes
// out negative ("non-symmetric matrix"), which can't happen mathematically
// for a symmetric A with no preconditioner but could in principle be
// nudged negative by floating-point roundoff on a near-singular Hessian.
// This port clamps that quantity to zero instead of raising, since crashing
// the whole CASSCF optimization over a roundoff-level sign flip would be
// worse than a defensive clamp; iterations/istop still reflect whatever
// happens afterward.
MinresResult minres_solve(const Matrix& A, const Vector& b, double rtol = 1e-5, int maxiter = -1);

} // namespace casscf
