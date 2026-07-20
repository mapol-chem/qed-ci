#pragma once

#include "casscf/types.hpp"

namespace casscf {

enum class TrsStrategy { DavidsonSubspaceBisection, Gltr };

// CONFIRMED runtime dispatch (not a guess): inside the `qn_optimization ==
// False` branch of microiteration_optimization6, when the gradient isn't
// already tiny, the code picks the solver from exactly this diagonal-sign
// statistic (helper_PFCI.py:11276-11277, 11291-11292):
//     n_negative = np.sum(self.reduced_hessian_diagonal < 0)
//     ...
//     if n_negative == 0:
//         solve_gltr_trust_region(...)
//     else:
//         [LSTRS-style beta-bisection loop, structurally the same shape as
//          internal_optimization3's dense loop but calling
//          Davidson_augmented_hessian_solve6 in place of a direct dense
//          eigh at each bisection step, helper_PFCI.py:11292-16085]
// i.e. Davidson-driven LSTRS is used while any reduced-Hessian diagonal
// entry is negative (the diagonal gives a good initial Ritz-vector guess
// for those directions -- see DavidsonAugmentedHessianSolver's index
// ranking), and GLTR takes over once the diagonal is uniformly
// non-negative, matching what the developer described.
//
// SCOPE NOTE: this only covers the `qn_optimization == False` branch. When
// qn_optimization is True, a *different* dispatch applies (exact-Hessian
// GLTR vs. L-BFGS-operator GLTR; see should_reset_bfgs_reference() below,
// helper_PFCI.py:11114-11149). Also, the DavidsonSubspaceBisection strategy
// here names the *outer* beta-bisection driver (helper_PFCI.py:11337-16085)
// that repeatedly calls Davidson_augmented_hessian_solve6 -- that outer
// driver loop itself is not yet ported; only its inner subproblem solve is
// (DavidsonAugmentedHessianSolver::solve_iteration_one).
TrsStrategy select_trs_strategy(const Vector& reduced_hessian_diagonal);

// State needed for the BFGS-reference-reset decision, mirroring the
// instance attributes read at helper_PFCI.py:11118
// (self.density_norm_change, self.predicted_energy, qn_count,
// self.consecutive_skips). Kept as a plain struct instead of hanging off a
// God-object, unlike the Python `self`.
struct BfgsReferenceState {
    double density_norm_change = 0.0;
    double predicted_energy = 0.0;
    int qn_count = 0;
    int consecutive_skips = 0;
};

// Faithful port of the condition at helper_PFCI.py:11118:
//   (density_norm_change > 0.025 and qn_optimization) or predicted_energy > 0
//     or qn_count == 1 or consecutive_skips >= 3
// True  -> reset the BFGS reference point and solve against the exact
//          (freshly rebuilt) reduced Hessian via GLTR
//          ("solve step for original hessian", helper_PFCI.py:11119-11131).
// False -> reuse the running L-BFGS approximation via GLTR-with-operator
//          ("solve bfgs for updated hessian", helper_PFCI.py:11133-11149).
// Note Python `and` binds tighter than `or`, which is preserved here.
bool should_reset_bfgs_reference(const BfgsReferenceState& state, bool qn_optimization);

// Faithful port of the DIFFERENT, 3-clause condition at helper_PFCI.py:11227
// (top-of-outer-pass reference-point-refresh trigger, distinct from
// should_reset_bfgs_reference()'s 4-clause dispatch condition above -- see
// that function's own comment, and BfgsOperator's header doc comment, for
// why the two are not the same condition in the real Python):
//   density_norm_change > 0.025 and qn_optimization or predicted_energy > 0
//     or qn_count == 1
// Note: no `consecutive_skips >= 3` disjunct here, unlike
// should_reset_bfgs_reference(). True -> adopt this pass's freshly-built
// (pre-step) U2/A_tilde/G/reduced_hessian_diagonal as the new BFGS
// reference point and clear its history (BfgsOperator::reset_reference()).
bool should_reset_bfgs_reference_point(const BfgsReferenceState& state, bool qn_optimization);

} // namespace casscf
