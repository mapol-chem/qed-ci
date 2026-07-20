#pragma once

#include "casscf/internal_optimization_step.hpp" // reuses CasscfPhysicalConstants
#include "casscf/macroiteration_driver.hpp"
#include "casscf/quasi_newton_policy.hpp"
#include "casscf/tensor_types.hpp"
#include "casscf/types.hpp"

namespace casscf {

// Faithful port of microiteration_optimization6(U, eigenvecs, c_get_roots,
// convergence_threshold), helper_PFCI.py:10908-12423 -- the real
// MicroiterationOptimizationStep, including the QN (quasi-Newton/BFGS) path
// (see "DOCUMENTED DEVIATIONS" below for exactly what's ported vs.
// intentionally omitted there).
//
// STRUCTURE (confirmed by reading the Python's actual indentation, not
// assumed from its variable names -- an earlier read of this function had
// mis-placed the CI-solve call inside the inner accept branch; it is not
// there). This describes the non-QN branch specifically -- see
// "DOCUMENTED DEVIATIONS" for the separate QN flat block that steps 3-4
// dispatch to instead, once qn_optimization has activated:
//
//   outer "microiteration" loop (while microiteration < N_microiterations):
//     1. build_intermediates once -- sets this outer iteration's *fixed
//        reference point* (fi.A/fi.G/fi.fock_core/fi.E_core/fi.active_fock_core/
//        fi.active_twoeint/fi.L, all read-only for the rest of this outer
//        iteration) and context.E_core (mirrors self.E_core's reassignment).
//     2. zero_energy (local helper below) + current_energy; small-energy-change
//        break (microiteration >= 2 only).
//     3. build_gradient + a plain zero-pad embed of A_tilde (NOT
//        embed_and_symmetrize_A_tilde -- see intermediates.hpp's own doc
//        comment on that function; build_hessian_diagonal needs the
//        unsymmetrized full embed) + build_hessian_diagonal +
//        reduced_gradient extraction + n_negative.
//     4. inner "orbital optimization step" loop (while orbital_optimization_step
//        < N_orbital_optimization_steps): gradient-norm break checks: solve
//        via GltrTrustRegionSolver (n_negative==0) / DavidsonDrivenLstrsSolver
//        (n_negative>0) / linear_equation_solve with a real minres_solve
//        fallback (||g||<=1e-3, a faithful port, not a substitution -- see
//        "DOCUMENTED DEVIATIONS") : rotation-generator
//        unpack -> build_unitary_matrix -> trial energy via
//        microiteration_exact_energy : single shared accept/reject test
//        (energy_change < 0.0 || hard_case == 2). On accept: commit U2,
//        step_control, rebuild gradient/hessian/reduced_gradient on the SAME
//        fi.A/fi.G (not a fresh build_intermediates call), and call
//        microiteration_ci_integrals_transform to refresh this *local*
//        (active_fock_core, active_twoeint, d_cmo, E_core2) tuple -- these
//        are local, function-scope-persistent state in the Python (declared
//        once before the outer loop, NOT reset at the top of each outer
//        pass), not anything on `self`/context, so this port mirrors that by
//        keeping them as run()-local variables that only change on an
//        accepted inner step (or the convergence-fallback copy below).
//     5. AFTER the inner loop (regardless of QN/non-QN branch, at the same
//        indentation level as the branch dispatch, i.e. unconditionally once
//        per OUTER pass -- confirmed via precise indentation counting, not
//        guessed): if the inner loop broke via a gradient-norm convergence
//        check *and* accepted nothing this pass (convergence==1 and
//        count==0, helper_PFCI.py:12300), fall back to this outer
//        iteration's own reference-point values (fi.active_fock_core/
//        fi.active_twoeint/context.d_cmo/fi.E_core) for that same local
//        tuple. Then unconditionally: build the CI-solver-ready
//        occupied-sized gkl2/occupied_J/occupied_fock_core/occupied_d_cmo
//        from that tuple and commit them into `context` (see below for why
//        context, not a fresh return channel), then call
//        ci_solver_->solve(eigenvecs) exactly once for this outer pass.
//
// WHY COMMIT INTO context.gkl2/occupied_J/occupied_fock_core/occupied_d_cmo:
// these are, textually, fresh LOCAL variables in the Python (NOT
// self.occupied_J/self.occupied_fock_core/self.occupied_d_cmo/self.gkl2 --
// confirmed by grep: this function never reads or writes those self.
// attributes at all), distinct from the same-named fields
// InternalOptimizationStep commits. But CiStateAverageSolver::solve()'s
// interface (macroiteration_driver.hpp) takes only eigenvecs_guess, with no
// other channel to receive them, and its own doc comment already
// anticipates a real implementation reading them off context. Since both
// Steps' local/self occupied_J-shaped values serve the *same* physical role
// (whatever occupied-restricted integrals c_get_roots should use for its
// next call) and are never read across the boundary between the two Steps
// in a way that would observe a stale value (internal_optimization3 always
// runs to completion, including its own final CI re-solve, before
// microiteration_optimization6 starts; and nothing after
// microiteration_optimization6 in one macroiteration reads context.occupied_J
// again until the next macroiteration's IntegralTransformer::
// transform_macroiteration call rebuilds it from scratch), reusing these
// context fields as a shared "CI-solver input staging area" between the two
// Steps reproduces the Python's actual data flow rather than inventing a new
// one -- just consolidated through one field instead of two disjoint
// variables that happen to feed the same downstream C function role.
// context.E_core2 was added (additive) to CasscfContext for the same reason
// (self.E_core2 -- yes, an actual `self.` attribute here, unlike the other
// four -- has no other home).
//
// DOCUMENTED DEVIATIONS from the Python (both intentional, both flagged;
// the gradient-small Newton fallback that used to be a third, documented
// substitution here is now a faithful port -- see
// linear_equation_solve.hpp/linear_rm_solver.hpp -- with one remaining,
// inherent (not fixable) non-reproducibility flagged on that function
// itself, not repeated here):
//
// 1. The quasi-Newton (QN/L-BFGS) path (helper_PFCI.py:11258-11428, the
//    `if qn_optimization == True:` branch) IS implemented, gated by
//    `QuasiNewtonPolicy::enabled` (default true, matching the Python's
//    de-facto behavior once the `step_norm < 0.05` activation trigger,
//    helper_PFCI.py:12222-12228, fires inside the non-QN accept branch).
//    The QN branch is a flat block (not wrapped in the inner
//    orbital-optimization-step while loop): it computes and unconditionally
//    accepts exactly one trial step per outer pass, dispatching via
//    should_reset_bfgs_reference() to either the exact reduced Hessian at a
//    frozen BFGS reference point (U_zero/A_tilde_zero/G_zero, via
//    orbital_sigma3) or the running BfgsOperator approximation, both fed
//    through GltrTrustRegionSolver -- confirmed the Python's QN branch
//    exclusively uses GLTR, never Davidson/LSTRS or the Newton fallback.
//    Several pieces of Python state confirmed dead/inert by direct grep
//    across this whole function are deliberately NOT reinvented into
//    "working" behavior, matching this codebase's existing precedent for
//    Python dead code: `consecutive_skips` (never incremented anywhere in
//    the active path -- the reset-to-0 inside the QN branch's own
//    exact-Hessian sub-branch is ported for faithfulness, but it's a no-op
//    either way), `s_history`/`y_history` (populated with `[]` at function
//    entry, never appended to or read -- `bfgs_history`, via BfgsOperator,
//    is the real mechanism), and the activation-time reference-point
//    capture at helper_PFCI.py:12229-12233 (provably superseded before any
//    read: nothing reads self.U_zero/A_tilde_zero/G_blocks_zero/
//    reduced_hessian_diagonal_zero between that write and the very next
//    outer pass's top-of-loop reset, helper_PFCI.py:11227-11235, which
//    fires unconditionally on that next pass since qn_count==1 -- so this
//    port only has the one top-of-loop reset call site).
//
//    The top-of-loop reference-point-refresh check (helper_PFCI.py:11227,
//    should_reset_bfgs_reference_point()) is gated here on `qn_optimization`
//    even though the Python evaluates it unconditionally every pass
//    (including pre-activation ones, since `predicted_energy` starts at the
//    sentinel 10 > 0): this is a behavior-preserving optimization, not a
//    deviation -- nothing reads U_zero/A_tilde_zero/G_zero/
//    reduced_hessian_diagonal_zero until the QN branch itself does, and the
//    very first QN pass always has qn_count==1, which unconditionally
//    forces a fresh reset that pass regardless of what any earlier,
//    pre-activation reset attempt would have produced. Gating just avoids
//    speculatively constructing/copying a BfgsOperator on every pass of
//    every non-QN run.
//
//    Two further Python-state subtleties, confirmed by direct trace rather
//    than assumed, needed for the CI-solve-input-staging fallback
//    (`accepted_count == 0`, at the end of run()'s per-pass tail) to remain
//    correct once QN is active: `accepted_count` (Python's bare local
//    `count`) and the small-gradient-convergence flag (Python's bare local
//    `convergence`) are NOT reset every outer pass -- `count = 0` only
//    appears once in the Python, at the top of the non-QN branch itself
//    (helper_PFCI.py:11438); `convergence` is initialized once before the
//    whole outer loop (helper_PFCI.py:11002) and never reset back to 0
//    anywhere. Both are therefore promoted to run()-scope-persistent locals
//    here (not re-declared inside the outer while loop, as an earlier,
//    QN-less version of this port had them) -- `accepted_count` reset to 0
//    only at the top of the non-QN branch, `small_gradient_convergence`
//    latched true forever once either inner-loop gradient-norm break fires.
//    In practice this only changes observable behavior for edge cases
//    reachable after QN activates (the QN branch always accepts, so
//    `accepted_count` only grows from there and the fallback naturally
//    never fires again in QN mode); confirmed to reproduce the existing
//    all-zero-gradient regression test's behavior unchanged, since that
//    test hits the gradient-small break on every single pass regardless of
//    the reset timing.
//
// 2. The cross-microiteration hard_case==1 "warm start" shortcut inside the
//    Davidson bisection (helper_PFCI.py:11467-11500, reusing the *previous*
//    inner iteration's bordered-eigenproblem root components algebraically)
//    is not re-ported here either -- it already wasn't ported into
//    DavidsonDrivenLstrsSolver (see that class's own doc comment), and this
//    class calls that solver fresh each inner iteration, so it inherits the
//    same performance-only (not correctness) gap CasscfInternalOptimizationStep
//    already documents for the analogous shortcut in internal_optimization3.
//
// The gradient-small Newton fallback (||reduced_gradient|| in (1e-7, 1e-3],
// having already survived the two gradient-norm break checks above it) is
// now a faithful port of the Python's own `linear_equation_solve`
// (LinearRMSolver-based) falling back to real `minres_solve` on
// non-convergence (helper_PFCI.py:12103-12137) -- previously substituted
// with `PcgTrustRegionSolver` at an effectively unconstrained trust radius,
// which is no longer needed. `hard_case` is still forced to `2` either way,
// matching the Python.
class CasscfMicroiterationOptimizationStep final : public MicroiterationOptimizationStep {
public:
    CasscfMicroiterationOptimizationStep(Dimensions dims, CasscfPhysicalConstants constants,
                                          CiStateAverageSolver& ci_solver, int max_microiterations = 20,
                                          QuasiNewtonPolicy qn_policy = {});

    void run(CasscfContext& context, const Matrix& U, Matrix& eigenvecs, double convergence_threshold) override;

    const Matrix& last_U2() const override { return U2_; }

private:
    Dimensions dims_;
    CasscfPhysicalConstants constants_;
    CiStateAverageSolver* ci_solver_;
    int max_microiterations_;
    QuasiNewtonPolicy qn_policy_;
    Matrix U2_;
};

} // namespace casscf
