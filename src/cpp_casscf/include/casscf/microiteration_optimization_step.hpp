#pragma once

#include "casscf/internal_optimization_step.hpp" // reuses CasscfPhysicalConstants
#include "casscf/macroiteration_driver.hpp"
#include "casscf/quasi_newton_policy.hpp"
#include "casscf/tensor_types.hpp"
#include "casscf/types.hpp"

namespace casscf {

// Faithful port of microiteration_optimization6(U, eigenvecs, c_get_roots,
// convergence_threshold), helper_PFCI.py:10908-12423 -- the real
// MicroiterationOptimizationStep. NOT the QN (quasi-Newton/BFGS) path: see
// "DOCUMENTED DEVIATIONS" below.
//
// STRUCTURE (confirmed by reading the Python's actual indentation, not
// assumed from its variable names -- an earlier read of this function had
// mis-placed the CI-solve call inside the inner accept branch; it is not
// there):
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
//        (n_negative>0) / PcgTrustRegionSolver-as-plain-CG (||g||<=1e-3,
//        documented substitution, same precedent as
//        DavidsonDrivenLstrsSolver's own hard_case==2 gap) : rotation-generator
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
// DOCUMENTED DEVIATIONS from the Python (all intentional, all flagged):
//
// 1. The quasi-Newton (QN/L-BFGS) path (helper_PFCI.py:11211-11381, the
//    `if qn_optimization == True:` branch, and the `step_norm < 0.05`
//    activation trigger inside the non-QN accept branch that would flip
//    qn_optimization on) is NOT implemented. This port behaves as if
//    QuasiNewtonPolicy::enabled is permanently false: the `step_norm < 0.05`
//    QN-activation check (helper_PFCI.py:12173-12182) is simply not ported,
//    so qn_optimization never turns on and every outer iteration always
//    takes the non-QN branch. A real, load-bearing gap (once the Python's
//    trigger would fire, this changes which solve path executes for the
//    rest of the run) -- not a provably-equivalent substitution -- per the
//    developer's own view (quasi_newton_policy.hpp) that QN "doesn't
//    reliably help MCSCF convergence," a correct non-QN core loop is the
//    lower-risk, higher-value thing to land first. BfgsOperator/
//    should_reset_bfgs_reference/QuasiNewtonPolicy remain ready for whoever
//    wires the QN path in as a follow-up.
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
// 3. gradient-small Newton fallback (||reduced_gradient|| in (1e-7, 1e-3],
//    having already survived the two gradient-norm break checks above it):
//    the Python tries a custom `linear_equation_solve`, falling back to
//    scipy MINRES (helper_PFCI.py:12055-12073, not yet read in detail, not
//    ported). Substituted with PcgTrustRegionSolver at an effectively
//    unconstrained trust radius (reduces Steihaug-CG to plain CG) --
//    mathematically the appropriate substitute the same way
//    DavidsonDrivenLstrsSolver's own hard_case==2 gap already documents (this
//    path is exactly that same "near-PSD, solve directly" regime; hard_case
//    is forced to 2 either way, matching the Python).
class CasscfMicroiterationOptimizationStep final : public MicroiterationOptimizationStep {
public:
    CasscfMicroiterationOptimizationStep(Dimensions dims, CasscfPhysicalConstants constants,
                                          CiStateAverageSolver& ci_solver, int max_microiterations = 20);

    void run(CasscfContext& context, const Matrix& U, Matrix& eigenvecs, double convergence_threshold) override;

    const Matrix& last_U2() const override { return U2_; }

private:
    Dimensions dims_;
    CasscfPhysicalConstants constants_;
    CiStateAverageSolver* ci_solver_;
    int max_microiterations_;
    Matrix U2_;
};

} // namespace casscf
