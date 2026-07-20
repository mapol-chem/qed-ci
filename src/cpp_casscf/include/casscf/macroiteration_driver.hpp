#pragma once

#include "casscf/casscf_context.hpp"
#include "casscf/tensor_types.hpp"
#include "casscf/types.hpp"

#include <optional>

namespace casscf {

// Result of the CI diagonalization + weighted state-average energy + RDM
// build that runs at the top of each macroiteration after the first,
// helper_PFCI.py:2398-2521 (c_get_roots, the weighted avg_energy
// accumulation loop, and build_state_average_rdms).
struct CiStateAverageResult {
    Vector eigenvalues;
    Matrix eigenvectors;
    double avg_energy = 0.0;

    // build_state_average_rdms(eigenvecs)'s output (helper_PFCI.py:7992+),
    // which in the Python is written directly into the persistent
    // self.D_tu_avg/self.D_tuvw_avg/self.Dpe_tu_avg instance attributes
    // right after this same solve -- MacroiterationDriver::run copies these
    // into CasscfContext (see its doc comment) so InternalOptimizationStep
    // can read them later in the same macroiteration.
    Matrix D_tu_avg;    // (n_act_orb, n_act_orb)
    Tensor4 D_tuvw_avg; // (n_act_orb, n_act_orb, n_act_orb, n_act_orb)
    Matrix Dpe_tu_avg;  // (n_act_orb, n_act_orb)

    // self.constint[8] == 0 after c_get_roots (helper_PFCI.py:7699,
    // 7712-7728-ish): whether the CI Davidson diagonalization itself
    // converged (distinct from avg_energy's own convergence, which
    // MacroiterationDriver checks separately). Read by
    // InternalOptimizationStep's own convergence test
    // (helper_PFCI.py:7808-7810: `gradient_norm < 1e-4 and
    // self.constint[8] == 0`). Defaults to true so mocked
    // CiStateAverageSolver implementations that don't model CI-solver
    // convergence (e.g. test_macroiteration_driver.cpp's) don't need to set
    // it explicitly.
    bool ci_diagonalization_converged = true;

    // The achieved RMS residual norm across roots, helper_PFCI.py:12433
    // (current_residual = self.constdouble[4]) -- confirmed by reading
    // ci_solver.c directly (get_roots/build_H_diag_cas_spin's convergence
    // loop, e.g. ~1625-1669): constdouble[4] is an INPUT (the requested
    // Davidson convergence threshold) on the way in, but the C function
    // OVERWRITES it before returning with the actual achieved residual
    // norm averaged over davidson_roots -- a genuine second output, not
    // just an echoed threshold. Used by MicroiterationOptimizationStep's
    // QN-adjacent "total_norm" outer-loop break (helper_PFCI.py:11243-11256).
    double residual_norm = 0.0;
};

// Not yet implemented: needs the CI Davidson solver (c_get_roots) and the
// state-averaged RDM machinery (build_state_average_rdms), neither of which
// is part of this trust-region-solver module. See cpp_casscf/README.md,
// "What's still open".
//
// use_staged_inputs distinguishes this interface's two genuinely different
// call sites (confirmed by reading the real Python's own two distinct
// c_get_roots/c_H_diag_cas_spin call sites, not assumed):
//  - false (default): MacroiterationDriver's own top-of-macroiteration
//    solve (helper_PFCI.py:2424-2521) -- integrals recomputed fresh from
//    context.H_spatial2/J/K, which are correct and current at that exact
//    point (right after IntegralTransformer::transform_macroiteration).
//  - true: InternalOptimizationStep's and MicroiterationOptimizationStep's
//    own inner accept/reject-loop calls (helper_PFCI.py:7748,
//    internal_optimization3; 12418, microiteration_optimization6) -- both
//    real Python call sites read LOCAL/self.-prefixed staged quantities
//    (self.gkl2/occupied_J/occupied_fock_core/occupied_d_cmo, plus a
//    core-energy scalar -- self.E_core for internal_optimization3,
//    self.E_core2 for microiteration_optimization6) that reflect the
//    orbital rotation accumulated *within that same call*, not
//    self.H_spatial2/J/K (which don't update until each function's own
//    convergence). An implementation honoring this flag must read
//    context.gkl2/occupied_J/occupied_fock_core/occupied_d_cmo/E_core2
//    instead of recomputing from context.H_spatial2/J/K -- callers using
//    `true` are responsible for having already committed the correct
//    per-call-site values into those same context fields (matching
//    InternalOptimizationStep's own occupied_J/K commit and
//    MicroiterationOptimizationStep's commit_ci_solver_inputs) before
//    calling solve(), including staging context.E_core2 with whichever
//    core-energy scalar their own Python call site actually uses.
class CiStateAverageSolver {
public:
    virtual ~CiStateAverageSolver() = default;

    // davidson_threshold_override/davidson_maxiter_override: helper_PFCI.py:
    // 12395-12404 -- microiteration_optimization6's CI-solve call resets
    // constdouble[4]/constint[8] to 1e-9/5 every pass by default, but
    // overrides them to 0.1*||reduced_gradient||/10000 once qn_count > 0
    // (a looser, cheaper CI resolve once the QN path is active and orbital
    // steps are already small). std::nullopt (default, both call sites
    // that predate QN) means "use the constructor-time
    // CasscfCiConfig::davidson_threshold/davidson_maxiter unchanged" --
    // only MicroiterationOptimizationStep's QN-active passes pass a real
    // value here.
    virtual CiStateAverageResult solve(const Matrix& eigenvecs_guess, bool use_staged_inputs = false,
                                        std::optional<double> davidson_threshold_override = std::nullopt,
                                        std::optional<int> davidson_maxiter_override = std::nullopt) = 0;
};

// Corresponds to internal_optimization3(E0, eigenvecs), helper_PFCI.py:6847-7961:
// builds the small active-inactive rotation intermediates and dispatches to
// LstrsSolver. Like the Python -- which mutates self.U_total (and, on
// acceptance, self.H_spatial2/self.d_cmo/self.J/self.K/self.occupied_*)
// directly at helper_PFCI.py:7787-7805 rather than returning a step -- an
// implementation of this reads and mutates `context` in place; the driver
// does not read anything back from the return value. context must be the
// *same* CasscfContext instance the caller passed to MacroiterationDriver::run,
// so internal_optimization3's own rotation of H_spatial2/d_cmo/U_total is
// visible to (and composes correctly with) the rotation
// MacroiterationDriver::run applies itself later in the same macroiteration
// (helper_PFCI.py:2925-2937) -- see CasscfContext's doc comment for why a
// single shared instance is required here, not per-call copies.
//
// eigenvecs is taken by non-const reference (not the Matrix& elsewhere in
// this header that stay read-only) because internal_optimization3's own
// accept branch re-diagonalizes the CI problem and overwrites the caller's
// `eigenvecs` array in place (c_get_roots writes into the same numpy array
// object passed in, helper_PFCI.py:7699) -- that update must be visible to
// MacroiterationDriver::run's local `eigenvecs` after this call returns,
// since it's reused by the microiteration step right after
// (helper_PFCI.py:2841-2850).
//
// See internal_optimization_step.hpp for the real implementation.
class InternalOptimizationStep {
public:
    virtual ~InternalOptimizationStep() = default;
    virtual void run(CasscfContext& context, double E0, Matrix& eigenvecs) = 0;
};

// Corresponds to microiteration_optimization6(U, eigenvecs, c_get_roots,
// convergence_threshold), helper_PFCI.py:10908-12423: the inner
// microiteration loop that rebuilds intermediates each outer iteration and
// dispatches to GltrTrustRegionSolver / DavidsonDrivenLstrsSolver / a
// QN-based GLTR variant per solver_selector::select_trs_strategy.
//
// eigenvecs is taken by non-const reference, not const& -- same reasoning as
// InternalOptimizationStep::run() above: microiteration_optimization6 calls
// c_get_roots exactly once per outer ("microiteration") loop pass (helper_PFCI.py:
// ~12225, after the inner orbital-optimization-step loop completes for that
// pass -- NOT once per accepted inner step), and c_get_roots mutates its
// eigenvecs argument in place. MacroiterationDriver::run's local `eigenvecs`
// must observe that update, since it's reused by the restart branch's second
// microiteration_step_->run() call and by the final result.eigenvectors.
//
// context: same sharing requirement as InternalOptimizationStep::run() --
// must be the same CasscfContext instance the caller passed to
// MacroiterationDriver::run(). A real implementation reads context.J/K/
// H_spatial2/d_cmo/D_tu_avg/D_tuvw_avg/Dpe_tu_avg (build_intermediates'
// inputs) and, once per outer microiteration pass (see run()'s own doc
// comment above), commits context.gkl2/occupied_J/occupied_fock_core/
// occupied_d_cmo/E_core2/H_diag3 -- the CI-solver-input staging fields --
// before calling the injected CiStateAverageSolver. Note these are the same
// context fields InternalOptimizationStep commits on its own, different
// timing/schedule within one macroiteration (see
// microiteration_optimization_step.hpp's class doc comment for why sharing
// those fields between the two Steps is the intended design, not an
// accident of naming).
//
// See microiteration_optimization_step.hpp for the real implementation.
class MicroiterationOptimizationStep {
public:
    virtual ~MicroiterationOptimizationStep() = default;
    virtual void run(CasscfContext& context, const Matrix& U, Matrix& eigenvecs, double convergence_threshold) = 0;

    // self.U2 in the Python: the resulting orbital rotation from the call
    // just made to run(). Only valid after run() has been called at least
    // once.
    virtual const Matrix& last_U2() const = 0;
};

// Corresponds to the integral-transformation call sites in the
// macroiteration/internal-optimization loops, both wrapping
// c_full_transformation_internal_optimization:
//  - transform_internal_rotation ~ the "RESTART MICROITERATION" branch,
//    helper_PFCI.py:2875-2889 (U_delta from the microiteration step's
//    internal-rotation correction), AND internal_optimization3's own
//    convergence, helper_PFCI.py:7803 (U_delta == self.U1, the accumulated
//    internal-rotation-only rotation from its bisection loop) -- both are
//    "apply an internal (active-inactive-only) integral transformation",
//    the same operation on different rotation matrices, so they share this
//    one interface method. A real implementation must be constructed once
//    and shared between MacroiterationDriver and InternalOptimizationStep
//    (see internal_optimization_step.hpp), the same sharing pattern
//    CasscfContext already establishes for orbital/integral state.
//  - transform_macroiteration ~ c_full_transformation_macroiteration /
//    transform_JK_with_df (density-fitted variant), helper_PFCI.py:2944-2973,
//    run once per macroiteration on the accumulated U_total.
//
// Not yet implemented.
class IntegralTransformer {
public:
    virtual ~IntegralTransformer() = default;
    virtual void transform_internal_rotation(const Matrix& U_delta) = 0;
    virtual void transform_macroiteration(const Matrix& U_total) = 0;
};

struct MacroiterationDriverConfig {
    Dimensions dims;
    int max_macroiterations = 1000;                     // helper_PFCI.py:2394
    double energy_convergence = 1e-10;                   // helper_PFCI.py:2527
    double internal_rotation_restart_threshold = 1e-4;   // helper_PFCI.py:2865
    double first_iteration_convergence_threshold = 1e-3; // helper_PFCI.py:2843
    double later_iteration_convergence_threshold = 1e-4; // helper_PFCI.py:2845
};

struct MacroiterationResult {
    bool converged = false;
    int macroiterations_run = 0;
    double avg_energy = 0.0;
    Vector eigenvalues;
    Matrix eigenvectors;
    // No U_total field here: it lives on the CasscfContext the caller
    // passed to run() (and is updated in place there), so it isn't
    // duplicated into the result -- see CasscfContext's doc comment on why
    // a single shared instance, not per-call copies, is required.
};

// Faithful port of the macroiteration while-loop *shape* at
// helper_PFCI.py:2394-3060 -- orchestration only. CI diagonalization,
// intermediates-building/solver dispatch, and integral transformation are
// behind the interfaces above, none of which are implemented yet.
//
// Deliberately excludes the convergence-time state-analysis / dipole /
// natural-orbital reporting block (helper_PFCI.py:2538-2801) and the
// per-macroiteration energy bookkeeping print block
// (helper_PFCI.py:2976-3057): neither touches the trust-region solver
// orchestration this module covers.
//
// One documented deviation: the Python has a
// `if macroiteration >= 1000: self.casscf_converged = False` check inside
// the loop body (helper_PFCI.py:2529-2531) that is unreachable dead code --
// the enclosing `while macroiteration < 1000` guard already guarantees
// macroiteration < 1000 whenever the body runs. This port just falls out of
// the loop with converged == false if max_macroiterations is exhausted,
// which is the behavior that dead check was presumably meant to express.
class MacroiterationDriver {
public:
    MacroiterationDriver(MacroiterationDriverConfig config,
                          CiStateAverageSolver& ci_solver,
                          InternalOptimizationStep& internal_step,
                          MicroiterationOptimizationStep& microiteration_step,
                          IntegralTransformer& integral_transformer);

    // eigenvecs0 / avg_energy0: the CI state and state-averaged energy
    // already available before the loop starts (computed by the caller
    // exactly like the Python's pre-loop CI solve feeds into
    // `avg_energy`/`eigenvecs`, helper_PFCI.py:2270-2275-ish).
    // context: caller-owned, caller-initialized (context.H_spatial2 /
    // context.d_cmo set to the starting one-electron Hamiltonian / PF
    // dipole-coupling integrals; context.U_total is reset to identity by
    // this call, matching the Python's self.U_total = eye(nmo) at the top
    // of the macroiteration loop). Read and mutated in place by this driver
    // and by internal_step/microiteration_step/integral_transformer, all of
    // which must share this same instance -- see CasscfContext's doc
    // comment.
    MacroiterationResult run(Matrix eigenvecs0, double avg_energy0, CasscfContext& context);

private:
    MacroiterationDriverConfig config_;
    CiStateAverageSolver* ci_solver_;
    InternalOptimizationStep* internal_step_;
    MicroiterationOptimizationStep* microiteration_step_;
    IntegralTransformer* integral_transformer_;
};

} // namespace casscf
