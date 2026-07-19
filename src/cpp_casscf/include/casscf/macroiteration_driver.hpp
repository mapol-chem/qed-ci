#pragma once

#include "casscf/casscf_context.hpp"
#include "casscf/types.hpp"

namespace casscf {

// Result of the CI diagonalization + weighted state-average energy + RDM
// build that runs at the top of each macroiteration after the first,
// helper_PFCI.py:2398-2521 (c_get_roots, the weighted avg_energy
// accumulation loop, and build_state_average_rdms).
struct CiStateAverageResult {
    Vector eigenvalues;
    Matrix eigenvectors;
    double avg_energy = 0.0;
};

// Not yet implemented: needs the CI Davidson solver (c_get_roots) and the
// state-averaged RDM machinery (build_state_average_rdms), neither of which
// is part of this trust-region-solver module. See cpp_casscf/README.md,
// "What's still open".
class CiStateAverageSolver {
public:
    virtual ~CiStateAverageSolver() = default;
    virtual CiStateAverageResult solve(const Matrix& eigenvecs_guess) = 0;
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
// Not yet implemented: needs the intermediates-building tensor contractions
// (build_intermediates and friends). See cpp_casscf/README.md.
class InternalOptimizationStep {
public:
    virtual ~InternalOptimizationStep() = default;
    virtual void run(CasscfContext& context, double E0, const Matrix& eigenvecs) = 0;
};

// Corresponds to microiteration_optimization6(U, eigenvecs, c_get_roots,
// convergence_threshold), helper_PFCI.py:10843-12000+: the inner
// microiteration loop that rebuilds intermediates each iteration and
// dispatches to GltrTrustRegionSolver / DavidsonDrivenLstrsSolver / a
// QN-based GLTR variant per solver_selector::select_trs_strategy.
//
// Not yet implemented: needs the intermediates-building tensor contractions.
// This is the natural next home for exercising the already-ported solver
// layer once that dependency exists.
class MicroiterationOptimizationStep {
public:
    virtual ~MicroiterationOptimizationStep() = default;
    virtual void run(const Matrix& U, const Matrix& eigenvecs, double convergence_threshold) = 0;

    // self.U2 in the Python: the resulting orbital rotation from the call
    // just made to run(). Only valid after run() has been called at least
    // once.
    virtual const Matrix& last_U2() const = 0;
};

// Corresponds to the two integral-transformation call sites in the
// macroiteration loop:
//  - transform_internal_rotation ~ c_full_transformation_internal_optimization,
//    called only on the "RESTART MICROITERATION" branch, helper_PFCI.py:2875-2889.
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
