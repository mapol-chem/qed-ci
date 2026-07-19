#pragma once

#include "casscf/macroiteration_driver.hpp"
#include "casscf/types.hpp"

namespace casscf {

// The molecule-level physical constants internal_optimization3 (and the
// intermediates it builds) need but that never change across a CASSCF run
// -- as opposed to the state-averaged RDMs (StateAverageData's other
// fields), which are refreshed every macroiteration and live on
// CasscfContext instead (see CasscfContext::D_tu_avg and friends).
// Constructor-injected into CasscfInternalOptimizationStep once, the same
// way Dimensions is.
struct CasscfPhysicalConstants {
    int N_p = 0;
    int num_det = 0;
    double omega = 0.0;
    double Enuc = 0.0;
    double d_c = 0.0;
    double d_exp = 0.0;
    Vector weight; // (davidson_roots) -- state-average weights
};

// Faithful port of internal_optimization3(E0, eigenvecs), helper_PFCI.py:
// 6847-7961 -- the real InternalOptimizationStep. Builds the small
// active-inactive rotation intermediates (build_intermediates_internal /
// build_gradient_and_hessian, intermediates.hpp), dispatches each
// microiteration's trust-region subproblem to the already-tested
// LstrsSolver, evaluates the trial step via internal_transformation /
// internal_optimization_exact_energy / internal_optimization_predicted_energy
// / step_control (internal_optimization.hpp), and on acceptance commits the
// result into CasscfContext and re-diagonalizes the CI problem via the
// injected CiStateAverageSolver.
//
// DOCUMENTED DEVIATIONS from the Python (both intentional, both flagged
// rather than silently absorbed -- see cpp_casscf/README.md's
// "internal_optimization_step.hpp/.cpp" section for the full reasoning):
//
// 1. Does NOT port the cross-microiteration "hard_case==1 warm start"
//    shortcut (helper_PFCI.py:7043-7076, `if hard_case == 1 and
//    reduce_step == 1 and ...: adjusted_step = first_component + ...`),
//    which reuses the *previous* microiteration's bordered-eigenproblem
//    root components to build the new step algebraically instead of
//    re-running the full bisection from scratch after a trust-radius
//    shrink. LstrsSolver's own doc comment already flags this as
//    out-of-scope for a single subproblem solve; this class is the
//    "not-yet-ported microiteration driver" that comment refers to, and it
//    still doesn't port it, because LstrsSolver::solve() is fully
//    self-contained -- calling it again with the shrunk trust_radius
//    independently re-derives the same hard-case step through the full
//    bisection, just without the cheap shortcut. This is a performance
//    difference (more Lanczos-like bisection iterations on repeated
//    hard-case rejections), not a correctness one.
//
// 2. `bool ci_converged` is initialized to false at the top of every run()
//    call, not carried over from a previous internal_optimization3
//    invocation the way self.constint[8] persists across the whole CASSCF
//    run as a instance attribute in the Python. Substituted because this
//    class has no cross-call channel for that flag (and no real
//    CiStateAverageSolver exists yet to source an initial value from) --
//    the practical effect is that a run() call cannot satisfy its own
//    convergence check on iteration 0 before its own first CI solve, which
//    is the only case this could ever differ from the Python's actual
//    per-run behavior (a run() call whose first LSTRS step already both
//    lands within gradient tolerance AND happens to be rejected outright
//    can't happen simultaneously, since the convergence check only runs
//    after the accept/reject branch either commits a fresh CI solve or
//    leaves the previous iteration's -- already tested -- gradient/ci_converged
//    pair unchanged).
class CasscfInternalOptimizationStep final : public InternalOptimizationStep {
public:
    CasscfInternalOptimizationStep(Dimensions dims, CasscfPhysicalConstants constants,
                                    CiStateAverageSolver& ci_solver, IntegralTransformer& integral_transformer,
                                    int max_microiterations = 20, int lstrs_max_iter = 50);

    void run(CasscfContext& context, double E0, Matrix& eigenvecs) override;

private:
    Dimensions dims_;
    CasscfPhysicalConstants constants_;
    CiStateAverageSolver* ci_solver_;
    IntegralTransformer* integral_transformer_;
    int max_microiterations_;
    int lstrs_max_iter_;
};

} // namespace casscf
