#pragma once

#include "casscf/types.hpp"
#include <functional>

namespace casscf {

// Two lowest eigenpairs of the (n+1)-dim bordered matrix [[alpha,g^T],[g,H]]
// at a given trial alpha, in the FULL n+1 space (border component at
// index 0 of each eigenvector).
struct TwoLowestEigenpairs {
    double mu0 = 0.0, mu1 = 0.0;
    Vector w0, w1;
};

// Returns the two lowest bordered-matrix eigenpairs at the given alpha.
// `alpha_range` is |alpha_u - alpha_l| from the caller's current bracket
// (used by the Davidson-driven implementation to decide nroots=1 vs 2, see
// DavidsonAugmentedHessianSolver); `restart` is false only for the very
// first call of a solve() invocation.
using EigenpairAtAlphaFn = std::function<TwoLowestEigenpairs(double alpha, double alpha_range, bool restart)>;

// Standard trust-region quadratic model g^T x + 0.5 x^T H x.
using ModelValueFn = std::function<double(const Vector& x)>;

// Solves H x = -g for the "hard_case==2" (near-)interior branch, where H is
// known to be (near-)positive-semidefinite at that point (mu0 > -1e-8).
using HardCase2SolveFn = std::function<Vector()>;

// Shared secular-equation bisection + hard-case resolution used by both the
// dense small-block LSTRS solver (LstrsSolver) and the large-block
// Davidson-driven LSTRS solver (DavidsonDrivenLstrsSolver).
//
// The Python has this ~450-line loop inlined twice, nearly verbatim:
// internal_optimization3 (helper_PFCI.py:6996-7548, small dense block) and
// microiteration_optimization6's non-QN branch (helper_PFCI.py:11337-16085,
// large matrix-free block). They differ only in:
//   1. how the two lowest bordered-matrix eigenpairs are obtained at each
//      trial alpha -- a direct dense eigh (projection_step2) for the small
//      block, vs. a Davidson subspace approximation
//      (Davidson_augmented_hessian_solve6) for the large block;
//   2. how the "hard_case==2" (interior, near-Newton) step is solved -- a
//      direct dense solve for the small block, vs. an iterative matrix-free
//      solve (linear_equation_solve + MINRES fallback in the Python; not
//      ported here, see DavidsonDrivenLstrsSolver's doc comment) for the
//      large block;
//   3. the quadratic-model evaluation used for the hard_case==3
//      "quasi-optimal" test -- dense (internal_optimization_predicted_energy)
//      vs. matrix-free (microiteration_predicted_energy2), which are the
//      same formula (2*g.x + x^T H x) either way.
// Rather than port that duplication twice, this is the one shared
// implementation; LstrsSolver and DavidsonDrivenLstrsSolver each supply the
// three callbacks above.
TrustRegionResult solve_lstrs_bisection(const Vector& gradient,
                                         const Vector& hessian_diagonal,
                                         double trust_radius,
                                         int max_iter,
                                         const EigenpairAtAlphaFn& eigenpairs_at,
                                         const ModelValueFn& model_value,
                                         const HardCase2SolveFn& hard_case2_solve);

} // namespace casscf
