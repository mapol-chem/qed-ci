#pragma once

#include "casscf/tensor_types.hpp"
#include "casscf/types.hpp"

#include <functional>

namespace casscf {

struct LinearEquationSolveResult {
    Vector solution;
    bool converged = false;
};

// Faithful port of PFHamiltonianGenerator.linear_equation_solve
// (helper_PFCI.py:16342-16399) -- solves the reduced-space Newton linear
// system H*x + reduced_gradient = 0 via LinearRMSolver (linear_rm_solver.hpp),
// diagonal-preconditioned by `denom`, with the Hessian-vector product
// supplied by the caller as `apply`. In the Python this is always
// self.mv2/build_sigma_reduced5 (a *different* implementation strategy for
// the identical Hessian-vector product than build_sigma_reduced7, which
// orbital_sigma3 ports -- same substitution precedent already established
// for microiteration_predicted_energy2's own use of orbital_sigma3 in
// place of build_sigma_reduced4, see microiteration_energy.hpp's doc
// comment) -- but the Python has this exact formula (denom =
// self.reduced_hessian_diagonal, max_iter = 20, conv_thresh = 1e-6, MINRES
// rtol = 1e-6 on non-convergence) written out TWICE, at two different
// call sites: CasscfMicroiterationOptimizationStep's own gradient-small
// fallback (helper_PFCI.py:12103-12137, uses orbital_sigma3 directly, see
// the convenience overload below) and DavidsonDrivenLstrsSolver's
// hard_case==2 resolution inside the Davidson bisection
// (helper_PFCI.py:12052-12094, uses the SAME self.mv2 but through a
// generic HessianOperator on the C++ side) -- this generic overload is
// the shared core both wrap, same "duplicated in Python, one shared core
// in C++" precedent already established for the LSTRS bisection itself
// (lstrs_bisection_core.hpp).
//
// One inherent, NOT fixable non-reproducibility, worth flagging explicitly:
// when ||reduced_gradient|| < 1e-3 (always true at this function's only
// call site -- CasscfMicroiterationOptimizationStep only reaches this
// branch when ||reduced_gradient|| <= 1e-3), helper_PFCI.py:16347-16358
// draws a normalized RANDOM probe vector to seed the initial residual.
// This port draws its own (C++ <random>, not numpy's) -- the specific
// probe vector, and hence the specific Krylov subspace built, will differ
// from any given Python run. This does not bias the SOLUTION accuracy (an
// arbitrary initial probe direction doesn't change where a
// residual-minimization method converges to for a well-conditioned
// system), only which intermediate subspace path gets there -- same
// category of intrinsic, already-documented non-reproducibility as this
// module's GLTR noise-capture handling, just not captured/replayed here
// (this path fires far less often than GLTR's; capturing Python's actual
// realized probe vector was judged not worth a new dump hook for this port).
//
// `random_seed` (default 0, matching GltrConfig::random_seed's own default
// and doc-comment reasoning): a *fresh* std::mt19937 is seeded with this
// value on every call (not a persistent/advancing generator) -- same
// pattern GltrTrustRegionSolver::solve() already uses for its own noise
// draw. This makes a given call's random probe fully deterministic and
// reproducible run-to-run (this port's earlier std::random_device-seeded
// version was NOT -- confirmed empirically to cause macroiterations_run to
// vary between repeated executions of the same end-to-end run, an
// unwanted extra source of noise on top of the inherent Python-vs-C++
// mismatch this doc comment already describes). Does not make this port
// match Python's specific draws (numpy's RNG is a different algorithm and
// is itself unseeded by default -- see cpp_casscf/validation/dump_lih_case.py
// for a temporary np.random.seed(...) matching this file's determinism
// goal on the Python side, for controlled comparison).
LinearEquationSolveResult linear_equation_solve(const std::function<Vector(const Vector&)>& apply,
                                                  const Vector& reduced_gradient, const Vector& denom, int max_iter,
                                                  double conv_thresh, unsigned int random_seed = 0);

// Convenience overload for CasscfMicroiterationOptimizationStep's own use,
// wrapping the generic version above with orbital_sigma3 as `apply`.
LinearEquationSolveResult linear_equation_solve(const Matrix& U, const Matrix& A_tilde, const Tensor4& G,
                                                  const Vector& reduced_gradient, const Vector& denom, int max_iter,
                                                  double conv_thresh, const Dimensions& dims,
                                                  unsigned int random_seed = 0);

} // namespace casscf
