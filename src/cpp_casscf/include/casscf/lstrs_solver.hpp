#pragma once

#include "casscf/hessian_operator.hpp"
#include "casscf/trust_region_solver.hpp"
#include <memory>

namespace casscf {

// Dense LSTRS (Rojas, Santos & Sorensen) trust-region subproblem solver.
//
// Port target: the inline solver in internal_optimization3
// (helper_PFCI.py:6996-7548), used for the small active-inactive rotation
// block (dimension n_act_orb * n_in_a) where the Hessian is explicit and
// dense, so a fresh bordered-matrix eigh per bisection step is cheap.
//
// NOTE ON NAMING: earlier versions of this scaffold called this
// "DavidsonLstrsSolver" and cited a "Davidson_augmented_hessian_solve3" as
// the port target. That function is dead code -- it is called only from
// `ah_orbital_optimization`, which is never invoked outside a commented-out
// line in the macroiteration driver (helper_PFCI.py:2377). The real active
// solver for the small block is the inline LSTRS loop ported here. The
// large-dimension counterpart used inside microiteration_optimization6 is a
// genuinely different, subspace-based algorithm -- see
// DavidsonAugmentedHessianSolver.
//
// Algorithm: parametrize the bordered matrix [[alpha, g^T], [g, H]] by
// alpha, safeguard-bisect alpha so the step built from the two lowest
// eigenpairs lands on the trust-region boundary, with dedicated handling
// for the interior-Newton case, the classical hard case (gradient
// orthogonal to the lowest eigenspace), and the "quasi-optimal" case where
// the two lowest eigenvalues are nearly degenerate.
//
// SCOPE: this ports the self-contained per-call subproblem solve (given a
// fixed gradient/Hessian/trust_radius, find the step) -- i.e. what
// TrustRegionSolver::solve() is for. It intentionally does NOT port the
// warm-start fast path that reuses `first_component`/`second_component`
// from the *previous* outer microiteration when already in a hard case
// (helper_PFCI.py:7043-7076) -- that is cross-call state belonging to the
// not-yet-ported microiteration driver, not to a single subproblem solve.
//
// IMPLEMENTATION: this class is now a thin wrapper around
// solve_lstrs_bisection() (lstrs_bisection_core.hpp), which holds the
// ~450-line bisection/hard-case algorithm shared with
// DavidsonDrivenLstrsSolver -- the large-dimension counterpart that solves
// the exact same kind of subproblem but obtains its eigenpairs from a
// Davidson subspace approximation instead of a direct dense eigh. The
// Python has this loop written out twice, nearly verbatim; the C++ doesn't
// repeat that.
class LstrsSolver final : public TrustRegionSolver {
public:
    explicit LstrsSolver(std::shared_ptr<const DenseHessianOperator> hessian,
                          int max_iter = 50);

    TrustRegionResult solve(const Vector& gradient, double trust_radius) const override;

private:
    std::shared_ptr<const DenseHessianOperator> hessian_;
    int max_iter_;
};

} // namespace casscf
