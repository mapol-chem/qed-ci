#pragma once

#include "casscf/davidson_augmented_hessian_solver.hpp"
#include "casscf/trust_region_solver.hpp"
#include <memory>

namespace casscf {

struct DavidsonDrivenLstrsConfig {
    int max_bisection_iter = 50; // count10 == 50 cap, helper_PFCI.py:11797-11798

    // hard_case==2 (interior/near-Newton) is solved with a matrix-free CG
    // instead of the Python's linear_equation_solve/MINRES -- see class doc.
    int cg_max_iter = 200;
    double cg_tol = 1e-8;
};

// Large-dimension counterpart of LstrsSolver: the beta-bisection driver
// inside microiteration_optimization6's `qn_optimization == False`,
// `n_negative > 0` branch (helper_PFCI.py:11337-16085). Structurally the
// same bisection/hard-case algorithm as LstrsSolver -- both are built on
// solve_lstrs_bisection() -- but the two lowest bordered-matrix eigenpairs
// at each trial alpha come from a Davidson subspace approximation
// (DavidsonAugmentedHessianSolver) instead of a direct dense eigh, since
// the full reduced Hessian is only ever available matrix-free here.
//
// DavidsonAugmentedHessianSolver is now a full port (initial guess subspace
// + expansion loop + soft restart), and this class constructs exactly one
// instance per solve() call, reused across every bisection step -- so
// subspace persistence across bisection steps (guess_vector/restart=True in
// the Python, helper_PFCI.py:11401, 11466, 11478) works as in the source.
//
// ONE KNOWN GAP relative to the Python, documented rather than silently
// papered over:
//
// hard_case==2 SOLVED WITH PLAIN CG, NOT linear_equation_solve/MINRES.
//    The Python's interior-step solve (helper_PFCI.py:11855-11878) tries a
//    custom diagonally-preconditioned residual-minimization iterative solver
//    (`LinearRMSolver`, defined in residual_minimization.py -- not yet read)
//    and falls back to scipy's MINRES if that doesn't converge. Since
//    hard_case==2 implies the Hessian is (near-)positive-semidefinite at
//    that point (mu0 > -1e-8), a standard matrix-free CG is the
//    mathematically appropriate substitute; this reuses the already-tested
//    PcgTrustRegionSolver with an effectively-unconstrained trust radius
//    (which makes Steihaug-CG reduce to plain CG) rather than reimplementing
//    a second CG loop.
class DavidsonDrivenLstrsSolver final : public TrustRegionSolver {
public:
    DavidsonDrivenLstrsSolver(std::shared_ptr<const HessianOperator> hessian,
                               std::shared_ptr<const HessianGuessProvider> guess_provider,
                               Vector reduced_hessian_diagonal,
                               DavidsonDrivenLstrsConfig config = {},
                               DavidsonAugmentedHessianConfig davidson_config = {});

    TrustRegionResult solve(const Vector& gradient, double trust_radius) const override;

private:
    std::shared_ptr<const HessianOperator> hessian_;
    std::shared_ptr<const HessianGuessProvider> guess_provider_;
    Vector reduced_hessian_diagonal_;
    DavidsonDrivenLstrsConfig config_;
    DavidsonAugmentedHessianConfig davidson_config_;
};

} // namespace casscf
