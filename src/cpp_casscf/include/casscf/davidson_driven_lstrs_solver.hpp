#pragma once

#include "casscf/davidson_augmented_hessian_solver.hpp"
#include "casscf/trust_region_solver.hpp"
#include <memory>

namespace casscf {

struct DavidsonDrivenLstrsConfig {
    int max_bisection_iter = 50; // count10 == 50 cap, helper_PFCI.py:11797-11798

    // hard_case==2 (interior/near-Newton) resolution -- linear_equation_solve
    // falling back to MINRES on non-convergence, see class doc comment.
    // Values match the Python's own hard-coded constants at this call site
    // exactly (helper_PFCI.py:12058-12073): NOT independently tunable
    // knobs, just exposed here for consistency with this struct's other
    // fields.
    int linear_solve_max_iter = 20;
    double linear_solve_conv_thresh = 1e-6;
    double minres_rtol = 1e-6;
    unsigned int random_seed = 0; // see linear_equation_solve.hpp's doc comment
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
// hard_case==2 (interior/near-Newton) is now a faithful port, not a
// substitution: linear_equation_solve (LinearRMSolver, diagonally
// preconditioned by reduced_hessian_diagonal, matvec via `hessian_->apply`)
// falling back to real MINRES on non-convergence (helper_PFCI.py:
// 12052-12094 -- the same formula, with the same hard-coded constants, as
// CasscfMicroiterationOptimizationStep's own gradient-small-Newton fallback
// at helper_PFCI.py:12103-12137; see linear_equation_solve.hpp's doc
// comment for why these share one generic core rather than being ported
// twice). Previously substituted with PcgTrustRegionSolver at an
// effectively-unconstrained trust radius (mathematically reasonable, since
// hard_case==2 implies a near-PSD Hessian, but not what the Python
// actually does); no longer needed.
//
// One real, NOT fixed limitation this substitution's removal introduces:
// minres_solve (the real MINRES fallback) only accepts a DENSE matrix (see
// its own doc comment), so the MINRES fallback path here materializes the
// full `hessian_->apply` operator via unit-vector probing -- O(dimension)
// extra Hessian-vector products, only paid on the (expected to be rare)
// case where linear_equation_solve itself doesn't converge within
// max_iter. For the genuinely large-dimension problems this class exists
// to handle matrix-free in the first place, that fallback path would be
// expensive if hit often -- acceptable for now (same "flag for future
// performance work" precedent as orbital_sigma3), not a correctness
// concern.
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
