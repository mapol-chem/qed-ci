#pragma once

#include "casscf/hessian_operator.hpp"
#include "casscf/types.hpp"
#include <memory>
#include <vector>

namespace casscf {

// Provides small dense sub-blocks of the true reduced orbital Hessian and
// gradient at an arbitrary set of coordinate indices, e.g.
//     hessian_block(a,b) = H_reduced(indices[a], indices[b])
//     gradient_block(a)  = reduced_gradient(indices[a])
// Port target: build_orbital_hessian_guess -> hessian_guess
// (helper_PFCI.py:16413-16450+), which pulls individual elements out of the
// (A_tilde, G) intermediates via index_map. NOT implemented here -- it
// depends on the intermediates-building tensor contractions, which are out
// of scope for this pass (see cpp_casscf/README.md, "still open").
class HessianGuessProvider {
public:
    virtual ~HessianGuessProvider() = default;
    virtual void guess_block(const std::vector<int>& indices,
                              Matrix& hessian_block,
                              Vector& gradient_block) const = 0;
};

struct DavidsonAugmentedHessianConfig {
    // helper_PFCI.py:15287-15290: dim0 = 200 if index_map_size > 600 else
    // index_map_size // 2.
    int large_problem_threshold = 600;
    int large_problem_dim0 = 200;

    // helper_PFCI.py:15292: dim1 = max(count, dim0), where `count` is the
    // number of (near-)non-positive diagonal entries. Uncapped, this can
    // make the initial guess subspace as large as the whole problem when
    // many directions have non-positive curvature. Per the developer's
    // guidance, cap it at a "reasonable" size instead:
    //     dim1 = min(max(count, dim0), max_guess_dimension)
    // The index selection (by |gradient_i / diagonal_i|, most negative /
    // smallest diagonal first -- see run_iteration_one) already ranks the
    // worst directions first, so truncating to this cap keeps exactly the
    // directions that matter most and drops the long tail.
    int max_guess_dimension = 300;

    double convergence_threshold = 1e-7; // helper_PFCI.py:15280

    // helper_PFCI.py:15308: alpha_range > 1e-5 => only track the lowest
    // root (nroots=1); otherwise track the two lowest (nroots=2).
    double alpha_range_single_root_threshold = 1e-5;

    // helper_PFCI.py:15303: maxdim = min(H_dim, dim1 + min(dim1, 40)) --
    // the "40" is the max number of Krylov correction vectors the subspace
    // is allowed to grow by beyond the initial guess before a soft restart
    // is forced.
    int max_subspace_growth = 40;

    // helper_PFCI.py:15609: num_iter = 10000. Python treats exhausting this
    // as fatal (calls exit()); this port throws instead (see solve()).
    int max_davidson_iterations = 10000;

    // helper_PFCI.py:16028-16031: soft-restart target subspace size once
    // the cap is hit -- min(40, Lmax/2) vectors are kept.
    int soft_restart_target = 40;

    // helper_PFCI.py:14500: inner Jacobi-Davidson correction solve is
    // truncated at 5 CG iterations.
    int inner_pcg_max_iterations = 5;
};

struct DavidsonProblemStructure {
    int n_negative = 0;
    double min_diag = 0.0;
    double max_diag = 0.0;
    double gradient_norm = 0.0;
    int dim1 = 0; // size of the initial guess subspace actually used
};

struct DavidsonIterationResult {
    bool converged = false;
    // Populated when converged: columns are the (up to) two lowest
    // eigenvectors of the bordered matrix in the FULL (index_map_size+1)
    // augmented space, matching the aug_hessian_eigenvecs/aug_hessian_eigenvals
    // out-parameters of Davidson_augmented_hessian_solve6.
    Matrix eigenvectors;
    Vector eigenvalues;
    bool hard_case = false; // true if root 0 alone wasn't usable and root 1 was needed
    DavidsonProblemStructure structure;
    int davidson_iterations = 0; // 0 if converged within the initial guess subspace
};

// Full port of Davidson_augmented_hessian_solve6, helper_PFCI.py:15252-16085.
//
// Used from microiteration_optimization6 for the FULL non-redundant orbital
// rotation space (dimension index_map_size), where the Hessian is only
// available matrix-free. Builds a small initial guess subspace from the
// coordinates with the worst gradient/diagonal ratio (prioritizing
// non-positive-curvature directions), solves the bordered eigenproblem on
// that subspace, and checks the true (matrix-free) residual to decide
// whether the lowest root is usable (easy case) or the solver needs to fall
// back to the second-lowest root (hard case) -- the root-switching heuristic
// described by the developer, formalized in check_root_usable() below
// (port of check_root_0, helper_PFCI.py:14546-14561). If that initial guess
// isn't enough, expands the Davidson subspace with preconditioned
// correction vectors (plain diagonal preconditioning while no root has
// converged yet; a deflated Jacobi-Davidson PCG correction, inner_solve_pcg,
// once at least one has -- this two-phase "hybrid" strategy shortens the
// iteration count at the cost of more expensive per-iteration correction
// solves, per the developer), with a soft restart (collapsing to the best
// `soft_restart_target` Ritz vectors) when the subspace hits its cap.
//
// STATEFUL BY DESIGN: like the Python (which threads state through
// self.sigma_total / self.H_pp / self.H_qp / self.collapse_subspace_check /
// self.idx_hessian / self.indim across repeated calls with restart=True),
// one DavidsonAugmentedHessianSolver instance is meant to be reused across
// an entire outer beta-bisection sequence (see DavidsonDrivenLstrsSolver,
// which constructs exactly one and calls solve() once per bisection step).
// Constructing a fresh instance and calling solve() with restart=true would
// be a bug, not a valid "no-op restart" -- restart=true always resumes
// from this instance's own persisted subspace.
class DavidsonAugmentedHessianSolver {
public:
    DavidsonAugmentedHessianSolver(std::shared_ptr<const HessianOperator> hessian,
                                    std::shared_ptr<const HessianGuessProvider> guess_provider,
                                    DavidsonAugmentedHessianConfig config = {});

    // restart=false: build a fresh initial guess subspace (helper_PFCI.py:15322-15530).
    // restart=true: resume from this instance's persisted subspace at the
    // new (alpha, alpha_range) (helper_PFCI.py:15551-15607). Mirrors the
    // Python signature closely except `guess_vector` isn't a parameter --
    // it's exactly this instance's persisted Q, so there's nothing for the
    // caller to thread through.
    DavidsonIterationResult solve(const Vector& reduced_gradient,
                                   const Vector& reduced_hessian_diagonal,
                                   double alpha,
                                   double alpha_range,
                                   double trust_radius,
                                   bool restart);

private:
    // The Davidson subspace expansion loop (helper_PFCI.py:15608-16084),
    // entered once the setup phase (fresh guess subspace when !restart, or
    // resumed persisted subspace when restart) has left at least one root
    // unconverged. `nroots`/`roots_to_check`/`root_0_locked` come from
    // whichever setup phase ran; `L_old` is the subspace size at the point
    // this loop starts growing it further.
    DavidsonIterationResult run_expansion_loop(const Vector& reduced_gradient,
                                                const Vector& H_diag_augmented,
                                                double alpha,
                                                double trust_radius,
                                                int H_dim,
                                                int Lmax,
                                                int nroots,
                                                std::vector<int> roots_to_check,
                                                bool root_0_locked,
                                                int L_old,
                                                bool restart);

    std::shared_ptr<const HessianOperator> hessian_;
    std::shared_ptr<const HessianGuessProvider> guess_provider_;
    DavidsonAugmentedHessianConfig config_;

    // Persistent subspace state (helper_PFCI.py's self.* attributes).
    Matrix Q_;                            // self.Q, via `guess_vector` -- current trial-vector subspace
    Matrix sigma_total_;                  // self.sigma_total
    Matrix H_pp_;                         // self.H_pp
    Matrix H_qp_;                         // self.H_qp
    Matrix projected_augmented_hessian_;  // self.projected_augmented_hessian
    bool collapse_subspace_check_ = false;
    std::vector<int> idx_hessian_;        // self.idx_hessian
    int indim_ = 0;                       // self.indim
};

// Port of check_root_0, helper_PFCI.py:14546-14561. `augmented_eigvec` is
// one column of the bordered-matrix eigendecomposition in the full
// (index_map_size+1)-dimensional augmented space (border component at
// index 0). Returns true if this root gives a usable (normalizable, within
// the classical-hard-case-free regime) step -- the same
// lowest-root-vs-second-root switch the quantum-chemistry AH/RFO literature
// does implicitly by falling back to the next root.
bool check_root_usable(const Vector& augmented_eigvec, double eigenvalue,
                        const Vector& gradient, double trust_radius);

} // namespace casscf
