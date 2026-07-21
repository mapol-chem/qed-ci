#include "casscf/davidson_augmented_hessian_solver.hpp"
#include "casscf/bordered_eigensolve.hpp"
#include "casscf/gram_schmidt.hpp"
#include "casscf/jacobi_davidson_correction.hpp"

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace casscf {

namespace {

Matrix vstack(const Matrix& a, const Matrix& b) {
    Matrix out(a.rows() + b.rows(), a.cols());
    out.topRows(a.rows()) = a;
    out.bottomRows(b.rows()) = b;
    return out;
}

} // namespace

bool check_root_usable(const Vector& augmented_eigvec, double eigenvalue,
                        const Vector& gradient, double trust_radius) {
    constexpr double epsilon_v = 1e-4;
    const int n = static_cast<int>(gradient.size());
    const double norm_w = augmented_eigvec.norm();
    const double v1 = augmented_eigvec(0) / norm_w;
    const Vector u1 = augmented_eigvec.tail(n) / norm_w;
    const double aa1 = gradient.norm() * std::abs(v1);
    const double bb1 = std::sqrt(1.0 - v1 * v1);

    if ((eigenvalue > -1e-8 && u1.norm() < trust_radius * std::abs(v1)) || aa1 > epsilon_v * bb1) {
        return true; // "use first root" (easy case)
    }
    return false; // "use second root" (hard case)
}

DavidsonAugmentedHessianSolver::DavidsonAugmentedHessianSolver(
    std::shared_ptr<const HessianOperator> hessian,
    std::shared_ptr<const HessianGuessProvider> guess_provider,
    DavidsonAugmentedHessianConfig config)
    : hessian_(std::move(hessian)), guess_provider_(std::move(guess_provider)), config_(config) {}

DavidsonIterationResult DavidsonAugmentedHessianSolver::solve(
    const Vector& reduced_gradient,
    const Vector& reduced_hessian_diagonal,
    double alpha,
    double alpha_range,
    double trust_radius,
    bool restart) {

    const int index_map_size = static_cast<int>(reduced_gradient.size());
    const int H_dim = index_map_size + 1;

    DavidsonIterationResult result;
    DavidsonProblemStructure& structure = result.structure;
    structure.n_negative = static_cast<int>((reduced_hessian_diagonal.array() < 0.0).count());
    structure.min_diag = reduced_hessian_diagonal.minCoeff();
    structure.max_diag = reduced_hessian_diagonal.maxCoeff();
    structure.gradient_norm = reduced_gradient.norm();

    // helper_PFCI.py:15284-15292, with the developer-requested cap on dim1.
    int count = 0;
    for (int i = 0; i < index_map_size; ++i) {
        if (reduced_hessian_diagonal(i) <= 1e-14) ++count;
    }
    const int dim0 = (index_map_size > config_.large_problem_threshold)
                          ? config_.large_problem_dim0
                          : index_map_size / 2;
    int dim1 = std::max(count, dim0);
    dim1 = std::min(dim1, config_.max_guess_dimension); // developer-requested fix
    dim1 = std::min(dim1, index_map_size);
    structure.dim1 = dim1;

    const int indim_local = dim1 + 1;
    const int maxdim = std::min(H_dim, dim1 + std::min(dim1, config_.max_subspace_growth));

    Vector H_diag_augmented(H_dim);
    H_diag_augmented(0) = alpha;
    H_diag_augmented.tail(index_map_size) = reduced_hessian_diagonal;

    const bool single_root = alpha_range > config_.alpha_range_single_root_threshold;
    const int nroots = single_root ? 1 : 2;
    bool root_0_locked = false;
    std::vector<int> unconverged_idx;
    int L_old = 0;

    BorderedHessianOperator augmented_op(hessian_, reduced_gradient, alpha);

    if (!restart) {
        // --- helper_PFCI.py:15322-15530: build a fresh initial guess subspace ---
        std::vector<std::pair<double, int>> ranked(index_map_size);
        for (int i = 0; i < index_map_size; ++i) {
            const double diag = reduced_hessian_diagonal(i);
            const double d = (diag > 1e-5) ? std::abs(reduced_gradient(i)) / diag : 1e14;
            ranked[i] = {d, i};
        }
        std::stable_sort(ranked.begin(), ranked.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });
        idx_hessian_.assign(dim1, 0);
        for (int i = 0; i < dim1; ++i) idx_hessian_[i] = ranked[i].second;
        indim_ = indim_local;

        Matrix guess_hessian;
        Vector guess_gradient;
        guess_provider_->guess_block(idx_hessian_, guess_hessian, guess_gradient);

        H_pp_ = Matrix::Zero(indim_local, indim_local);
        H_pp_(0, 0) = alpha;
        H_pp_.block(0, 1, 1, dim1) = guess_gradient.transpose();
        H_pp_.block(1, 0, dim1, 1) = guess_gradient;
        H_pp_.block(1, 1, dim1, dim1) = guess_hessian;
        H_qp_ = Matrix(0, indim_local);
        sigma_total_ = Matrix::Zero(maxdim, H_dim);
        collapse_subspace_check_ = false;

        BorderedEigenPairs guess_eig = bordered_eigensolve(guess_gradient, guess_hessian, alpha);
        const Vector& theta = guess_eig.eigenvalues;

        Matrix Q_eigvec_based = Matrix::Zero(indim_local, H_dim);
        for (int i = 0; i < indim_local; ++i) {
            Q_eigvec_based(i, 0) = guess_eig.eigenvectors(0, i);
            for (int j = 0; j < dim1; ++j) {
                Q_eigvec_based(i, idx_hessian_[j] + 1) = guess_eig.eigenvectors(j + 1, i);
            }
        }

        std::vector<int> roots_to_check = single_root ? std::vector<int>{0} : std::vector<int>{0, 1};
        Vector residual_norm = Vector::Constant(2, std::numeric_limits<double>::infinity());
        std::vector<bool> conv_status(2, false);
        Matrix w = Matrix::Zero(2, H_dim); // residual vectors, indexed by root
        auto check_residual = [&](int root_idx) {
            const Vector vec = Q_eigvec_based.row(root_idx).transpose();
            const Vector r_vec = augmented_op.apply(vec) - theta(root_idx) * vec;
            w.row(root_idx) = r_vec;
            residual_norm(root_idx) = r_vec.norm();
            conv_status[root_idx] = residual_norm(root_idx) < config_.convergence_threshold;
        };
        for (int r : roots_to_check) check_residual(r);

#ifdef CASSCF_DEBUG_DAVIDSON
        std::fprintf(stderr, "initial guess check: nroots=%d resid0=%.3e resid1=%.3e conv0=%d conv1=%d theta0=%.6f theta1=%.6f\n",
                     nroots, residual_norm(0), residual_norm(1), (int)conv_status[0], (int)conv_status[1],
                     theta.size() > 0 ? theta(0) : 0.0, theta.size() > 1 ? theta(1) : 0.0);
#endif

        if (nroots == 1) {
            if (conv_status[0]) {
                if (check_root_usable(Q_eigvec_based.row(0).transpose(), theta(0), reduced_gradient, trust_radius)) {
                    result.converged = true;
                    result.hard_case = false;
                    result.eigenvalues = Vector::Constant(1, theta(0));
                    result.eigenvectors = Q_eigvec_based.row(0).transpose();
                } else {
                    root_0_locked = true;
                    check_residual(1);
                    if (conv_status[1]) {
                        result.converged = true;
                        result.hard_case = true;
                        result.eigenvalues = theta.head(2);
                        result.eigenvectors = Q_eigvec_based.topRows(2).transpose();
                    } else {
                        unconverged_idx = {1};
                    }
                }
            } else {
                unconverged_idx = {0};
            }
        } else {
            if (conv_status[0] && conv_status[1]) {
                result.converged = true;
                result.hard_case = false;
                result.eigenvalues = theta.head(2);
                result.eigenvectors = Q_eigvec_based.topRows(2).transpose();
            } else {
                if (!conv_status[0]) unconverged_idx.push_back(0);
                if (!conv_status[1]) unconverged_idx.push_back(1);
            }
        }

        // helper_PFCI.py:15509-15530: regardless of whether iteration 1
        // already converged, execution falls through unconditionally into
        // rebuilding Q as the plain unit-vector basis for the guess block
        // (not the eigenvector combinations used just above for the
        // residual check) -- this (and the alpha-zeroed H_pp-based
        // projected Hessian) is what a future restart=True call resumes
        // from, whether or not this call already found a converged root.
        Matrix Q_unit_based = Matrix::Zero(indim_local, H_dim);
        Q_unit_based(0, 0) = 1.0;
        for (int i = 0; i < dim1; ++i) Q_unit_based(i + 1, idx_hessian_[i] + 1) = 1.0;

        projected_augmented_hessian_ = Matrix::Zero(indim_local, indim_local);
        projected_augmented_hessian_(0, 0) = 0.0;
        projected_augmented_hessian_.block(0, 1, 1, dim1) = guess_gradient.transpose();
        projected_augmented_hessian_.block(1, 0, dim1, 1) = guess_gradient;
        projected_augmented_hessian_.block(1, 1, dim1, dim1) = guess_hessian;

        // helper_PFCI.py:15532-15547: diagonally precondition the
        // unconverged residuals (empty if already converged -- a no-op
        // append below) and append as the first correction vectors.
        Matrix preconditioned_w(unconverged_idx.size(), H_dim);
        for (size_t i = 0; i < unconverged_idx.size(); ++i) {
            const int root_idx = unconverged_idx[i];
            const Vector precon_denom = (Vector::Constant(H_dim, theta(root_idx)) - H_diag_augmented);
            Vector row(H_dim);
            for (int c = 0; c < H_dim; ++c) {
                row(c) = (precon_denom(c) != 0.0) ? (w(root_idx, c) / precon_denom(c)) : 0.0;
            }
            preconditioned_w.row(i) = row;
        }

        Q_ = vstack(Q_unit_based, preconditioned_w);
        if (!unconverged_idx.empty()) {
            gram_schmidt_add(Q_, indim_local, static_cast<int>(unconverged_idx.size()));
        }
        L_old = indim_local;

        if (result.converged) return result;
    } else {
        // --- helper_PFCI.py:15551-15607: resume from the persisted subspace ---
        const int L = static_cast<int>(Q_.rows());
        const int indim = indim_; // helper_PFCI.py:15559 -- overrides the locally recomputed value

        if (!collapse_subspace_check_) {
            if (L == indim) {
                projected_augmented_hessian_(0, 0) = alpha;
            } else {
                const int b_dim = L - indim;
                projected_augmented_hessian_(0, 0) = alpha;
                Matrix Sq = Matrix::Zero(b_dim, H_dim);
                Sq.col(0) = alpha * Q_.block(indim, 0, b_dim, 1);
                sigma_total_.block(indim, 0, b_dim, 1) += Sq.col(0);
                if (indim > 0) {
                    for (int i = 0; i < b_dim; ++i) {
                        projected_augmented_hessian_(indim + i, 0) += Sq(i, 0);
                        projected_augmented_hessian_(0, indim + i) += Sq(i, 0);
                        H_qp_(i, 0) += Sq(i, 0);
                    }
                    H_pp_(0, 0) = alpha;
                    const Matrix H_qq = Sq * Q_.block(indim, 0, b_dim, H_dim).transpose();
                    projected_augmented_hessian_.block(indim, indim, b_dim, b_dim) += H_qq;
                } else {
                    projected_augmented_hessian_.block(indim, indim, L - indim, L - indim) =
                        sigma_total_.topRows(L) * Q_.topRows(L).transpose();
                }
            }
        } else {
            const int L_new = L;
            Matrix Sq = Matrix::Zero(L_new, H_dim);
            Sq.col(0) = alpha * Q_.col(0);
            sigma_total_.block(0, 0, L_new, 1) += Sq.col(0);
            projected_augmented_hessian_ = sigma_total_.topRows(L_new) * Q_.topRows(L_new).transpose();

            if (nroots == 1 && !root_0_locked) unconverged_idx = {0};
            else if (nroots == 1 && root_0_locked) unconverged_idx = {1};
            else if (nroots == 2) unconverged_idx = {0, 1};
        }
        L_old = indim_;
    }

    std::vector<int> roots_to_check = single_root ? std::vector<int>{0} : std::vector<int>{0, 1};
    DavidsonIterationResult expansion_result = run_expansion_loop(
        reduced_gradient, H_diag_augmented, alpha, trust_radius, H_dim, maxdim, nroots, roots_to_check,
        root_0_locked, L_old, restart);
    // run_expansion_loop() default-constructs its own DavidsonIterationResult
    // (it has no way to see `structure`, computed only up here in solve()) --
    // carry it over so callers see the real problem-structure diagnostics
    // instead of a silently all-zero default whenever the expansion loop
    // actually ran (the common case: `structure` was previously only
    // populated on the "converged already in the initial guess subspace"
    // fast-path return above). Not consumed by any production caller today
    // (DavidsonDrivenLstrsSolver ignores `structure` entirely), but this was
    // a real data-loss bug in the port, caught while adding test coverage
    // for the restart=true/no-collapse expansion-loop path.
    expansion_result.structure = structure;
    return expansion_result;
}

DavidsonIterationResult DavidsonAugmentedHessianSolver::run_expansion_loop(
    const Vector& reduced_gradient,
    const Vector& H_diag_augmented,
    double alpha,
    double trust_radius,
    int H_dim,
    int Lmax,
    int nroots,
    std::vector<int> roots_to_check,
    bool root_0_locked,
    int L_old,
    bool restart) {

    constexpr int nroots_target = 2;
    constexpr double threshold = 1e-7; // helper_PFCI.py:15280
    const int dim1 = static_cast<int>(idx_hessian_.size());
    BorderedHessianOperator augmented_op(hessian_, reduced_gradient, alpha);

    bool collapse = true; // helper_PFCI.py:15612
    DavidsonIterationResult result;

    for (int davidson_iteration = 1; davidson_iteration < config_.max_davidson_iterations; ++davidson_iteration) {
        result.davidson_iterations = davidson_iteration;
        const int L_new = static_cast<int>(Q_.rows());
        const int L = L_new;

        Matrix projected_augmented_hessian;
        // helper_PFCI.py:15630: only the very first iteration of a
        // restart=True call reuses the persisted projected Hessian as-is;
        // every other iteration (including iteration 1 when restart==False)
        // rebuilds it incrementally from the current Q_.
        const bool use_persisted_directly = restart && (davidson_iteration == 1);
        if (use_persisted_directly) {
            projected_augmented_hessian = projected_augmented_hessian_;
        } else if (!collapse_subspace_check_) {
            const int c_dim = L_new - L_old;
            Matrix Sq1 = Matrix::Zero(c_dim, H_dim);
            for (int i = 0; i < c_dim; ++i) {
                // orbital_sigma3 only ever fills [:,1:] with the PLAIN H*v[1:]
                // (helper_PFCI.py:15643) -- the border/gradient corrections
                // are added separately below (matches aug_matvec's own split,
                // helper_PFCI.py:14576-14580). Using the bordered operator
                // here instead would double-count the g*v0 term.
                Sq1.row(i).tail(H_dim - 1) =
                    hessian_->apply(Q_.row(L_old + i).tail(H_dim - 1).transpose()).transpose();
            }
            Matrix Sq = Sq1;
            for (int i = 0; i < c_dim; ++i) {
                Sq.row(i).tail(H_dim - 1) += reduced_gradient.transpose() * Q_(L_old + i, 0);
                Sq(i, 0) = alpha * Q_(L_old + i, 0) + Q_.row(L_old + i).tail(H_dim - 1).dot(reduced_gradient);
            }
            sigma_total_.block(L_old, 0, c_dim, H_dim) = Sq;

            if (indim_ > 0) {
                Matrix H_qp_new = Matrix::Zero(c_dim, indim_);
                for (int i = 0; i < c_dim; ++i) {
                    for (int j = 0; j < dim1; ++j) {
                        const int index1 = idx_hessian_[j];
                        H_qp_new(i, 0) = Sq(i, 0);
                        H_qp_new(i, j + 1) = Sq(i, index1 + 1);
                    }
                }
                H_qp_ = vstack(H_qp_, H_qp_new);
                const Matrix H_qq = sigma_total_.block(indim_, 0, L_new - indim_, H_dim) *
                                     Q_.block(indim_, 0, L_new - indim_, H_dim).transpose();
                Matrix H1 = vstack(H_pp_, H_qp_);
                Matrix H2 = vstack(H_qp_.transpose(), H_qq);
                Matrix combined(H1.rows(), H1.cols() + H2.cols());
                combined << H1, H2;
                projected_augmented_hessian = combined;
            } else {
                const Matrix H_qq = sigma_total_.topRows(L_new) * Q_.topRows(L_new).transpose();
                projected_augmented_hessian = H_qq;
            }
        } else {
            if (!collapse) {
                const int c_dim = L_new - L_old;
                Matrix Sq1 = Matrix::Zero(c_dim, H_dim);
                for (int i = 0; i < c_dim; ++i) {
                    Sq1.row(i).tail(H_dim - 1) =
                        hessian_->apply(Q_.row(L_old + i).tail(H_dim - 1).transpose()).transpose();
                }
                Matrix Sq = Sq1;
                for (int i = 0; i < c_dim; ++i) {
                    Sq.row(i).tail(H_dim - 1) += reduced_gradient.transpose() * Q_(L_old + i, 0);
                    Sq(i, 0) = alpha * Q_(L_old + i, 0) + Q_.row(L_old + i).tail(H_dim - 1).dot(reduced_gradient);
                }
                sigma_total_.block(L_old, 0, c_dim, H_dim) = Sq;
                projected_augmented_hessian = sigma_total_.topRows(L_new) * Q_.topRows(L_new).transpose();
                L_old = L_new;
            } else {
                const int c_dim = L_new;
                sigma_total_.setZero();
                Matrix Sq1 = Matrix::Zero(c_dim, H_dim);
                for (int i = 0; i < c_dim; ++i) {
                    Sq1.row(i).tail(H_dim - 1) = hessian_->apply(Q_.row(i).tail(H_dim - 1).transpose()).transpose();
                }
                Matrix Sq = Sq1;
                for (int i = 0; i < c_dim; ++i) {
                    Sq.row(i).tail(H_dim - 1) += reduced_gradient.transpose() * Q_(i, 0);
                    Sq(i, 0) = alpha * Q_(i, 0) + Q_.row(i).tail(H_dim - 1).dot(reduced_gradient);
                }
                sigma_total_.topRows(c_dim) = Sq;
                projected_augmented_hessian = sigma_total_.topRows(L_new) * Q_.topRows(L_new).transpose();
            }
        }

        Eigen::SelfAdjointEigenSolver<Matrix> solver(projected_augmented_hessian);
        const Vector theta = solver.eigenvalues();
        const Matrix aug_eigvecs = solver.eigenvectors();
        const Matrix full_eigvecs = aug_eigvecs.transpose().topRows(nroots_target) * Q_;
        const Matrix full_eigvecs2 = full_eigvecs;

        roots_to_check.clear();
        if (nroots == 1 && !root_0_locked) roots_to_check = {0};
        else if (nroots == 1 && root_0_locked) roots_to_check = {1};
        else roots_to_check = {0, 1};

        Matrix w = Matrix::Zero(nroots_target, H_dim);
        Vector residual_norm = Vector::Constant(nroots_target, std::numeric_limits<double>::infinity());
        std::vector<bool> conv_status(nroots_target, false);

        if (!roots_to_check.empty()) {
            Matrix w_sigma(roots_to_check.size(), H_dim);
            if (collapse_subspace_check_) {
                for (size_t i = 0; i < roots_to_check.size(); ++i) {
                    w_sigma.row(i) = aug_eigvecs.col(roots_to_check[i]).transpose() * sigma_total_.topRows(L_new);
                }
            } else {
                for (size_t i = 0; i < roots_to_check.size(); ++i) {
                    w_sigma.row(i) = augmented_op.apply(full_eigvecs.row(roots_to_check[i]).transpose()).transpose();
                }
            }
            for (size_t i = 0; i < roots_to_check.size(); ++i) {
                const int root_idx = roots_to_check[i];
                const Vector vec = full_eigvecs.row(root_idx).transpose();
                const Vector r_vec = w_sigma.row(i).transpose() - theta(root_idx) * vec;
                w.row(root_idx) = r_vec;
                residual_norm(root_idx) = r_vec.norm();
                if (residual_norm(root_idx) < threshold) conv_status[root_idx] = true;
            }
        }
        if (root_0_locked) conv_status[0] = true;

#ifdef CASSCF_DEBUG_DAVIDSON
        std::fprintf(stderr, "iter=%d L=%d L_old=%d collapse_check=%d collapse=%d theta0=%.6f theta1=%.6f resid0=%.3e resid1=%.3e conv0=%d conv1=%d root_0_locked=%d\n",
                     davidson_iteration, L, L_old, collapse_subspace_check_, collapse,
                     theta.size() > 0 ? theta(0) : 0.0, theta.size() > 1 ? theta(1) : 0.0,
                     residual_norm(0), residual_norm(1), (int)conv_status[0], (int)conv_status[1], root_0_locked);
#endif

        std::vector<int> new_unconverged;
        bool exit_solver = false;
        if (nroots == 1) {
            if (conv_status[0] && !root_0_locked) {
                if (check_root_usable(full_eigvecs.row(0).transpose(), theta(0), reduced_gradient, trust_radius)) {
                    result.converged = true;
                    result.hard_case = false;
                    // helper_PFCI.py:16138-16140: Python unconditionally
                    // overwrites aug_hessian_eigenvals[:]/[:,:] with
                    // theta[:nroots_target]/full_eigvecs.T on every
                    // exit_solver=True exit (nroots_target is always 2, see
                    // line 15563), regardless of which branch triggered the
                    // exit -- so root 1's already-computed theta(1)/
                    // full_eigvecs.row(1) is what a real Easy Case exit
                    // actually returns, not a placeholder. Keep both
                    // entries here to match (see README.md's "Davidson
                    // subspace expansion loop" section, bug #3).
                    result.eigenvalues = theta.head(2);
                    result.eigenvectors = full_eigvecs.topRows(2).transpose();
                    exit_solver = true;
                } else {
                    root_0_locked = true;
                    Vector w1_sigma;
                    if (collapse_subspace_check_) {
                        w1_sigma = (aug_eigvecs.col(1).transpose() * sigma_total_.topRows(L_new)).transpose();
                    } else {
                        w1_sigma = augmented_op.apply(full_eigvecs.row(1).transpose());
                    }
                    const Vector r1 = w1_sigma - theta(1) * full_eigvecs.row(1).transpose();
                    w.row(1) = r1;
                    residual_norm(1) = r1.norm();
                    if (residual_norm(1) < threshold) {
                        conv_status[1] = true;
                        result.converged = true;
                        result.hard_case = true;
                        result.eigenvalues = theta.head(2);
                        result.eigenvectors = full_eigvecs.topRows(2).transpose();
                        exit_solver = true;
                    } else {
                        new_unconverged = {1};
                    }
                }
            } else if (!conv_status[0]) {
                new_unconverged = {0};
            } else if (root_0_locked && !conv_status[1]) {
                new_unconverged = {1};
            }
        } else {
            if (!conv_status[0]) new_unconverged.push_back(0);
            if (!conv_status[1]) new_unconverged.push_back(1);
        }

        const bool all_required_converged = (nroots == 2 && new_unconverged.empty()) ||
                                             (nroots == 1 && root_0_locked && new_unconverged.empty());
        if (all_required_converged) {
            exit_solver = true;
            result.converged = true;
            result.hard_case = (nroots == 1);
            result.eigenvalues = theta.head(2);
            result.eigenvectors = full_eigvecs.topRows(2).transpose();
        }

        if (exit_solver) {
            // helper_PFCI.py:15882-15921: persist state, with this call's
            // alpha contribution subtracted back out, for a future
            // restart=True call at a different alpha.
            if (!collapse_subspace_check_) {
                const int b_dim = L - indim_;
                projected_augmented_hessian_ = projected_augmented_hessian;
                projected_augmented_hessian_(0, 0) = alpha;
                if (b_dim > 0) {
                    Matrix Sq = Matrix::Zero(b_dim, H_dim);
                    Sq.col(0) = alpha * Q_.block(indim_, 0, b_dim, 1);
                    sigma_total_.block(indim_, 0, b_dim, 1) -= Sq.col(0);
                    if (indim_ > 0) {
                        for (int i = 0; i < b_dim; ++i) {
                            projected_augmented_hessian_(indim_ + i, 0) -= Sq(i, 0);
                            projected_augmented_hessian_(0, indim_ + i) -= Sq(i, 0);
                            H_qp_(i, 0) -= Sq(i, 0);
                        }
                        const Matrix H_qq = Sq * Q_.block(indim_, 0, b_dim, H_dim).transpose();
                        projected_augmented_hessian_.block(indim_, indim_, b_dim, b_dim) -= H_qq;
                    } else {
                        projected_augmented_hessian_.block(indim_, indim_, b_dim, b_dim).setZero();
                    }
                }
            } else {
                Matrix Sq = Matrix::Zero(L_new, H_dim);
                Sq.col(0) = alpha * Q_.col(0);
                sigma_total_.block(0, 0, L_new, 1) -= Sq.col(0);
            }
            return result;
        }

        // --- Hybrid preconditioning (helper_PFCI.py:15925-15972) ---
        std::vector<int> converged_indices;
        for (int i = 0; i < nroots; ++i) {
            if (std::find(new_unconverged.begin(), new_unconverged.end(), i) == new_unconverged.end())
                converged_indices.push_back(i);
        }

        Matrix preconditioned_w(new_unconverged.size(), H_dim);
        if (converged_indices.empty()) {
            for (size_t i = 0; i < new_unconverged.size(); ++i) {
                const int idx = new_unconverged[i];
                Vector row(H_dim);
                for (int c = 0; c < H_dim; ++c) {
                    const double denom = theta(idx) - H_diag_augmented(c);
                    row(c) = (denom != 0.0) ? (w(idx, c) / denom) : 0.0;
                }
                preconditioned_w.row(i) = row;
            }
        } else {
            Matrix Q_conv(converged_indices.size(), H_dim);
            for (size_t i = 0; i < converged_indices.size(); ++i) Q_conv.row(i) = full_eigvecs2.row(converged_indices[i]);
            for (size_t i = 0; i < new_unconverged.size(); ++i) {
                const int idx = new_unconverged[i];
                const Vector current_residual = w.row(idx).transpose();
                const double current_theta = theta(idx);
                Vector P_diag = H_diag_augmented.array() - current_theta;
                for (int c = 0; c < H_dim; ++c) {
                    if (std::abs(P_diag(c)) < 1e-12) P_diag(c) = 1.0;
                }
                preconditioned_w.row(i) = inner_solve_pcg(hessian_, reduced_gradient, alpha, current_residual,
                                                           current_theta, Q_conv, P_diag,
                                                           config_.inner_pcg_max_iterations);
            }
        }

        L_old = static_cast<int>(Q_.rows());

        if (Lmax - L < static_cast<int>(new_unconverged.size())) {
            const int num_restart_vecs = (Lmax > config_.soft_restart_target)
                                              ? config_.soft_restart_target
                                              : Lmax / 2;
            const Matrix best_projected_vecs = aug_eigvecs.leftCols(std::min(num_restart_vecs, static_cast<int>(aug_eigvecs.cols())));
            const Matrix Q_restarted = best_projected_vecs.transpose() * Q_;
            Q_ = vstack(Q_restarted, preconditioned_w);
            gram_schmidt_orthogonalize(Q_);
            collapse_subspace_check_ = true;
            collapse = true;
            L_old = 0;
        } else {
            Q_ = vstack(Q_, preconditioned_w);
            gram_schmidt_add(Q_, L, static_cast<int>(preconditioned_w.rows()));
            collapse = false;
        }
    }

    throw std::runtime_error(
        "DavidsonAugmentedHessianSolver: maximum Davidson iterations reached without a converged root "
        "(helper_PFCI.py:15614-15618 treats this as fatal too)");
}

} // namespace casscf
