#include "casscf/gltr_trust_region_solver.hpp"

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <random>

namespace casscf {

GltrTrustRegionSolver::GltrTrustRegionSolver(std::shared_ptr<const HessianOperator> hessian,
                                              Vector diag_preconditioner,
                                              GltrConfig config)
    : hessian_(std::move(hessian)), diag_preconditioner_(std::move(diag_preconditioner)), config_(config) {}

TrustRegionResult GltrTrustRegionSolver::solve(const Vector& gradient, double trust_radius) const {
    const int n = static_cast<int>(gradient.size());
    // helper_PFCI.py:14789-14790: max_iter capped at the problem dimension.
    const int max_iter = std::max(1, std::min(config_.max_iter, n));

    const Vector M_abs = (diag_preconditioner_.cwiseAbs().array() + 1e-12).matrix();

    TrustRegionResult result;

    // helper_PFCI.py:14799-14802: perturb the Lanczos start vector so exact
    // orthogonality to a hidden negative-curvature direction doesn't hide it
    // from the process. See GltrConfig doc comment re: seeded vs. Python's
    // unseeded global RNG.
    Vector r = gradient;
    if (config_.add_noise) {
        std::mt19937 rng(config_.random_seed);
        std::normal_distribution<double> dist(0.0, 1.0);
        Vector noise(n);
        for (int i = 0; i < n; ++i) noise(i) = dist(rng);
        r = gradient + (config_.noise_relative_scale * gradient.norm()) * noise;
    }

    const Vector z0 = r.cwiseQuotient(M_abs);
    const double inner_prod0 = r.dot(z0);
    if (inner_prod0 <= 1e-20) {
        result.step = Vector::Zero(n);
        result.reason = TerminationReason::SuccessGradientZero;
        return result;
    }
    const double beta_0 = std::sqrt(inner_prod0);

    Matrix Q = Matrix::Zero(max_iter + 1, n);
    Vector alpha = Vector::Zero(max_iter);
    Vector beta = Vector::Zero(max_iter);
    Q.row(0) = (z0 / beta_0).transpose();

    Vector p_current = Vector::Zero(n);

    for (int k = 0; k < max_iter; ++k) {
        result.iterations = k + 1;
        const Vector v_curr = Q.row(k).transpose();
        const Vector H_v = hessian_->apply(v_curr);
        const double al = v_curr.dot(H_v);
        alpha(k) = al;

        const Vector w = H_v.cwiseQuotient(M_abs);
        Vector r_next = (k == 0) ? Vector(w - al * v_curr)
                                  : Vector(w - al * v_curr - beta(k - 1) * Q.row(k - 1).transpose());

        for (int j = 0; j <= k; ++j) {
            const Vector Qj = Q.row(j).transpose();
            const double overlap = r_next.dot(Qj.cwiseProduct(M_abs));
            r_next -= overlap * Qj;
        }

        const double inner_prod = r_next.dot(r_next.cwiseProduct(M_abs));
        const double be = std::sqrt(std::max(0.0, inner_prod));
        beta(k) = be;

        const bool solve_final = be < 1e-12;
        if (!solve_final && k < max_iter - 1) {
            Q.row(k + 1) = (r_next / be).transpose();
        }
        const int curr_beta_len = k; // beta[:k] in both branches, helper_PFCI.py:14845/14849

        const int dim = k + 1;
        Matrix T = Matrix::Zero(dim, dim);
        for (int i = 0; i < dim; ++i) T(i, i) = alpha(i);
        for (int i = 0; i < curr_beta_len; ++i) {
            T(i, i + 1) = beta(i);
            T(i + 1, i) = beta(i);
        }
        Eigen::SelfAdjointEigenSolver<Matrix> tridiag(T);
        const Vector w_eig = tridiag.eigenvalues();  // ascending
        const Matrix v_eig = tridiag.eigenvectors();

        const double min_eig = w_eig.minCoeff();
        const Vector g_inner = beta_0 * v_eig.row(0).transpose();

        double step_norm_val = 0.0;

        // --- CHECK A: Interior Newton step ---
        if (min_eig > 1e-12) {
            const Vector z_newton = -g_inner.cwiseQuotient(w_eig);
            const Vector y_T_newton = v_eig * z_newton;
            const Vector p_newton = Q.topRows(dim).transpose() * y_T_newton;
            const double norm_newton = p_newton.norm();
            const double resid = be * std::abs(y_T_newton(dim - 1));

            p_current = p_newton;
            step_norm_val = norm_newton;

            if (norm_newton <= trust_radius && (resid < config_.tol || solve_final)) {
                result.step = p_newton;
                result.reason = TerminationReason::SuccessInteriorSolution;
                result.hard_case = false;
                result.predicted_decrease = gradient.dot(result.step) + 0.5 * result.step.dot(hessian_->apply(result.step));
                return result;
            }
        }

        // --- CHECK B: Boundary / hard case ---
        const bool should_solve_boundary = solve_final || (k == max_iter - 1) || (min_eig < 0.0) ||
                                            (min_eig > 0.0 && step_norm_val > trust_radius);

        if (should_solve_boundary) {
            auto get_norm_error = [&](double lam) {
                const Vector z_local = -g_inner.cwiseQuotient((w_eig.array() + lam).matrix());
                const Vector coeffs = v_eig * z_local;
                const Vector p_temp = Q.topRows(dim).transpose() * coeffs;
                return p_temp.norm() - trust_radius;
            };

            const double lam_min = std::max(0.0, -min_eig) + 1e-6;
            const double err_low = get_norm_error(lam_min);
            const bool hard_case_branch = err_low < 0.0;
            double lam_final;

            if (hard_case_branch) {
                lam_final = lam_min;
            } else {
                double low = lam_min;
                double high = lam_min + 1.0;
                bool found_bracket = false;
                for (int i = 0; i < 15; ++i) {
                    if (get_norm_error(high) < 0.0) { found_bracket = true; break; }
                    high = std::max(high * 2.0, high + 10.0);
                }
                if (!found_bracket) {
                    lam_final = high;
                } else {
                    for (int i = 0; i < 30; ++i) {
                        const double mid = (low + high) / 2.0;
                        const double err = get_norm_error(mid);
                        if (std::abs(err) < 1e-4 * trust_radius) { low = mid; break; }
                        if (err > 0.0) low = mid; else high = mid;
                    }
                    lam_final = low;
                }
            }

            const Vector z_boundary = -g_inner.cwiseQuotient((w_eig.array() + lam_final).matrix());
            const Vector y_T_boundary = v_eig * z_boundary;
            Vector p_boundary = Q.topRows(dim).transpose() * y_T_boundary;

            if (hard_case_branch) {
                int idx_min = 0;
                w_eig.minCoeff(&idx_min);
                if (std::abs(g_inner(idx_min)) < 1e-6) {
                    const double curr_norm = p_boundary.norm();
                    if (curr_norm < trust_radius) {
                        const Vector eig_vec_raw = Q.topRows(dim).transpose() * v_eig.col(idx_min);
                        const double tau = std::sqrt(std::max(0.0, trust_radius * trust_radius - curr_norm * curr_norm));
                        p_boundary += tau * (eig_vec_raw / eig_vec_raw.norm());
                    }
                }
            }

            p_current = p_boundary;
            const double resid_val = be * std::abs(y_T_boundary(dim - 1));

            if (resid_val < config_.tol || solve_final) {
                result.step = p_boundary;
                result.reason = hard_case_branch ? TerminationReason::HardCase : TerminationReason::TrustBoundary;
                result.hard_case = hard_case_branch;
                result.predicted_decrease = gradient.dot(result.step) + 0.5 * result.step.dot(hessian_->apply(result.step));
                return result;
            }
        }
        // solve_final implies should_solve_boundary and (resid_val < tol ||
        // solve_final) both hold, so a solve_final iteration always returns
        // above; the loop only continues here when still genuinely
        // unconverged (helper_PFCI.py's implicit fallthrough to the next k).
    }

    // helper_PFCI.py:15011/15247: ran out of iterations without meeting tol.
    result.step = p_current;
    result.reason = TerminationReason::MaxIterations;
    result.predicted_decrease = gradient.dot(result.step) + 0.5 * result.step.dot(hessian_->apply(result.step));
    return result;
}

} // namespace casscf
