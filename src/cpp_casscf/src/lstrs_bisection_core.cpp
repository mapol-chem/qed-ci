#include "casscf/lstrs_bisection_core.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace casscf {

TrustRegionResult solve_lstrs_bisection(const Vector& gradient,
                                         const Vector& hessian_diagonal,
                                         double trust_radius,
                                         int max_iter,
                                         const EigenpairAtAlphaFn& eigenpairs_at,
                                         const ModelValueFn& model_value,
                                         const HardCase2SolveFn& hard_case2_solve) {
    const int n = static_cast<int>(gradient.size());
    const double gradient_norm = gradient.norm();

    TrustRegionResult result;
    result.step = Vector::Zero(n);
    int hard_case = 0;

    if (gradient_norm > 1e-4) {
        double delta_u = hessian_diagonal.minCoeff();
        double alpha_l = 0.0;
        double alpha_u = delta_u + gradient_norm * trust_radius;
        double beta = alpha_u;

        double lambda1 = 0.0, lambda2 = 0.0, phi1 = 0.0, phi2 = 0.0;
        Vector x1 = Vector::Zero(n), x2 = Vector::Zero(n);
        Vector second_component = Vector::Zero(n);
        Vector x_tilde = Vector::Zero(n);
        double x2_norm = 0.0;
        bool both_roots_normalization = false;
        int count10 = 0;

        while (true) {
            result.iterations = count10 + 1;
            const bool restart = count10 != 0;
            double alpha_range = std::abs(alpha_u - alpha_l);
            TwoLowestEigenpairs eig = eigenpairs_at(beta, alpha_range, restart);
            double mu0 = eig.mu0, mu1 = eig.mu1;
            Vector w0 = eig.w0, w1 = eig.w1;

            if (count10 == 0) {
                alpha_l = mu0 - gradient_norm / trust_radius;
            }

            auto compute_uv = [&](const Vector& w, double& v, Vector& u, double& aa, double& bb) {
                const double norm_w = w.norm();
                v = w(0) / norm_w;
                u = w.tail(n) / norm_w;
                aa = gradient_norm * std::abs(v);
                bb = std::sqrt(1.0 - v * v);
            };

            double v1, v2, aa1, bb1, aa2, bb2;
            Vector u1, u2;
            compute_uv(w0, v1, u1, aa1, bb1);
            compute_uv(w1, v2, u2, aa2, bb2);

            if (mu0 > -1e-8 && u1.norm() < trust_radius * std::abs(v1)) {
                hard_case = 2;
                break;
            }

            double delta_u2 = mu0 - v1 * gradient.dot(u1) / (1.0 - v1 * v1);
            delta_u = std::min(delta_u, delta_u2);

            constexpr double epsilon_v = 1e-4;
            double alpha = beta;
            while ((aa1 <= epsilon_v * bb1) && (aa2 <= epsilon_v * bb2) &&
                   std::abs(alpha_u - alpha_l) > 1e-8 * std::max(std::abs(alpha_u), std::abs(alpha_l))) {
                alpha_u = alpha;
                alpha = (alpha_l + alpha_u) / 2.0;
                alpha_range = std::abs(alpha_u - alpha_l);
                eig = eigenpairs_at(alpha, alpha_range, /*restart=*/true);
                mu0 = eig.mu0; mu1 = eig.mu1; w0 = eig.w0; w1 = eig.w1;
                compute_uv(w0, v1, u1, aa1, bb1);
                compute_uv(w1, v2, u2, aa2, bb2);
                delta_u2 = mu0 - v1 * gradient.dot(u1) / (1.0 - v1 * v1);
                delta_u = std::min(delta_u, delta_u2);
            }

            if ((aa1 <= epsilon_v * bb1) && (aa2 <= epsilon_v * bb2)) {
                throw std::runtime_error("solve_lstrs_bisection: degenerate bisection interval (unhandled in source)");
            }
            both_roots_normalization = (aa1 > epsilon_v * bb1) && (aa2 > epsilon_v * bb2);
            beta = alpha;

            lambda1 = lambda2;
            x1 = x2;
            phi1 = phi2;

            Vector step10;
            if (aa1 > epsilon_v * bb1) {
                step10 = u1 / v1;
                second_component = u2;
                lambda2 = mu0;
                if (step10.norm() < trust_radius) alpha_l = beta;
                if (step10.norm() > trust_radius) alpha_u = beta;
            } else {
                step10 = u2 / v2;
                second_component = u1;
                lambda2 = mu1;
                alpha_u = beta;
            }
            const double step10_norm = step10.norm();
            x2 = step10;

            if (mu0 > -1e-8 && u1.norm() < trust_radius * std::abs(v1)) {
                hard_case = 2;
                break;
            }
            phi2 = -gradient.dot(x2);
            const double x1_norm = x1.norm();
            x2_norm = x2.norm();
            const double phi2_p = x2.dot(x2);
            const double phi1_p = x1.dot(x1);

            if (std::abs((step10_norm - trust_radius) / trust_radius) <= 1e-3 && mu0 <= 0.0) {
                result.step = step10;
                hard_case = 0;
                break;
            }

            const double vv1 = w0(0), vv2 = w1(0);
            const Vector uu1 = w0.tail(n), uu2 = w1.tail(n);
            const double qs = (1.0 + trust_radius * trust_radius) * (vv1 * vv1 + vv2 * vv2);
            constexpr double epsilon_hc = 1e-6;
            const double eta = epsilon_hc / (1.0 - epsilon_hc);
            double tau1 = 1.0, tau2 = 1.0;
            if (qs > 1.0) {
                const double s = std::sqrt(qs - 1.0);
                const double denom_tau = (vv1 * vv1 + vv2 * vv2) * std::sqrt(1.0 + trust_radius * trust_radius);
                tau1 = (vv1 - vv2 * s) / denom_tau;
                tau2 = (vv2 + vv1 * s) / denom_tau;
            } else if (std::abs(qs - 1.0) < 2e-308) {
                const double denom_tau = std::sqrt(vv1 * vv1 + vv2 * vv2);
                tau1 = vv1 / denom_tau;
                tau2 = vv2 / denom_tau;
            }
            x_tilde = (tau1 * uu1 + tau2 * uu2) / (tau1 * vv1 + tau2 * vv2);
            double psi_tilde = 0.5 * model_value(x_tilde);

            if (qs > 1.0 || std::abs(qs - 1.0) < 2e-308) {
                if ((mu1 - mu0) * tau2 * tau2 * (1.0 + trust_radius * trust_radius) < -2.0 * eta * psi_tilde) {
                    hard_case = 3;
                    break;
                }
                if (qs > 1.0) {
                    const double s = std::sqrt(qs - 1.0);
                    const double denom_tau = (vv1 * vv1 + vv2 * vv2) * std::sqrt(1.0 + trust_radius * trust_radius);
                    tau1 = (vv1 + vv2 * s) / denom_tau;
                    tau2 = (vv2 - vv1 * s) / denom_tau;
                    x_tilde = (tau1 * uu1 + tau2 * uu2) / (tau1 * vv1 + tau2 * vv2);
                    psi_tilde = 0.5 * model_value(x_tilde);
                    if ((mu1 - mu0) * tau2 * tau2 * (1.0 + trust_radius * trust_radius) < -2.0 * eta * psi_tilde) {
                        hard_case = 3;
                        break;
                    }
                }
            }

            if (std::abs(alpha_u - alpha_l) <= 1e-8 * std::max(std::abs(alpha_u), std::abs(alpha_l))) {
                if (x2_norm < trust_radius && !both_roots_normalization) hard_case = 1;
                if (both_roots_normalization) hard_case = 4;
                break;
            }

            if (count10 == 0) {
                beta = beta + (beta - lambda2) / x2_norm * (trust_radius - x2_norm) / trust_radius *
                       (trust_radius + 1.0 / x2_norm);
            } else {
                const double denom = trust_radius * (x2_norm - x1_norm);
                double lambda_c;
                if (std::abs(denom) > 2e-308) {
                    lambda_c = (lambda1 * x1_norm * (x2_norm - trust_radius) +
                                lambda2 * x2_norm * (trust_radius - x1_norm)) /
                               denom;
                } else {
                    lambda_c = delta_u;
                }
                if (lambda_c > delta_u) lambda_c = delta_u;

                if (std::abs(lambda2 - lambda1) <= 2e-308) {
                    beta = (alpha_l + alpha_u) / 2.0;
                } else {
                    const double omega_k = (lambda2 - lambda_c) / (lambda2 - lambda1);
                    const double num = x1_norm * x2_norm * (x2_norm - x1_norm) * (lambda1 - lambda_c) * (lambda2 - lambda_c);
                    const double denom2 = (omega_k * x2_norm + (1.0 - omega_k) * x1_norm) * (lambda2 - lambda1);
                    if (std::abs(denom2) <= 2e-308) {
                        beta = (alpha_l + alpha_u) / 2.0;
                    } else {
                        beta = lambda_c + omega_k * phi1 + (1.0 - omega_k) * phi2 + num / denom2;
                    }
                }
            }

            if (beta < alpha_l || beta > alpha_u) {
                if (count10 == 0) {
                    beta = delta_u + phi2 + phi2_p * (delta_u - lambda2);
                } else if (x2_norm < x1_norm) {
                    beta = delta_u + phi2 + phi2_p * (delta_u - lambda2);
                } else {
                    beta = delta_u + phi1 + phi1_p * (delta_u - lambda1);
                }
                if (beta < alpha_l || beta > alpha_u) {
                    beta = (alpha_l + alpha_u) / 2.0;
                }
            }

            ++count10;
            if (count10 == max_iter) break;
        }

        switch (hard_case) {
            case 1: {
                const Vector& first_component = x2;
                if (x2_norm > trust_radius) {
                    throw std::runtime_error("solve_lstrs_bisection: hard case 1 with ||x2|| > trust_radius (treated as a logic error in the source)");
                }
                const double xy = first_component.dot(second_component);
                const double x_sq = first_component.dot(first_component);
                const double y_sq = second_component.dot(second_component);
                const double delta = 4.0 * xy * xy - 4.0 * y_sq * (x_sq - trust_radius * trust_radius);
                const double t1 = (-2.0 * xy - std::sqrt(delta)) / (2.0 * y_sq);
                const double t2 = (-2.0 * xy + std::sqrt(delta)) / (2.0 * y_sq);
                result.step = first_component + std::min(std::abs(t1), std::abs(t2)) * second_component;
                result.reason = TerminationReason::HardCase;
                result.hard_case = true;
                break;
            }
            case 2: {
                result.step = hard_case2_solve();
                result.reason = TerminationReason::SuccessInteriorSolution;
                break;
            }
            case 3: {
                result.step = x_tilde;
                result.reason = TerminationReason::HardCase;
                result.hard_case = true;
                break;
            }
            case 4: {
                result.step = (x2_norm > trust_radius) ? (x2 / x2_norm * trust_radius) : x2;
                result.reason = TerminationReason::HardCase;
                result.hard_case = true;
                break;
            }
            default: {
                result.reason = (count10 == max_iter) ? TerminationReason::MaxIterations
                                                        : TerminationReason::TrustBoundary;
                break;
            }
        }
    } else if (gradient_norm > 1e-8) {
        result.step = hard_case2_solve();
        result.reason = TerminationReason::SuccessInteriorSolution;
    } else {
        result.reason = TerminationReason::SuccessGradientZero;
    }

    result.predicted_decrease = model_value(result.step);
    return result;
}

} // namespace casscf
