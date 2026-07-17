#include "casscf/pcg_trust_region_solver.hpp"

#include <algorithm>
#include <cmath>

namespace casscf {

namespace {

// Positive root of a*tau^2 + b*tau + c = 0, matching the quadratic solved at
// both the negative-curvature and trust-boundary branches of
// solve_pcg_trust_region (helper_PFCI.py:16211-16238).
double positive_root(double a, double b, double c) {
    const double discriminant = b * b - 4.0 * a * c;
    return (-b + std::sqrt(discriminant)) / (2.0 * a);
}

} // namespace

PcgTrustRegionSolver::PcgTrustRegionSolver(std::shared_ptr<const HessianOperator> hessian,
                                            Vector diag_preconditioner,
                                            double tol,
                                            int max_iter)
    : hessian_(std::move(hessian)),
      diag_preconditioner_(std::move(diag_preconditioner)),
      tol_(tol),
      max_iter_(max_iter) {
    // M_diag = np.maximum(np.abs(M_diag), epsilon), helper_PFCI.py:16175-16178.
    constexpr double epsilon = 1e-6;
    diag_preconditioner_ = diag_preconditioner_.cwiseAbs().cwiseMax(epsilon);
}

TrustRegionResult PcgTrustRegionSolver::solve(const Vector& gradient, double trust_radius) const {
    TrustRegionResult result;
    const int n = static_cast<int>(gradient.size());

    Vector p = Vector::Zero(n);
    Vector r = -gradient;
    Vector z = r.cwiseQuotient(diag_preconditioner_);
    Vector d = z;
    double r_dot_z = r.dot(z);

    if (std::sqrt(r_dot_z) < tol_) {
        result.step = p;
        result.reason = TerminationReason::SuccessGradientZero;
        return result;
    }

    for (int j = 0; j < max_iter_; ++j) {
        result.iterations = j + 1;

        const Vector Hd = hessian_->apply(d);
        const double d_H_d = d.dot(Hd);

        if (d_H_d <= 0.0) {
            const double a = d.dot(d);
            const double b = 2.0 * p.dot(d);
            const double c = p.dot(p) - trust_radius * trust_radius;
            const double tau = positive_root(a, b, c);
            result.step = p + tau * d;
            result.reason = TerminationReason::NegativeCurvature;
            return result;
        }

        const double alpha = r_dot_z / d_H_d;
        const Vector p_new = p + alpha * d;

        if (p_new.norm() > trust_radius) {
            const double a = d.dot(d);
            const double b = 2.0 * p.dot(d);
            const double c = p.dot(p) - trust_radius * trust_radius;
            const double tau = positive_root(a, b, c);
            result.step = p + tau * d;
            result.reason = TerminationReason::TrustBoundary;
            return result;
        }

        p = p_new;
        r = r - alpha * Hd;

        if (r.norm() < tol_) {
            result.step = p;
            result.reason = TerminationReason::SuccessInteriorSolution;
            return result;
        }

        z = r.cwiseQuotient(diag_preconditioner_);
        const double r_dot_z_new = r.dot(z);
        const double beta = r_dot_z_new / r_dot_z;
        d = z + beta * d;
        r_dot_z = r_dot_z_new;
    }

    result.step = p;
    result.reason = TerminationReason::MaxIterations;
    return result;
}

} // namespace casscf
