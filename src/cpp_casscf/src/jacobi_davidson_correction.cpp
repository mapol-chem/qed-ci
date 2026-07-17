#include "casscf/jacobi_davidson_correction.hpp"

namespace casscf {

Vector inner_solve_pcg(const std::shared_ptr<const HessianOperator>& hessian,
                        const Vector& gradient,
                        double alpha,
                        const Vector& residual_k,
                        double theta_k,
                        const Matrix& Q_conv,
                        const Vector& P_diag,
                        int max_inner_iter) {
    const int H_dim = static_cast<int>(residual_k.size());
    const int n_conv = static_cast<int>(Q_conv.rows());

    auto project_out = [&](const Vector& v) -> Vector {
        if (n_conv == 0) return v;
        const Vector coeffs = Q_conv * v;               // (n_conv)
        return v - Q_conv.transpose() * coeffs;          // deflate converged directions
    };

    BorderedHessianOperator augmented_op(hessian, gradient, alpha);
    auto apply_H_minus_theta = [&](const Vector& v) -> Vector {
        return augmented_op.apply(v) - theta_k * v;
    };

    Vector v_corr = Vector::Zero(H_dim);
    Vector r = -residual_k;

    Vector z = r.cwiseQuotient(P_diag);
    z = project_out(z);

    Vector p = z;
    double rho = r.dot(z);

    for (int iter = 0; iter < max_inner_iter; ++iter) {
        const Vector p_proj = project_out(p);
        Vector q = apply_H_minus_theta(p_proj);
        q = project_out(q);

        const double cg_alpha = rho / p_proj.dot(q);
        v_corr += cg_alpha * p_proj;
        r -= cg_alpha * q;

        if (r.norm() < 0.1 * residual_k.norm()) break;

        z = r.cwiseQuotient(P_diag);
        z = project_out(z);

        const double rho_new = r.dot(z);
        const double beta = rho_new / rho;
        rho = rho_new;

        p = z + beta * p;
    }

    return v_corr;
}

} // namespace casscf
