#include "casscf/bfgs_operator.hpp"

#include "casscf/orbital_sigma.hpp"

#include <utility>

namespace casscf {

BfgsOperator::BfgsOperator(Matrix U_zero, Matrix A_tilde_zero, Tensor4 G_zero, Dimensions dims, int m_history)
    : U_zero_(std::move(U_zero)),
      A_tilde_zero_(std::move(A_tilde_zero)),
      G_zero_(std::move(G_zero)),
      dims_(dims),
      m_history_(m_history) {}

Vector BfgsOperator::apply(const Vector& v) const {
    Vector sigma = orbital_sigma3(U_zero_, A_tilde_zero_, G_zero_, v, dims_);
    for (const BfgsHistoryEntry& e : history_) {
        const double y_dot_v = e.y.dot(v);
        const double s_dot_sigma = e.s.dot(sigma);
        sigma = sigma + e.y * (y_dot_v * e.rho_y) - e.Bs * (s_dot_sigma * e.rho_Bs);
    }
    return sigma;
}

void BfgsOperator::update(const Vector& s_vec, const Vector& y_vec) {
    const Vector Bs_vec = apply(s_vec);
    const double ys_dot = y_vec.dot(s_vec);
    const double sBs_dot = s_vec.dot(Bs_vec);
    const double damping_sigma = 0.1; // helper_PFCI.py:11154

    Vector final_y;
    double final_rho_y = 0.0;
    if (ys_dot < damping_sigma * sBs_dot) {
        const double theta = ((1.0 - damping_sigma) * sBs_dot) / (sBs_dot - ys_dot);
        final_y = theta * y_vec + (1.0 - theta) * Bs_vec;
        final_rho_y = 1.0 / (damping_sigma * sBs_dot);
    } else {
        final_y = y_vec;
        final_rho_y = 1.0 / ys_dot;
    }
    const double rho_Bs = 1.0 / sBs_dot;

    history_.push_back(BfgsHistoryEntry{s_vec, final_y, Bs_vec, final_rho_y, rho_Bs});
    if (static_cast<int>(history_.size()) > m_history_) history_.erase(history_.begin());
}

void BfgsOperator::reset_reference(Matrix U_zero, Matrix A_tilde_zero, Tensor4 G_zero) {
    U_zero_ = std::move(U_zero);
    A_tilde_zero_ = std::move(A_tilde_zero);
    G_zero_ = std::move(G_zero);
    history_.clear();
}

} // namespace casscf
