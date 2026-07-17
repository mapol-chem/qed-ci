#pragma once

#include "casscf/hessian_operator.hpp"
#include <memory>

namespace casscf {

// Port of inner_solve_pcg, helper_PFCI.py:14455-14543. A 5-iteration
// truncated, diagonally-preconditioned CG solving the deflated correction
// equation
//     (I - Q_conv Q_conv^T)(H_aug - theta_k I)(I - Q_conv Q_conv^T) v = -residual_k
// for the Jacobi-Davidson correction vector v, where H_aug is the bordered
// operator (helper_PFCI.py:14474-14497 builds exactly aug_matvec's action,
// i.e. BorderedHessianOperator here) and Q_conv's rows are the already-
// converged eigenvectors being projected out. Used once at least one root
// has converged, in place of the plain diagonal Davidson preconditioner
// (see DavidsonAugmentedHessianSolver).
Vector inner_solve_pcg(const std::shared_ptr<const HessianOperator>& hessian,
                        const Vector& gradient,
                        double alpha,
                        const Vector& residual_k,
                        double theta_k,
                        const Matrix& Q_conv,
                        const Vector& P_diag,
                        int max_inner_iter = 5);

} // namespace casscf
