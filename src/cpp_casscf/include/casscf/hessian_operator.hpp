#pragma once

#include "casscf/types.hpp"
#include <functional>
#include <memory>

namespace casscf {

// Abstraction over "apply the orbital Hessian to a vector", so PCG and GLTR
// can share one implementation whether the Hessian is:
//   (a) a small dense matrix, explicitly formed and eigh'd for the
//       active-inactive step (hessian_tilde_ai in internal_optimization3,
//       helper_PFCI.py:6970-6991), or
//   (b) applied matrix-free via a sigma-vector-style build over the full
//       non-redundant rotation space (mv2 -> build_sigma_reduced5,
//       helper_PFCI.py:16704-16738; and the BFGS variant get_bfgs_mv,
//       helper_PFCI.py:10792-10841).
class HessianOperator {
public:
    virtual ~HessianOperator() = default;
    virtual Vector apply(const Vector& v) const = 0;
    virtual int dimension() const = 0;
};

// Explicit dense Hessian, used by the Davidson+LSTRS solver on the small
// active-inactive block (dimension n_act_orb * n_in_a).
class DenseHessianOperator final : public HessianOperator {
public:
    explicit DenseHessianOperator(Matrix hessian) : hessian_(std::move(hessian)) {}

    Vector apply(const Vector& v) const override { return hessian_ * v; }
    int dimension() const override { return static_cast<int>(hessian_.rows()); }

    const Matrix& matrix() const { return hessian_; }

private:
    Matrix hessian_;
};

// Matrix-free Hessian action, used by GLTR/PCG on the full reduced rotation
// space (dimension = Dimensions::index_map_size()). The callable wraps
// whatever kernel provides the sigma-vector build (mv2 for the exact
// intermediates-based Hessian, get_bfgs_mv for the L-BFGS approximation) so
// the solver code itself never needs to know which one it's driving.
class MatrixFreeHessianOperator final : public HessianOperator {
public:
    using ApplyFn = std::function<Vector(const Vector&)>;

    MatrixFreeHessianOperator(ApplyFn apply_fn, int dim)
        : apply_fn_(std::move(apply_fn)), dim_(dim) {}

    Vector apply(const Vector& v) const override { return apply_fn_(v); }
    int dimension() const override { return dim_; }

private:
    ApplyFn apply_fn_;
    int dim_;
};

// Wraps an inner HessianOperator H (acting on the n-dim reduced rotation
// space) into the (n+1)-dim "bordered"/augmented operator
//     [ alpha   g^T ] [v0]   [alpha*v0 + g.v_reduced   ]
//     [ g       H   ] [v_reduced] = [g*v0 + H*v_reduced]
// used by the LSTRS-family solvers (dense in LstrsSolver, matrix-free in
// DavidsonAugmentedHessianSolver). Faithful port of aug_matvec,
// helper_PFCI.py:14566-14584 -- note that in the Python, orbital_sigma3
// computes only the H*v_reduced piece into Hp[:,1:]; the alpha*v0 and
// g-dot-v_reduced border terms are added by the caller exactly as done here.
class BorderedHessianOperator final : public HessianOperator {
public:
    BorderedHessianOperator(std::shared_ptr<const HessianOperator> inner, Vector gradient, double alpha)
        : inner_(std::move(inner)), gradient_(std::move(gradient)), alpha_(alpha) {}

    Vector apply(const Vector& v) const override {
        const double v0 = v(0);
        const Vector v_reduced = v.tail(inner_->dimension());
        Vector result(dimension());
        result(0) = alpha_ * v0 + gradient_.dot(v_reduced);
        result.tail(inner_->dimension()) = inner_->apply(v_reduced) + gradient_ * v0;
        return result;
    }

    int dimension() const override { return inner_->dimension() + 1; }

private:
    std::shared_ptr<const HessianOperator> inner_;
    Vector gradient_;
    double alpha_;
};

} // namespace casscf
