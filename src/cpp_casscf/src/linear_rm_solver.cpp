#include "casscf/linear_rm_solver.hpp"

#include <Eigen/Eigenvalues>
#include <cmath>
#include <utility>

namespace casscf {
namespace {

// residual_minimization.py:39-58: canonical orthogonalization via
// symmetric eigendecomposition, discarding near-zero eigenvalues to
// explicitly remove linear dependencies in the trial-direction subspace.
Matrix canonical_orthogonalization(const Matrix& S, double thresh = 1e-9) {
    Eigen::SelfAdjointEigenSolver<Matrix> es(S);
    const Vector& eigvals = es.eigenvalues();
    const Matrix& eigvecs = es.eigenvectors();

    std::vector<int> keep;
    for (int i = 0; i < eigvals.size(); ++i) {
        if (eigvals(i) > thresh) keep.push_back(i);
    }
    if (keep.empty()) return Matrix(S.rows(), 0);

    Matrix X(S.rows(), static_cast<int>(keep.size()));
    for (size_t j = 0; j < keep.size(); ++j) {
        X.col(static_cast<int>(j)) = eigvecs.col(keep[j]) / std::sqrt(eigvals(keep[j]));
    }
    return X;
}

} // namespace

LinearRMSolver::LinearRMSolver(Vector b_vector, int max_subspace)
    : b_(std::move(b_vector)), max_subspace_(max_subspace) {}

Vector LinearRMSolver::update_subspace_and_extrapolate(const Vector& c_new, const Vector& s_new) {
    // residual_minimization.py:64-69: drop the second-oldest vector when
    // full, keeping index 0 (the current best, per get_solution()'s own
    // convention).
    if (static_cast<int>(c_vectors_.size()) == max_subspace_) {
        c_vectors_.erase(c_vectors_.begin() + 1);
        s_vectors_.erase(s_vectors_.begin() + 1);
    }
    c_vectors_.push_back(c_new);
    s_vectors_.push_back(s_new);
    const int size = static_cast<int>(c_vectors_.size());

    // residual_minimization.py:72-81.
    Matrix mat = Matrix::Zero(size, size);
    Matrix overlap = Matrix::Zero(size, size);
    Vector prod(size);
    for (int i = 0; i < size; ++i) {
        prod(i) = -s_vectors_[static_cast<size_t>(i)].dot(b_);
        for (int j = i; j < size; ++j) {
            const double mat_ij = s_vectors_[static_cast<size_t>(i)].dot(s_vectors_[static_cast<size_t>(j)]);
            mat(i, j) = mat_ij;
            mat(j, i) = mat_ij;
            const double overlap_ij = c_vectors_[static_cast<size_t>(i)].dot(c_vectors_[static_cast<size_t>(j)]);
            overlap(i, j) = overlap_ij;
            overlap(j, i) = overlap_ij;
        }
    }

    // residual_minimization.py:88-91.
    Matrix transform = canonical_orthogonalization(overlap);
    if (transform.cols() == 0) {
        return b_ + s_new;
    }

    // residual_minimization.py:93-100: solve the small (orthogonalized)
    // linear system; a singular mat_ortho falls back to the same "just use
    // the newest residual" behavior as an empty transform.
    Matrix mat_ortho = transform.transpose() * mat * transform;
    Vector prod_ortho = transform.transpose() * prod;

    Eigen::FullPivLU<Matrix> lu(mat_ortho);
    if (!lu.isInvertible()) {
        return b_ + s_new;
    }
    Vector coeffs_ortho = lu.solve(prod_ortho);
    Vector coeffs = transform * coeffs_ortho;

    // residual_minimization.py:104-119: rebuild the subspace basis so the
    // new optimal linear combination becomes vector 0.
    Matrix T = Matrix::Identity(size, size);
    T.col(0) = coeffs;

    Matrix old_c(c_vectors_[0].size(), size);
    Matrix old_s(s_vectors_[0].size(), size);
    for (int i = 0; i < size; ++i) {
        old_c.col(i) = c_vectors_[static_cast<size_t>(i)];
        old_s.col(i) = s_vectors_[static_cast<size_t>(i)];
    }
    Matrix new_c = old_c * T;
    Matrix new_s = old_s * T;
    for (int i = 0; i < size; ++i) {
        c_vectors_[static_cast<size_t>(i)] = new_c.col(i);
        s_vectors_[static_cast<size_t>(i)] = new_s.col(i);
    }

    // residual_minimization.py:121-124.
    return b_ + s_vectors_[0];
}

Vector LinearRMSolver::get_solution() const {
    if (c_vectors_.empty()) return Vector::Zero(b_.size());
    return c_vectors_[0];
}

} // namespace casscf
