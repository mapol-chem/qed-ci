#pragma once

#include "casscf/types.hpp"

#include <unsupported/Eigen/CXX11/Tensor>

#include <utility>
#include <vector>

namespace casscf {

// RowMajor (not Eigen::Tensor's ColMajor default) so that .reshape() on
// these matches numpy's C-order reshape semantics directly -- most of the
// intermediates-building code below is a literal transcription of
// `some_flat_array.reshape((n, n, n, n))` calls in helper_PFCI.py, and
// getting the storage order wrong would silently transpose axes.
using Tensor2 = Eigen::Tensor<double, 2, Eigen::RowMajor>;
using Tensor3 = Eigen::Tensor<double, 3, Eigen::RowMajor>;
using Tensor4 = Eigen::Tensor<double, 4, Eigen::RowMajor>;

// Explicit element-wise copy from an Eigen::MatrixXd (ColMajor storage) to
// a Tensor2 (RowMajor storage) -- deliberately not a raw-memory TensorMap
// alias, since the storage orders differ and aliasing would silently
// transpose the result. These matrices are all small (orbital-space
// dimensions), so the copy cost is negligible next to correctness risk.
inline Tensor2 matrix_to_tensor2(const Matrix& m) {
    Tensor2 t(m.rows(), m.cols());
    for (int i = 0; i < m.rows(); ++i)
        for (int j = 0; j < m.cols(); ++j) t(i, j) = m(i, j);
    return t;
}

// Wraps a flat Eigen::VectorXd (e.g. self.D_tu_avg, self.D_tuvw_avg) as a
// higher-rank RowMajor tensor, matching `flat.reshape((dims...))` in the
// Python -- flat's C-order layout is exactly what TensorMap with RowMajor
// dims expects.
inline Eigen::TensorMap<const Tensor2> reshape2(const Vector& flat, int d0, int d1) {
    return Eigen::TensorMap<const Tensor2>(flat.data(), d0, d1);
}

inline Eigen::TensorMap<const Tensor4> reshape4(const Vector& flat, int d0, int d1, int d2, int d3) {
    return Eigen::TensorMap<const Tensor4>(flat.data(), d0, d1, d2, d3);
}

// Copies a rank-2 Tensor into an Eigen::MatrixXd (row-major Tensor entries
// map directly onto (row, col) regardless of Matrix's own ColMajor storage,
// since this goes through element-wise assignment, not a raw memory alias).
inline Matrix tensor2_to_matrix(const Tensor2& t) {
    Matrix m(t.dimension(0), t.dimension(1));
    for (int i = 0; i < t.dimension(0); ++i)
        for (int j = 0; j < t.dimension(1); ++j) m(i, j) = t(i, j);
    return m;
}

// Faithful port of the index_map construction loop, helper_PFCI.py:2349-2362:
// enumerates the non-redundant orbital-rotation parameter pairs (r, s) with
// r < s, skipping same-block pairs (both inactive, both active, or both
// virtual). Returns pairs as (s, r) -- s the larger index, r the smaller --
// matching self.index_map[i] = (s, r) exactly (see e.g. helper_PFCI.py's
// `s = self.index_map[i][0]; l = self.index_map[i][1]` usage). Shared by
// anything that packs a dense gradient/Hessian-diagonal array into its
// "reduced" vector form (build_hessian_diagonal, the gradient-building
// callers) using this same enumeration.
inline std::vector<std::pair<int, int>> build_index_map(const Dimensions& dims) {
    std::vector<std::pair<int, int>> index_map;
    index_map.reserve(dims.index_map_size());
    for (int r = 0; r < dims.nmo; ++r) {
        for (int s = r + 1; s < dims.nmo; ++s) {
            if (r < dims.n_in_a && s < dims.n_in_a) continue;
            if (dims.n_in_a <= r && r < dims.n_occupied && dims.n_in_a <= s && s < dims.n_occupied) continue;
            if (r >= dims.n_occupied && s >= dims.n_occupied) continue;
            index_map.emplace_back(s, r);
        }
    }
    return index_map;
}

} // namespace casscf
