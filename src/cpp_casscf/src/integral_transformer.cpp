#include "casscf/integral_transformer.hpp"

#include "casscf/ci_orbital_backend.hpp"

#include <Eigen/Dense>
#include <stdexcept>

namespace casscf {
namespace {

// full_transformation_internal_optimization/full_transformation_macroiteration
// (orbital.c) require C-contiguous (row-major) 2D double* buffers --
// Eigen::MatrixXd (casscf::Matrix) is column-major by default, so 2D
// arguments need to go through this row-major type rather than `.data()`
// directly (same reasoning tensor_types.hpp's matrix_to_tensor2 documents
// for the analogous Tensor2 case). Assignment between Eigen storage orders
// is a correct element-wise copy, not a raw memcpy -- safe both ways.
using RowMajorMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// helper_PFCI.py:2376-2402: for a block of size M, row k enumerates pairs
// (r, s) with 0 <= r <= s < M (r outer loop, s inner loop) and stores
// [s, r] -- the larger index first. Shared formula for index_map_ab (M =
// n_virtual) and index_map_kl (M = n_occupied); index_map_pq (M = nmo) is
// not needed until transform_macroiteration is implemented.
std::vector<int32_t> build_upper_triangular_pair_map(int M) {
    std::vector<int32_t> map(static_cast<size_t>(M) * (M + 1) / 2 * 2);
    int idx = 0;
    for (int r = 0; r < M; ++r) {
        for (int s = r; s < M; ++s) {
            map[static_cast<size_t>(idx) * 2 + 0] = s;
            map[static_cast<size_t>(idx) * 2 + 1] = r;
            ++idx;
        }
    }
    return map;
}

} // namespace

CasscfIntegralTransformer::CasscfIntegralTransformer(CasscfContext& context, Dimensions dims)
    : context_(&context), dims_(dims), index_map_ab_(build_upper_triangular_pair_map(dims.n_virtual)),
      index_map_kl_(build_upper_triangular_pair_map(dims.n_occupied)) {}

void CasscfIntegralTransformer::transform_internal_rotation(const Matrix& U_delta) {
    CasscfContext& context = *context_;
    const int nmo = dims_.nmo;
    const int n_occupied = dims_.n_occupied;

    RowMajorMatrix U_rm = U_delta;
    RowMajorMatrix h_rm = context.H_spatial2;
    RowMajorMatrix d_cmo_rm = context.d_cmo;

    // Same-object reuse for the "1" (output) parameters as the Python call
    // sites (helper_PFCI.py:2907-2921, 7852-7865), which pass self.J/self.K/
    // self.H_spatial2/self.d_cmo for both the input and output argument
    // slots -- context.J/K (already row-major Tensor4 storage) are mutated
    // truly in place; h_rm/d_cmo_rm alias correctly because the C function
    // fully consumes each input (via an intermediate cblas_dgemm product)
    // before writing its corresponding output, confirmed by reading
    // orbital.c:869-881.
    full_transformation_internal_optimization(U_rm.data(), context.J.data(), context.K.data(), h_rm.data(),
                                                d_cmo_rm.data(), context.J.data(), context.K.data(), h_rm.data(),
                                                d_cmo_rm.data(), index_map_ab_.data(), index_map_kl_.data(), nmo,
                                                n_occupied);

    context.H_spatial2 = h_rm;
    context.d_cmo = d_cmo_rm;
}

void CasscfIntegralTransformer::transform_macroiteration(const Matrix&) {
    throw std::logic_error(
        "CasscfIntegralTransformer::transform_macroiteration is not implemented -- "
        "see this class's header doc comment for why (needs the full (nmo,nmo,nmo,nmo) "
        "two-electron integral tensor, which this port doesn't build, and the live call "
        "sites may be dead code under the density-fitted path this codebase actually uses).");
}

} // namespace casscf
