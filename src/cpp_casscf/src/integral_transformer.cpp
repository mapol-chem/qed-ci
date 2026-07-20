#include "casscf/integral_transformer.hpp"

#include "casscf/ci_orbital_backend.hpp"
#include "casscf/ci_setup.hpp"

namespace casscf {
namespace {

// helper_PFCI.py:2376-2402: for a block of size M, row k enumerates pairs
// (r, s) with 0 <= r <= s < M (r outer loop, s inner loop) and stores
// [s, r] -- the larger index first. Shared formula for index_map_ab (M =
// n_virtual), index_map_kl (M = n_occupied), and index_map_pq (M = nmo).
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
      index_map_kl_(build_upper_triangular_pair_map(dims.n_occupied)),
      index_map_pq_(build_upper_triangular_pair_map(dims.nmo)) {}

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

void CasscfIntegralTransformer::transform_macroiteration(const Matrix& U_total) {
    CasscfContext& context = *context_;
    const int nmo = dims_.nmo;
    const int n_occupied = dims_.n_occupied;

    RowMajorMatrix U_rm = U_total;

    // context.J/context.K are pure outputs here -- see this class's header
    // doc comment for why (recomputed fully fresh from context.twoeint and
    // the full accumulated U_total, not incrementally).
    full_transformation_macroiteration(U_rm.data(), context.twoeint.data(), context.J.data(), context.K.data(),
                                         index_map_pq_.data(), index_map_kl_.data(), nmo, n_occupied);

    // helper_PFCI.py:3041-3069: immediately after the JK rebuild, refresh
    // the occupied-restricted "views" (self.occupied_J/K/h1/d_cmo/
    // fock_core) and self.E_core from the freshly-transformed J/K/
    // H_spatial2 -- see this class's header doc comment for why this lives
    // here (MacroiterationDriver::run's own doc comment already flagged
    // this rebuild as belonging "behind IntegralTransformer::
    // transform_macroiteration as an implementation detail"). Reuses
    // compute_active_block_intermediates (ci_setup.hpp) -- the exact same
    // fock_core/E_core formula, and here we DO want the E_core
    // reassignment (unlike CasscfCiStateAverageSolver's own use of this
    // same helper, where reassigning context.E_core would be wrong).
    ActiveBlockIntermediates active = compute_active_block_intermediates(context.H_spatial2, context.J, context.K,
                                                                            dims_);
    context.E_core = active.E_core;
    context.occupied_h1 = context.H_spatial2.topLeftCorner(n_occupied, n_occupied);
    context.occupied_d_cmo = context.d_cmo.topLeftCorner(n_occupied, n_occupied);
    context.occupied_fock_core = active.fock_core.topLeftCorner(n_occupied, n_occupied);

    // occupied_J/occupied_K: (n_occupied)^4, sliced from J/K's own
    // (n_occupied,n_occupied,nmo,nmo) shape restricted to the first
    // n_occupied of the last two axes too -- see CasscfContext::occupied_J's
    // corrected doc comment.
    context.occupied_J = Tensor4(n_occupied, n_occupied, n_occupied, n_occupied);
    context.occupied_K = Tensor4(n_occupied, n_occupied, n_occupied, n_occupied);
    for (int p = 0; p < n_occupied; ++p)
        for (int q = 0; q < n_occupied; ++q)
            for (int r = 0; r < n_occupied; ++r)
                for (int s = 0; s < n_occupied; ++s) {
                    context.occupied_J(p, q, r, s) = context.J(p, q, r, s);
                    context.occupied_K(p, q, r, s) = context.K(p, q, r, s);
                }
}

} // namespace casscf
