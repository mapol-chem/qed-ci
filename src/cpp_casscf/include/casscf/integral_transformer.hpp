#pragma once

#include "casscf/macroiteration_driver.hpp"
#include "casscf/types.hpp"

#include <cstdint>
#include <vector>

namespace casscf {

// The real IntegralTransformer, wrapping orbital.c's
// full_transformation_internal_optimization / full_transformation_macroiteration
// (see ci_orbital_backend.hpp for the extern "C" declarations, CMakeLists.txt
// for how orbital.c gets compiled and linked in).
//
// `index_map_ab`/`index_map_kl`/`index_map_pq` here are NOT the same thing as
// build_index_map() (tensor_types.hpp) -- that one enumerates the
// non-redundant orbital-ROTATION parameters (skipping inactive-inactive/
// active-active/virtual-virtual pairs, used by the trust-region solvers).
// These are plain upper-triangular pair enumerations of a single orbital
// block (virtual-virtual for `ab`, occupied-occupied for `kl`,
// full-MO-space for `pq`), matching helper_PFCI.py:2376-2402's
// `self.index_map_ab`/`self.index_map_kl`/`self.index_map_pq` exactly: for a
// block of size M, row k enumerates pairs (r, s) with 0 <= r <= s < M in
// nested-loop order (r outer, s inner), and stores [s, r] (the larger index
// first) -- i.e. the column order is swapped relative to the loop variables.
// Built once per Dimensions (these never change across a CASSCF run, same
// as in the Python).
class CasscfIntegralTransformer final : public IntegralTransformer {
public:
    CasscfIntegralTransformer(CasscfContext& context, Dimensions dims);

    // Wraps full_transformation_internal_optimization (orbital.c), the C
    // function backing BOTH of the Python call sites
    // transform_internal_rotation's doc comment (macroiteration_driver.hpp)
    // describes: the "RESTART MICROITERATION" branch's U_delta
    // (helper_PFCI.py:2907-2921) and internal_optimization3's own U1
    // (helper_PFCI.py:7852-7865) -- both are "apply an in-place similarity
    // transform of context.J/K/H_spatial2/d_cmo under U_delta," identical in
    // shape/semantics at both call sites. context.J/context.K are mutated
    // in place (Tensor4 is already RowMajor storage, matching the C
    // function's expected layout with no copy needed); context.H_spatial2/
    // context.d_cmo are marshaled through a row-major temporary and copied
    // back (Eigen::MatrixXd is column-major, but the C function requires
    // C-contiguous/row-major 2D arrays -- see integral_transformer.cpp).
    void transform_internal_rotation(const Matrix& U_delta) override;

    // NOT YET IMPLEMENTED. Would wrap full_transformation_macroiteration
    // (orbital.c), whose `h2e` argument needs the full (nmo,nmo,nmo,nmo)
    // two-electron integral tensor (self.twoeint in the Python,
    // helper_PFCI.py:5495) -- infrastructure this port has never needed
    // before (CasscfContext only carries the occupied-restricted J/K,
    // (n_occupied,n_occupied,nmo,nmo), not the full 4-index ERI tensor) and
    // doesn't build here. Also worth noting: every live call site of this
    // function in helper_PFCI.py is gated by `if self.density_fitting ==
    // False:` (e.g. helper_PFCI.py:2976), and self.twoeint's own runtime
    // shape at those call sites looks inconsistent with what this C
    // function's `ndim=4` ctypes argtype requires (self.twoeint is set as a
    // 2D-reshaped array by build2DSO, helper_PFCI.py:3510-3512, never
    // reshaped back to 4D) -- i.e. this call path may be effectively dead
    // under the density-fitted path this codebase actually exercises in
    // practice. Throws std::logic_error if called; a real implementation
    // needs a resolution to that ambiguity (and a CasscfContext extension
    // for the full ERI tensor) before it can be written faithfully.
    void transform_macroiteration(const Matrix& U_total) override;

private:
    CasscfContext* context_;
    Dimensions dims_;
    std::vector<int32_t> index_map_ab_; // (n_virtual*(n_virtual+1)/2, 2), flat row-major
    std::vector<int32_t> index_map_kl_; // (n_occupied*(n_occupied+1)/2, 2), flat row-major
};

} // namespace casscf
