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

    // Wraps full_transformation_macroiteration (orbital.c) -- run once per
    // macroiteration on the fully-accumulated context.U_total
    // (helper_PFCI.py:2976-2991, the `if self.density_fitting == False:`
    // branch). Unlike transform_internal_rotation, context.J/context.K are
    // pure OUTPUTS here (not also inputs): this recomputes them fully fresh
    // from context.twoeint (the FIXED, never-mutated AO-derived
    // two-electron-integral tensor -- see CasscfContext::twoeint's doc
    // comment) and the *full* accumulated rotation, not incrementally from
    // the previous J/K.
    //
    // CORRECTION to an earlier, mistaken conclusion from a prior session
    // (worth flagging explicitly, since it was previously documented here
    // and in the README as a real gap): this call path is NOT dead code
    // under density fitting. Confirmed directly: `density_fitting` has no
    // default and is only ever set True if `"df_basis_scf"` is a key in the
    // caller's psi4_options_dict (helper_PFCI.py:4100-4103, 4266-4267) --
    // grepping every driver/example/test script in this repo (including
    // cpp_casscf/validation/dump_lih_case.py) shows none of them set that
    // key, so `density_fitting == False` (this function's branch) is the
    // ONLY path any real run in this repo actually takes; the
    // density-fitted alternative (`transform_JK_with_df`,
    // helper_PFCI.py:6688-6732) is unused dead weight in practice, not the
    // other way around. The apparent `h2e` ndim mismatch that led to the
    // earlier "possibly dead code" conclusion was simply a misreading of
    // which ctypes argtypes-list entry corresponds to which C parameter --
    // re-checked directly against helper_PFCI.py:345-354: `h2e`'s argtype
    // really is `ndim=2` (matching self.twoeint's real, persistent
    // (nmo*nmo, nmo*nmo) shape exactly, set once at helper_PFCI.py:
    // 3510-3511 and never reshaped back to 4D); only `J`/`K` (the 3rd/4th
    // argtypes-list entries) are `ndim=4`. This function is real, live,
    // and correctly-shaped as written in the Python.
    //
    // Also performs the occupied_J/K/h1/d_cmo/fock_core/E_core refresh that
    // immediately follows the JK rebuild in the Python (helper_PFCI.py:
    // 3041-3069) -- MacroiterationDriver::run's own doc comment already
    // anticipated this belongs here ("an implementation detail of
    // IntegralTransformer... operating on context.J/context.K"). Found to
    // be load-bearing, not optional bookkeeping: context.E_core is read by
    // CasscfCiStateAverageSolver/CasscfCiSetup (which deliberately do NOT
    // recompute it themselves -- see ActiveBlockIntermediates's doc
    // comment), so without this refresh it would silently go stale after
    // the first macroiteration.
    void transform_macroiteration(const Matrix& U_total) override;

private:
    CasscfContext* context_;
    Dimensions dims_;
    std::vector<int32_t> index_map_ab_; // (n_virtual*(n_virtual+1)/2, 2), flat row-major
    std::vector<int32_t> index_map_kl_; // (n_occupied*(n_occupied+1)/2, 2), flat row-major
    std::vector<int32_t> index_map_pq_; // (nmo*(nmo+1)/2, 2), flat row-major
};

} // namespace casscf
