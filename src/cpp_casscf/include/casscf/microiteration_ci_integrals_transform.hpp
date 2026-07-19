#pragma once

#include "casscf/tensor_types.hpp"
#include "casscf/types.hpp"

namespace casscf {

struct MicroiterationCiIntegralsResult {
    double E_core2 = 0.0;
    Matrix active_fock_core; // (n_act_orb, n_act_orb)
    Tensor4 active_twoeint;  // (n_act_orb, n_act_orb, n_act_orb, n_act_orb)
    Matrix d_cmo;            // (nmo, nmo)
};

// Faithful port of microiteration_ci_integrals_transform, helper_PFCI.py:
// 8689-8798 (the `eigenvecs` parameter in the Python is unused in the
// function body -- confirmed by inspection -- so it's dropped here).
//
// Rotates the outer-microiteration reference-point quantities (fock_core,
// active_twoeint, d_cmo -- all side outputs of build_intermediates, see
// FullBlockIntermediates's doc comment) by the *accumulated* trial
// rotation U (self.U2 at the call site: identity at the top of the outer
// iteration, updated by each accepted inner-loop step) to produce the
// values microiteration_optimization6 actually feeds into the CI solver
// (via the not-yet-implemented CiStateAverageSolver) after each accepted
// step. Called once per accepted inner-loop iteration -- fock_core/L/
// active_twoeint_ref/E_core_ref stay fixed at their build_intermediates
// values for the whole outer iteration; only U (and hence the returned
// values) changes between calls.
MicroiterationCiIntegralsResult microiteration_ci_integrals_transform(
    const Matrix& U, double E_core_ref, const Matrix& fock_core, const Tensor4& L, const Tensor4& J,
    const Tensor4& K, const Tensor4& active_twoeint_ref, const Matrix& d_cmo_ref, const Dimensions& dims);

} // namespace casscf
