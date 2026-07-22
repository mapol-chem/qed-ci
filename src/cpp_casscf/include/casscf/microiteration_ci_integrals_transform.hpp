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

// FAST path -- identical signature and outputs, same legacy/fast arrangement
// as build_intermediates(_internal)_fast: the loop form above stays as the
// line-for-line Python correspondence, the TAMM-retarget reference, and this
// path's oracle.
//
// Two independent kinds of saving are applied here, and the distinction
// matters if you ever change either implementation:
//
// 1. EXACT ALGEBRA -- holds for arbitrary inputs, no symmetry assumed.
//    The active_fock_core J/K terms are, in the legacy loop,
//      acc(t,u) = sum_{r,i} U(r,i) * sum_s Jslab(t,u)(r,s) * U(s,i)
//    i.e. an O(n_act^2 * nmo^2 * n_in_a) quintuple loop. The sum over the
//    inactive index factorizes out of the slab entirely:
//      acc(t,u) = trace(Uocc^T Jslab(t,u) Uocc) = <Jslab(t,u), D>,
//      D = Uocc Uocc^T  (an inactive density matrix, built ONCE)
//    dropping a whole factor of n_in_a. J and K then combine into a single
//    GEVM against <2J - K, D>, since the two terms enter with coefficients
//    +2 and -1.
//
// 2. ERI PERMUTATIONAL SYMMETRY -- verified on real captured dumps, NOT
//    assumed from the formulas (dumps_lih and
//    dumps_compare_h2o_stretched_2photon, agreement ~1e-16 against a ~1e0
//    scale):
//      J(k,l,r,s) = (rs|kl)  is symmetric in k<->l AND independently in r<->s
//      K(k,l,r,s) = (rk|sl)  is symmetric ONLY under the joint swap
//                            K(k,l,r,s) == K(l,k,s,r); each swap alone is
//                            genuinely NOT a symmetry (measured 0.37/1.03,
//                            so this is a real constraint, not a near-miss)
//    Both two-index transforms in active_twoeint are therefore computed over
//    a triangle of slabs only:
//      J part: N(v,w) = Uact^T Jslab(v,w) Uact, and Jslab(w,v) == Jslab(v,w)
//              so N(w,v) == N(v,w)          -> compute v <= w, copy
//      K part: M(t,v) = Tact^T Kslab(t,v) Tact, and Kslab(v,t) == Kslab(t,v)^T
//              so M(v,t) == M(t,v)^T        -> compute t <= v, transpose
//    Each halves the dominant O(n_act^3 * nmo^2) cost.
//
// CONSEQUENCE FOR TESTING: because of (2), this function is NOT equivalent to
// the legacy loop for arbitrary J/K -- only for J/K that carry the physical
// ERI symmetries. test_microiteration_ci_integrals_transform_fast.cpp
// therefore builds its random inputs the way real ones are structured, from
// a random 8-fold-symmetric (nmo)^4 ERI tensor g via J(k,l,r,s) = g(r,s,k,l)
// and K(k,l,r,s) = g(r,k,s,l), rather than filling J/K with independent
// random numbers. Filling them independently WILL produce spurious failures.
MicroiterationCiIntegralsResult microiteration_ci_integrals_transform_fast(
    const Matrix& U, double E_core_ref, const Matrix& fock_core, const Tensor4& L, const Tensor4& J,
    const Tensor4& K, const Tensor4& active_twoeint_ref, const Matrix& d_cmo_ref, const Dimensions& dims);

} // namespace casscf
