#pragma once

#include "casscf/intermediates.hpp" // StateAverageData
#include "casscf/tensor_types.hpp"
#include "casscf/types.hpp"

namespace casscf {

struct InternalTransformationResult {
    Tensor4 J;      // (n_occupied, n_occupied, n_occupied, n_occupied)
    Tensor4 K;      // (n_occupied, n_occupied, n_occupied, n_occupied)
    Matrix h1;      // (n_occupied, n_occupied)
    Matrix d_cmo1;  // (n_occupied, n_occupied)
};

// Faithful port of internal_transformation, helper_PFCI.py:6452-6517.
// Rotates the occupied-space one- and two-electron integrals
// (occupied_h1/occupied_d_cmo/occupied_J) by a trial rotation U -- only
// U's top-left (n_occupied, n_occupied) block is ever read (confirmed via
// this function's one call site in internal_optimization3, which always
// passes (n_occupied)-sized outputs and self.U_delta, an (nmo, nmo)
// matrix, as U). K is derived from the same transformed two-electron
// tensor as J via an index relabeling (K(i,j,k,l) = J_transformed(j,l,i,k)),
// not from a separate occupied_K input -- matches the Python, which never
// reads self.occupied_K here at all.
//
// The one-electron terms reduce to plain similarity transforms
// (h1 = U^T @ occupied_h1 @ U, same for d_cmo1) once the two chained
// einsums are multiplied out -- implemented as such rather than as literal
// nested loops. The two-electron term genuinely needs all 4 indices
// transformed (the standard 4-index "quarter transformation" cascade), so
// that's implemented as 4 sequential single-index contractions, matching
// the Python term for term.
InternalTransformationResult internal_transformation(const Matrix& U, const Matrix& occupied_h1,
                                                       const Matrix& occupied_d_cmo,
                                                       const Tensor4& occupied_J, const Dimensions& dims);

struct InternalOptimizationEnergyResult {
    double energy_change = 0.0; // sum_energy - E0, helper_PFCI.py:6871's return value
    bool accepted = false;      // sum_energy - E0 <= 0.0 || hard_case == 2, helper_PFCI.py:6815

    // Only meaningful when accepted == true. The Python conditionally
    // commits these into self.occupied_* / self.E_core / self.gkl2 as a
    // side effect right here (helper_PFCI.py:6815-6856); this port keeps
    // that as data the caller commits into its own persistent context,
    // rather than mutating shared state internally -- see
    // macroiteration_driver.hpp's InternalOptimizationStep doc comment for
    // why the driver layer needs to own that storage.
    Tensor4 occupied_J;         // (n_occupied)^4
    Tensor4 occupied_K;         // (n_occupied)^4
    Matrix occupied_h1;         // (n_occupied, n_occupied)
    Matrix occupied_d_cmo;      // (n_occupied, n_occupied)
    Matrix occupied_fock_core;  // (n_occupied, n_occupied)
    double E_core = 0.0;
    Matrix gkl2; // (n_act_orb, n_act_orb) -- feeds c_H_diag_cas_spin's gkl argument elsewhere
};

// Faithful port of internal_optimization_exact_energy, helper_PFCI.py:6734-6871.
// Evaluates the exact CASSCF energy with the proposed rotated occupied
// integrals (occupied_h1/occupied_d_cmo/occupied_J/occupied_K, as produced
// by internal_transformation) and returns the change relative to E0 -- this
// is what internal_optimization3's outer accept/reject loop compares
// against the predicted (quadratic-model) energy change. "Frobenius
// dot-of-flattened-arrays" terms in the Python (e.g.
// `np.dot(X.flatten(), Y_flat_1d)`) are implemented as elementwise
// products summed over all indices, which is exactly what flatten+dot
// computes for two same-shape arrays regardless of memory layout -- no
// reshape/transpose subtleties to get wrong there, unlike some of the
// intermediates-building einsums.
InternalOptimizationEnergyResult internal_optimization_exact_energy(
    double E0, const Matrix& eigenvecs, const Matrix& occupied_h1, const Matrix& occupied_d_cmo,
    const Tensor4& occupied_J, const Tensor4& occupied_K, int hard_case,
    const StateAverageData& sad, const Dimensions& dims);

// Faithful port of internal_optimization_predicted_energy, helper_PFCI.py:6873-6877
// (the "predicted" -- i.e. quadratic-model -- energy change for step Rai,
// evaluated against the fixed gradient_ai/hessian_ai built once per
// internal_optimization3 call). Plain quadratic form; ported directly
// rather than via einsum machinery. Note the Python's `print(...)` on
// helper_PFCI.py:6874 is debug output with no effect on the return value --
// not ported.
double internal_optimization_predicted_energy(const Vector& gradient_ai, const Matrix& hessian_ai,
                                                const Vector& Rai);

// Faithful port of step_control, helper_PFCI.py:8018-8025: classic
// trust-region ratio test (shrink on poor agreement, grow -- capped at
// 0.75 -- on very good agreement).
double step_control(double ratio, double trust_radius);

} // namespace casscf
