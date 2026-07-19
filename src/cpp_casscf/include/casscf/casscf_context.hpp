#pragma once

#include "casscf/tensor_types.hpp"
#include "casscf/types.hpp"

namespace casscf {

// Bundles the persistent, cross-call CASSCF orbital/integral state that in
// the Python lives as instance attributes on `self` (self.H_spatial2,
// self.d_cmo, self.U_total, self.J, self.K, self.occupied_J/K/h1/d_cmo/
// fock_core, self.E_core, self.gkl2, self.H_diag3) -- read and mutated both
// across macroiterations *and* from within a single internal_optimization3
// call, which rotates self.H_spatial2/self.d_cmo/self.J/self.K/self.U_total
// in place on step acceptance (helper_PFCI.py:7787-7805) *before* control
// returns to the macroiteration loop body that itself later rotates the
// same H_spatial2/d_cmo/U_total again (helper_PFCI.py:2925-2937). That's
// one shared piece of state touched from two different call sites within a
// single macroiteration, not two independent copies -- so
// MacroiterationDriver, InternalOptimizationStep, MicroiterationOptimizationStep,
// and IntegralTransformer implementations for one CASSCF run must all
// operate on the *same* CasscfContext instance (passed by reference) rather
// than each threading their own local copies through parameters/return
// values, the way MacroiterationDriver::run originally did for
// H_spatial2/d_cmo/U_total. See macroiteration_driver.hpp's
// InternalOptimizationStep doc comment.
struct CasscfContext {
    // --- Full molecular-orbital-space state ---
    Matrix H_spatial2; // (nmo, nmo) -- one-electron Hamiltonian in the current MO basis
    Matrix d_cmo;       // (nmo, nmo) -- PF dipole-coupling integrals in the current MO basis
    Matrix U_total;     // (nmo, nmo) -- accumulated orbital rotation since the start of the CASSCF run

    // self.J / self.K: (n_occupied, n_occupied, nmo, nmo) -- see
    // FullBlockIntermediates's doc comment in intermediates.hpp for why
    // this shape, not (nmo, nmo, nmo, nmo). Populated by IntegralTransformer.
    Tensor4 J;
    Tensor4 K;

    // --- Occupied-occupied-restricted state committed by
    // internal_optimization_exact_energy on step acceptance
    // (helper_PFCI.py:6816-6856) ---
    // occupied_J / occupied_K share J/K's (n_occupied, n_occupied, nmo, nmo)
    // shape -- internal_optimization_exact_energy only ever writes their
    // [:, :, :n_occupied, :n_occupied] sub-block (see
    // InternalOptimizationEnergyResult's doc comment in
    // internal_optimization.hpp); the remaining virtual-orbital columns
    // carry forward untouched from the last transform_macroiteration call.
    Tensor4 occupied_J;
    Tensor4 occupied_K;
    Matrix occupied_h1;         // (n_occupied, n_occupied)
    Matrix occupied_d_cmo;      // (n_occupied, n_occupied)
    Matrix occupied_fock_core;  // (n_occupied, n_occupied)
    double E_core = 0.0;
    Matrix gkl2; // (n_act_orb, n_act_orb)

    // self.H_diag3: the orbital-Hessian-diagonal guess, recomputed via
    // c_H_diag_cas_spin (a compiled extension outside this module's scope,
    // like the CI Davidson solver behind CiStateAverageSolver) each time
    // internal_optimization3 accepts a step. Slot reserved here so a real
    // InternalOptimizationStep has somewhere to put it once that dependency
    // exists; nothing in this module populates it yet.
    Vector H_diag3;

    // self.D_tu_avg / self.D_tuvw_avg / self.Dpe_tu_avg: the state-averaged
    // RDMs built by build_state_average_rdms (helper_PFCI.py:7992+) once
    // per macroiteration, right after the CI diagonalization that produces
    // `eigenvecs` -- i.e. exactly what CiStateAverageResult (see
    // macroiteration_driver.hpp) is meant to carry out of
    // CiStateAverageSolver::solve(). Belongs here (not just on
    // CiStateAverageResult) because, like H_spatial2/d_cmo/U_total/J/K
    // above, these are read afterward by InternalOptimizationStep across
    // that same macroiteration -- MacroiterationDriver::run copies them
    // from each CiStateAverageResult into context right after every solve()
    // call, matching how self.D_tu_avg etc. are persistent instance state
    // in the Python, not something threaded through a return value.
    Matrix D_tu_avg;    // (n_act_orb, n_act_orb)
    Tensor4 D_tuvw_avg; // (n_act_orb, n_act_orb, n_act_orb, n_act_orb)
    Matrix Dpe_tu_avg;  // (n_act_orb, n_act_orb)
};

} // namespace casscf
