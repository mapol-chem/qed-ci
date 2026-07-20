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

    // self.twoeint: the full AO-basis-derived two-electron integral tensor
    // in the CURRENT MO basis, (nmo*nmo, nmo*nmo)-shaped (helper_PFCI.py:
    // 3510-3511 -- NOT (nmo,nmo,nmo,nmo), despite that being the natural
    // rank; the Python reshapes to this flat 2D form once and never back).
    // Unlike H_spatial2/d_cmo/U_total/J/K, this is FIXED for the whole
    // CASSCF run -- built once from the original AO integrals (self.twoeint1
    // + the DSE dipole-dipole term if not ignore_dse_terms, build2DSO(),
    // called before any macroiteration) and never mutated afterward.
    // IntegralTransformer::transform_macroiteration reads it (never writes
    // it) together with the FULL accumulated U_total to recompute J/K fully
    // fresh each macroiteration (full_transformation_macroiteration,
    // orbital.c) -- not incrementally from the previous J/K, avoiding
    // numerical error accumulation across many small rotations. This port
    // does not build twoeint itself (no AO-integral pipeline exists here);
    // the caller must supply it, same as H_spatial2/d_cmo/the initial J/K.
    // RowMajorMatrix (not Matrix), since it crosses the ci_orbital_backend.hpp
    // FFI boundary on every read and is too large to want to re-marshal from
    // column-major on every macroiteration (see tensor_types.hpp).
    RowMajorMatrix twoeint;

    // --- Occupied-occupied-restricted state committed by
    // internal_optimization_exact_energy on step acceptance
    // (helper_PFCI.py:6816-6856) and refreshed once per macroiteration by
    // IntegralTransformer::transform_macroiteration (helper_PFCI.py:
    // 3041-3069, immediately after the JK rebuild) ---
    // CORRECTION to an earlier, mistaken assumption from a prior session
    // (worth flagging explicitly, since it shaped this doc comment and
    // several call sites that read it): occupied_J/occupied_K are
    // genuinely (n_occupied, n_occupied, n_occupied, n_occupied)-shaped in
    // the real Python, NOT (n_occupied, n_occupied, nmo, nmo) matching
    // J/K's own shape -- confirmed directly against 3 separate
    // self.occupied_J = self.J[:, :, :n_occupied, :n_occupied] assignment
    // sites (helper_PFCI.py:1434-1438, 1663-1667, 3123-3126) and
    // self.occupied_J3 = self.occupied_J.reshape(n_occupied**2,
    // n_occupied**2) (helper_PFCI.py:1672-1674), which only makes
    // dimensional sense if occupied_J truly has n_occupied**4 total
    // elements. There are no "virtual-orbital columns" that ever carry
    // forward -- every write (including internal_optimization_exact_energy's
    // own, InternalOptimizationEnergyResult's doc comment in
    // internal_optimization.hpp) stays within the n_occupied-bounded region
    // regardless of which of the two shapes is used, so this correction is
    // safe against all existing committed code (none of it happened to
    // notice, since every existing test has nmo == n_occupied, where the
    // two shapes coincide numerically).
    Tensor4 occupied_J;
    Tensor4 occupied_K;
    Matrix occupied_h1;         // (n_occupied, n_occupied)
    Matrix occupied_d_cmo;      // (n_occupied, n_occupied)
    Matrix occupied_fock_core;  // (n_occupied, n_occupied)
    double E_core = 0.0;
    Matrix gkl2; // (n_act_orb, n_act_orb)

    // self.E_core2: a second, distinct "core energy" scalar
    // microiteration_optimization6 feeds into c_H_diag_cas_spin/c_get_roots'
    // constdouble[5], alongside (not instead of) E_core/gkl2/occupied_J/
    // occupied_fock_core/occupied_d_cmo above -- see
    // microiteration_optimization_step.hpp's class doc comment for why
    // MicroiterationOptimizationStep commits into these same occupied_*/gkl2
    // fields InternalOptimizationStep also writes, and why E_core2 needed
    // its own separate field rather than reusing E_core (helper_PFCI.py
    // never conflates the two: E_core2 is set at 8709/12184/12304, always
    // independently of E_core).
    double E_core2 = 0.0;

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
