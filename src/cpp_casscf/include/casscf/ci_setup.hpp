#pragma once

#include "casscf/internal_optimization_step.hpp" // CasscfPhysicalConstants
#include "casscf/tensor_types.hpp"
#include "casscf/types.hpp"

#include <cstdint>
#include <vector>

namespace casscf {

// CI-solve-specific settings this module hasn't needed before -- the
// active-space electron count and Davidson/spin-adaptation configuration
// PFHamiltonianGenerator.__init__ reads from cavity_dictionary
// (helper_PFCI.py:3116+, parseCavityOptions) and reuses for a whole CASSCF
// run. Deliberately a separate struct from CasscfPhysicalConstants (which
// CasscfInternalOptimizationStep/CasscfMicroiterationOptimizationStep also
// use, and which is scoped to what THEY need) rather than extending it --
// none of these fields are relevant to those two classes.
struct CasscfCiConfig {
    int n_act_a = 0; // number of alpha electrons in the active space

    int davidson_roots = 1;
    double davidson_threshold = 1e-5;
    // Per-root subspace-size MULTIPLIERS, matching self.davidson_indim/
    // self.davidson_maxdim (helper_PFCI.py:3229-3236) -- NOT the raw
    // indim/maxdim themselves. CasscfCiSetup derives and stores the actual
    // indim/maxdim (= multiplier * davidson_roots) once, since num_alpha/
    // H_dim (needed to validate them) aren't known until construction.
    int davidson_indim = 4;
    int davidson_maxdim = 20;
    int davidson_maxiter = 100;

    double target_spin = -1.0; // -1.0: no spin adaptation (helper_PFCI.py:3281-3289)
    bool ignore_dse_terms = false;
};

// The active-active-restricted Fock/two-electron-integral blocks
// (self.active_fock_core/self.active_twoeint, and the full-space
// self.fock_core they're sliced from) that BOTH CasscfCiSetup's one-time
// index_Hdiag derivation and CasscfCiStateAverageSolver's every-macroiteration
// CI-input build need -- the exact same formula FullBlockIntermediates's
// side outputs already use (intermediates.cpp, helper_PFCI.py:5754-5760,
// 5955-5969), transcribed fresh here rather than reused via
// build_intermediates(): the Python's per-macroiteration CI-solve block
// (helper_PFCI.py:2424-2488) computes this INLINE, WITHOUT calling
// build_intermediates -- and critically, unlike build_intermediates
// (which reassigns self.E_core as a side effect), this block does NOT
// touch self.E_core at all (it's read here, whatever it currently holds
// from the last internal_optimization3/microiteration_optimization6 call,
// never recomputed). Reusing build_intermediates here would incorrectly
// reassign context.E_core as an unwanted side effect -- see
// ci_state_average_solver.cpp for where this matters.
struct ActiveBlockIntermediates {
    Matrix fock_core;        // (nmo, nmo) -- full-space fock_core active_fock_core is sliced from
    double E_core = 0.0;     // sum_j H_spatial2(j,j) + fock_core(j,j), j < n_in_a
    Matrix active_fock_core; // (n_act_orb, n_act_orb)
    Tensor4 active_twoeint;  // (n_act_orb)^4
};

ActiveBlockIntermediates compute_active_block_intermediates(const Matrix& H_spatial2, const Tensor4& J,
                                                               const Tensor4& K, const Dimensions& dims);

// Zero-pads active_fock_core/active_twoeint into occupied-sized arrays with
// the inactive block left zero -- same formula already established by
// internal_optimization_exact_energy/microiteration_ci_integrals_transform's
// commit_ci_solver_inputs, just producing fresh LOCAL arrays here rather
// than committing into CasscfContext (see ci_state_average_solver.cpp for
// why: this class doesn't read or write CasscfContext's occupied_* staging
// fields at all).
struct OccupiedCiBlocks {
    Matrix occupied_fock_core; // (n_occupied, n_occupied)
    // (n_occupied, n_occupied, n_occupied, n_occupied), RowMajor -- .data()
    // is directly the C-contiguous (n_occupied^2, n_occupied^2) flat buffer
    // get_roots/build_H_diag_cas_spin's h2e parameter expects.
    Tensor4 occupied_twoeint;
};

OccupiedCiBlocks build_occupied_ci_blocks(const ActiveBlockIntermediates& active, const Dimensions& dims);

// Bundles the CI graph/string-table setup (self.table/self.table_creation/
// self.table_annihilation/self.b_array/self.Y, built via get_graph/
// get_string) and the two setup-time-only derived quantities
// (self.S_diag/self.S_diag_projection via build_S_diag, and
// self.index_Hdiag -- see below for why this one is setup-time-only despite
// looking like it should track the per-macroiteration H_diag3) --
// PFHamiltonianGenerator.__init__'s CI setup block (helper_PFCI.py:
// 1507-1613, CAS "direct" branch), which depends only on the active-space
// definition (n_act_a/n_act_orb/n_in_a/N_p/num_alpha) and is computed once
// per CASSCF run, reused by every CasscfCiStateAverageSolver::solve() call.
//
// self.H_diag (filled by get_string as a side effect alongside table/
// table_creation/table_annihilation/b_array) is deliberately NOT kept --
// confirmed by exhaustive grep that self.H_diag is used as get_roots's
// Hdiag argument at exactly ONE call site in the whole file (the very
// first, pre-optimization CI solve inside __init__, helper_PFCI.py:
// 1936-1954), and self.index_Hdiag is ALWAYS self.H_diag3.argsort() (never
// self.H_diag.argsort()) at both places it's computed. Every other
// get_roots call (all six remaining sites, including the per-macroiteration
// one this module's CasscfCiStateAverageSolver ports) uses self.H_diag3.
// get_string still needs calling for its OTHER four outputs, so this
// class calls it but discards the H_diag buffer it also writes into.
//
// index_Hdiag is a genuine correctness subtlety, not an oversight worth
// "fixing": self.index_Hdiag = self.H_diag3.argsort() is computed exactly
// ONCE in __init__.py against the INITIAL (pre-optimization) H_diag3, and
// is NEVER recomputed inside the macroiteration loop even though
// self.H_diag3 itself IS freshly rebuilt every macroiteration from the
// current rotated integrals (confirmed via exhaustive grep: `index_Hdiag
// =` appears at exactly 2 line numbers in the whole file, both inside
// __init__, never inside the macroiteration loop) -- i.e. index_Hdiag
// becomes a "stale" ordering relative to the current H_diag3 after the
// first macroiteration. This class reproduces that literally: it derives
// index_Hdiag once, from the INITIAL H_spatial2/J/K/E_core passed to its
// constructor, and CasscfCiStateAverageSolver reuses this same
// (increasingly stale) ordering on every subsequent call, exactly matching
// the Python.
class CasscfCiSetup {
public:
    // H_spatial2/J/K/E_core: the INITIAL (pre-macroiteration-loop) state --
    // the caller must construct this BEFORE any macroiteration begins,
    // mirroring __init__'s own ordering (this CI setup block runs before
    // the macroiteration `while` loop in the Python). Not read from
    // CasscfContext directly (context is mutated across macroiterations;
    // this class needs a fixed snapshot at construction time only).
    CasscfCiSetup(Dimensions dims, CasscfCiConfig config, CasscfPhysicalConstants constants,
                  const Matrix& H_spatial2, const Tensor4& J, const Tensor4& K, double E_core);

    int num_alpha() const { return num_alpha_; }
    int H_dim() const { return h_dim_; }
    int indim() const { return indim_; }
    int maxdim() const { return maxdim_; }

    const std::vector<int32_t>& table() const { return table_; }
    const std::vector<int32_t>& table_creation() const { return table_creation_; }
    const std::vector<int32_t>& table_annihilation() const { return table_annihilation_; }
    const std::vector<int32_t>& b_array() const { return b_array_; }
    const std::vector<int32_t>& Y() const { return Y_; }
    const Vector& S_diag() const { return S_diag_; }
    const Vector& S_diag_projection() const { return S_diag_projection_; }
    const std::vector<int32_t>& index_Hdiag() const { return index_Hdiag_; }

private:
    int num_alpha_ = 0;
    int h_dim_ = 0;
    int indim_ = 0;
    int maxdim_ = 0;

    std::vector<int32_t> table_;
    std::vector<int32_t> table_creation_;
    std::vector<int32_t> table_annihilation_;
    std::vector<int32_t> b_array_;
    std::vector<int32_t> Y_;
    Vector S_diag_;
    Vector S_diag_projection_;
    std::vector<int32_t> index_Hdiag_;
};

} // namespace casscf
