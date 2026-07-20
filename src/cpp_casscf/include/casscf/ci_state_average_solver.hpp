#pragma once

#include "casscf/ci_setup.hpp"
#include "casscf/internal_optimization_step.hpp" // CasscfPhysicalConstants
#include "casscf/macroiteration_driver.hpp"
#include "casscf/types.hpp"

namespace casscf {

// The real CiStateAverageSolver -- the CI diagonalization + weighted
// state-average energy + RDM build that runs at the top of each
// macroiteration after the first (helper_PFCI.py:2424-2553, inside
// `while macroiteration < 1000: if macroiteration > 0:`). Constructor-
// injected with the same shared CasscfContext/CasscfCiSetup instances the
// caller holds, matching the pattern CasscfInternalOptimizationStep already
// establishes (CasscfCiSetup's own doc comment explains why the graph/
// string tables it bundles live in a separate, setup-once object rather
// than on CasscfContext itself).
//
// Two distinct input modes, selected by solve()'s use_staged_inputs
// parameter (see CiStateAverageSolver's own doc comment for the full
// reasoning and the real Python line numbers backing each mode):
//  - use_staged_inputs=false (default): recomputes occupied_fock_core/
//    occupied_J/gkl2 fresh from context.H_spatial2/J/K every call --
//    matches MacroiterationDriver's own top-of-macroiteration solve
//    (helper_PFCI.py:2424-2488). Does NOT read CasscfContext's
//    occupied_fock_core/occupied_J/gkl2/occupied_d_cmo "CI-solver input
//    staging" fields in this mode.
//  - use_staged_inputs=true: reads context.gkl2/occupied_J/
//    occupied_fock_core/occupied_d_cmo/E_core2 directly instead of
//    recomputing -- matches InternalOptimizationStep's and
//    MicroiterationOptimizationStep's own inner-loop CI-solve call sites
//    (helper_PFCI.py:7748/12418), both of which read locally-staged
//    quantities reflecting the orbital rotation accumulated within that
//    same call, not self.H_spatial2/J/K. **Real bug this mode fixes**:
//    before it existed, both of those call sites used the false-mode
//    behavior unconditionally (the only mode that existed), so their own
//    inner-loop CI re-solves silently ignored the very rotation staging
//    those classes were computing and committing -- confirmed via a
//    direct trace comparison against real Python (context.D_tu_avg's norm
//    was bit-identical across successive MicroiterationOptimizationStep
//    outer passes despite a substantial accumulated rotation, where
//    Python's own gradient jumps from ~5e-5 to ~1e-2 at the same
//    transition -- see README.md's "End-to-end integration" section).
//
// Either mode reads context.H_spatial2/J/K/d_cmo (E_core or E_core2
// depending on mode -- NOT recomputed, see ci_setup.hpp's
// ActiveBlockIntermediates doc comment for why) and context.D_tu_avg/
// D_tuvw_avg/Dpe_tu_avg (needed nowhere in THIS class actually --
// build_state_average_rdms only WRITES state-average RDMs, via
// CiStateAverageResult's return value, matching MacroiterationDriver::run's
// existing convention of copying them from each CiStateAverageResult into
// context right after every solve() call).
class CasscfCiStateAverageSolver final : public CiStateAverageSolver {
public:
    CasscfCiStateAverageSolver(Dimensions dims, CasscfCiConfig config, CasscfPhysicalConstants constants,
                                 CasscfCiSetup& setup, CasscfContext& context);

    CiStateAverageResult solve(const Matrix& eigenvecs_guess, bool use_staged_inputs = false) override;

private:
    Dimensions dims_;
    CasscfCiConfig config_;
    CasscfPhysicalConstants constants_;
    CasscfCiSetup* setup_;
    CasscfContext* context_;
};

} // namespace casscf
