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
// Does NOT read or write CasscfContext's occupied_fock_core/occupied_J/
// gkl2/occupied_d_cmo "CI-solver input staging" fields (the ones
// CasscfInternalOptimizationStep/CasscfMicroiterationOptimizationStep
// commit before calling their OWN CiStateAverageSolver::solve()) --
// confirmed by direct reading of the Python: this specific block
// (helper_PFCI.py:2424-2488) recomputes its own local occupied_fock_core/
// occupied_J/gkl2/E_core-independent quantities fresh from
// self.H_spatial2/self.J/self.K every call, never reading or writing the
// self.occupied_*/self.gkl2 attributes those OTHER two call sites use. This
// class mirrors that: it reads context.H_spatial2/J/K/d_cmo/E_core (E_core
// read-only, NOT recomputed -- see ci_setup.hpp's ActiveBlockIntermediates
// doc comment for why) and context.D_tu_avg/D_tuvw_avg/Dpe_tu_avg (needed
// nowhere in THIS class actually -- build_state_average_rdms only WRITES
// state-average RDMs, via CiStateAverageResult's return value, matching
// MacroiterationDriver::run's existing convention of copying them from
// each CiStateAverageResult into context right after every solve() call).
class CasscfCiStateAverageSolver final : public CiStateAverageSolver {
public:
    CasscfCiStateAverageSolver(Dimensions dims, CasscfCiConfig config, CasscfPhysicalConstants constants,
                                 CasscfCiSetup& setup, CasscfContext& context);

    CiStateAverageResult solve(const Matrix& eigenvecs_guess) override;

private:
    Dimensions dims_;
    CasscfCiConfig config_;
    CasscfPhysicalConstants constants_;
    CasscfCiSetup* setup_;
    CasscfContext* context_;
};

} // namespace casscf
