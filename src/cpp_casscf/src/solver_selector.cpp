#include "casscf/solver_selector.hpp"

namespace casscf {

TrsStrategy select_trs_strategy(const Vector& reduced_hessian_diagonal) {
    const int n_negative = static_cast<int>((reduced_hessian_diagonal.array() < 0.0).count());
    return n_negative > 0 ? TrsStrategy::DavidsonSubspaceBisection : TrsStrategy::Gltr;
}

bool should_reset_bfgs_reference(const BfgsReferenceState& state, bool qn_optimization) {
    return (state.density_norm_change > 0.025 && qn_optimization) ||
           state.predicted_energy > 0.0 ||
           state.qn_count == 1 ||
           state.consecutive_skips >= 3;
}

bool should_reset_bfgs_reference_point(const BfgsReferenceState& state, bool qn_optimization) {
    return (state.density_norm_change > 0.025 && qn_optimization) ||
           state.predicted_energy > 0.0 ||
           state.qn_count == 1;
}

} // namespace casscf
