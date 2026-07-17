#include "casscf/quasi_newton_policy.hpp"

namespace casscf {

bool should_activate_qn(const QuasiNewtonPolicy& policy, double energy_change, int hard_case, double step_norm) {
    if (!policy.enabled) return false;
    const bool trigger_condition = (energy_change < 0.0 || hard_case == 2);
    return trigger_condition && step_norm < policy.activation_step_norm_threshold;
}

} // namespace casscf
