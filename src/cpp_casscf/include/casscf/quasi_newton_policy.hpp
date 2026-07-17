#pragma once

namespace casscf {

// Governs whether microiteration_optimization6 switches from solving
// against the exact (freshly-rebuilt) reduced Hessian to a running L-BFGS
// approximation of it (get_bfgs_mv, helper_PFCI.py:10792-10841).
//
// In the Python, qn_optimization starts False and is switched on
// permanently (helper_PFCI.py:10417-10423, 11988-11994):
//     if (energy_change < 0.0 or hard_case == 2) and step_norm < 0.05:
//         qn_optimization = True
// with no corresponding switch-back-off path in the active code (the
// `else: qn_optimization = False` is commented out at both call sites).
// So in practice it behaves as "off until a small, energy-lowering step is
// taken nearby convergence, then permanently on" for the rest of that
// macroiteration's microiterations.
//
// Per the developer: quasi-Newton doesn't reliably accelerate MCSCF
// convergence, so this needs to be a user-facing choice rather than a purely
// automatic one. `enabled` is that master switch (default true, matching
// the Python's de-facto behavior of leaving it on once triggered) -- when
// false, should_activate() always returns false regardless of step_norm,
// i.e. the exact-Hessian GLTR path is used for the whole run.
struct QuasiNewtonPolicy {
    bool enabled = true;
    double activation_step_norm_threshold = 0.05; // helper_PFCI.py:10417, 11988
};

// Port of the activation condition at helper_PFCI.py:10417-10423 /
// 11988-11994, gated by policy.enabled.
bool should_activate_qn(const QuasiNewtonPolicy& policy, double energy_change, int hard_case, double step_norm);

} // namespace casscf
