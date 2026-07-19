#pragma once

#include "casscf/tensor_types.hpp"
#include "casscf/types.hpp"

#include <vector>

namespace casscf {

// One entry of self.bfgs_history: (s, y, Bs, rho_y, rho_Bs) --
// helper_PFCI.py:11169-11171. y may be Powell-damped (see BfgsOperator::update).
struct BfgsHistoryEntry {
    Vector s;
    Vector y;
    Vector Bs;
    double rho_y = 0.0;
    double rho_Bs = 0.0;
};

// Faithful port of get_bfgs_mv (helper_PFCI.py:10857-10906) plus the
// damped-BFGS history update block inside microiteration_optimization6
// (helper_PFCI.py:11128-11176) and the reference-point reset
// (helper_PFCI.py:11180-11187, 12183-12187).
//
// Represents a running L-BFGS approximation B of the reduced orbital
// Hessian, built as a limited-memory recursive update (Kreplin Eq. 63)
// around a fixed "reference point" (U_zero/A_tilde_zero/G_zero -- the
// full-space intermediates at the geometry where the BFGS approximation
// was last reset) whose base term B_0*v is exactly orbital_sigma3 applied
// there.
//
// NOTE: the reference-point *reset* condition
// (helper_PFCI.py:11180-11187: `(density_norm_change > 0.025 and
// qn_optimization) or predicted_energy > 0 or qn_count == 1`) is NOT the
// same as solver_selector.hpp's should_reset_bfgs_reference() -- that
// function ports a similar-looking but different condition
// (helper_PFCI.py's branch-dispatch decision, "solve step for original
// hessian" vs. "solve bfgs for updated hessian") that additionally checks
// `consecutive_skips >= 3`. Both conditions are real and distinct in the
// Python; this class only implements reset_reference() as a mechanical
// action (clear history, adopt a new reference point) -- deciding *when*
// to call it is the caller's (CasscfMicroiterationOptimizationStep's)
// responsibility, using its own port of the 11180-11187 condition, not
// should_reset_bfgs_reference().
class BfgsOperator {
public:
    BfgsOperator(Matrix U_zero, Matrix A_tilde_zero, Tensor4 G_zero, Dimensions dims, int m_history = 10);

    // get_bfgs_mv(v), helper_PFCI.py:10857-10906: B_0*v (via orbital_sigma3
    // at the current reference point) plus the recursive limited-memory
    // correction from every history entry, oldest to newest (matches the
    // Python's `for (s, y, Bs, rho_y, rho_Bs) in history:` iteration order
    // -- history_ is stored oldest-first, same as self.bfgs_history).
    Vector apply(const Vector& v) const;

    // Damped-BFGS history update, helper_PFCI.py:11128-11176 (the
    // `if qn_count > 1:` block). s_vec: the step just taken. y_vec: the
    // reduced-gradient difference (new - old) after that step. Computes
    // Bs_vec = apply(s_vec) against the *current* (pre-update) history,
    // applies Powell damping if the curvature condition `y.s >= 0.1*s.Bs`
    // fails, then appends the new entry and pops the oldest if history
    // exceeds m_history.
    void update(const Vector& s_vec, const Vector& y_vec);

    // helper_PFCI.py:11180-11187 / 12183-12187: adopt a new reference
    // point and clear all history.
    void reset_reference(Matrix U_zero, Matrix A_tilde_zero, Tensor4 G_zero);

    const std::vector<BfgsHistoryEntry>& history() const { return history_; }
    const Matrix& U_zero() const { return U_zero_; }
    const Matrix& A_tilde_zero() const { return A_tilde_zero_; }
    const Tensor4& G_zero() const { return G_zero_; }

private:
    Matrix U_zero_;
    Matrix A_tilde_zero_;
    Tensor4 G_zero_;
    Dimensions dims_;
    int m_history_;
    std::vector<BfgsHistoryEntry> history_;
};

} // namespace casscf
