#pragma once

#include <Eigen/Dense>
#include <string>

namespace casscf {

using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;

// Orbital-space dimensions for one macroiteration. Mirrors the attributes
// set on `self` around helper_PFCI.py:2319-2371 (n_in_a, n_act_orb,
// n_virtual, nmo, n_occupied, index_map_size, ...).
struct Dimensions {
    int n_in_a = 0;     // inactive (doubly occupied) orbitals
    int n_act_orb = 0;  // active orbitals
    int n_virtual = 0;  // virtual orbitals
    int nmo = 0;        // total molecular orbitals
    int n_occupied = 0; // n_in_a + n_act_orb

    // Reduced (non-redundant) orbital-rotation parameter counts, matching
    // n_ai / n_vi / n_va / index_map_size at helper_PFCI.py:2320-2324.
    int n_ai() const { return n_in_a * n_act_orb; }
    int n_vi() const { return n_in_a * n_virtual; }
    int n_va() const { return n_act_orb * n_virtual; }
    int index_map_size() const { return n_ai() + n_vi() + n_va(); }
};

// Why this solver terminated, matching the return-string convention used in
// solve_pcg_trust_region (helper_PFCI.py:16148) and its GLTR counterpart.
enum class TerminationReason {
    SuccessGradientZero,   // gradient already ~0 at the trial point
    SuccessInteriorSolution, // unconstrained Newton step found inside the radius
    NegativeCurvature,     // stopped on a direction of non-positive curvature
    TrustBoundary,         // step crossed the trust-region boundary
    HardCase,              // hard case: gradient (near-)orthogonal to the
                            // lowest eigenspace of the (augmented) Hessian
    MaxIterations,          // ran out of iterations without converging
    NotImplemented
};

// Common result type for every trust-region subproblem solver
// (Davidson+LSTRS, GLTR, PCG/Steihaug). Lets the macroiteration/microiteration
// driver stay solver-agnostic, the way the Python driver just calls whichever
// of internal_optimization3 / microiteration_optimization6 is active without
// caring which inner solve_* function produced the step.
struct TrustRegionResult {
    Vector step;
    double predicted_decrease = 0.0;
    TerminationReason reason = TerminationReason::NotImplemented;
    int iterations = 0;

    // True if the solver had to fall back to hard-case handling (relevant to
    // Davidson-lstrs in particular; see internal_optimization3's
    // hard_case/reduce_step bookkeeping, helper_PFCI.py:6998-7099).
    bool hard_case = false;

    // Which root of the (augmented) Hessian eigenproblem the step was built
    // from. Quantum-chemistry AH/RFO codes without explicit hard-case
    // handling fall back to root_index = 1 (second-lowest root) when the
    // lowest root fails to give a usable constrained step.
    int root_index = 0;
};

} // namespace casscf
