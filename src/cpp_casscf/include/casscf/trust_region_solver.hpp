#pragma once

#include "casscf/hessian_operator.hpp"
#include "casscf/types.hpp"

namespace casscf {

// Common interface for the trust-region subproblem solvers:
//   minimize   g^T p + 0.5 p^T H p
//   subject to ||p|| <= trust_radius
//
// Concrete solvers bind their own Hessian representation at construction
// time (dense for Davidson+LSTRS, matrix-free for GLTR/PCG) via
// HessianOperator, so the call site only ever deals with gradient + radius.
class TrustRegionSolver {
public:
    virtual ~TrustRegionSolver() = default;
    virtual TrustRegionResult solve(const Vector& gradient, double trust_radius) const = 0;
};

} // namespace casscf
