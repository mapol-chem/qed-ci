// Smoke tests for PcgTrustRegionSolver against hand-computable trust-region
// subproblems. Not a port of any specific Python test -- these are analytic
// sanity checks (interior Newton step, boundary step, negative curvature)
// meant to catch regressions while the rest of the solver suite is ported.
#include "casscf/hessian_operator.hpp"
#include "casscf/pcg_trust_region_solver.hpp"

#include <cmath>
#include <cstdio>
#include <memory>

using namespace casscf;

namespace {

int failures = 0;

void expect_near(double actual, double expected, double tol, const char* label) {
    if (std::abs(actual - expected) > tol) {
        std::printf("FAIL: %s -- expected %.10f, got %.10f\n", label, expected, actual);
        ++failures;
    } else {
        std::printf("PASS: %s (%.10f ~= %.10f)\n", label, actual, expected);
    }
}

double model_value(const Matrix& H, const Vector& g, const Vector& p) {
    return g.dot(p) + 0.5 * p.dot(H * p);
}

} // namespace

int main() {
    const double tol = 1e-6;

    // --- Case 1: positive definite, unconstrained minimum inside the radius ---
    {
        Matrix H = (Matrix(2, 2) << 2.0, 0.0, 0.0, 4.0).finished();
        Vector g = (Vector(2) << -2.0, -4.0).finished();
        auto op = std::make_shared<DenseHessianOperator>(H);
        PcgTrustRegionSolver solver(op, Vector::Ones(2), 1e-10, 100);

        auto result = solver.solve(g, /*trust_radius=*/5.0);
        expect_near(result.step(0), 1.0, tol, "interior case: step[0]");
        expect_near(result.step(1), 1.0, tol, "interior case: step[1]");
        if (result.reason != TerminationReason::SuccessInteriorSolution) {
            std::printf("FAIL: interior case: wrong termination reason\n");
            ++failures;
        }
    }

    // --- Case 2: positive definite, unconstrained minimum outside the radius ---
    {
        Matrix H = (Matrix(2, 2) << 2.0, 0.0, 0.0, 4.0).finished();
        Vector g = (Vector(2) << -2.0, -4.0).finished();
        auto op = std::make_shared<DenseHessianOperator>(H);
        PcgTrustRegionSolver solver(op, Vector::Ones(2), 1e-10, 100);

        const double trust_radius = 1.0;
        auto result = solver.solve(g, trust_radius);
        expect_near(result.step.norm(), trust_radius, tol, "boundary case: ||step|| == radius");
        if (result.reason != TerminationReason::TrustBoundary) {
            std::printf("FAIL: boundary case: wrong termination reason\n");
            ++failures;
        }
    }

    // --- Case 3: indefinite Hessian (negative curvature present) ---
    {
        Matrix H = (Matrix(2, 2) << -1.0, 0.0, 0.0, 4.0).finished();
        Vector g = (Vector(2) << 1.0, 1.0).finished();
        auto op = std::make_shared<DenseHessianOperator>(H);
        PcgTrustRegionSolver solver(op, Vector::Ones(2), 1e-10, 100);

        const double trust_radius = 2.0;
        auto result = solver.solve(g, trust_radius);
        expect_near(result.step.norm(), trust_radius, tol, "negative curvature: ||step|| == radius");
        const double m = model_value(H, g, result.step);
        if (m >= 0.0) {
            std::printf("FAIL: negative curvature: model value did not decrease (m=%.10f)\n", m);
            ++failures;
        } else {
            std::printf("PASS: negative curvature: model value decreased (m=%.10f)\n", m);
        }
    }

    if (failures == 0) {
        std::printf("\nAll tests passed.\n");
        return 0;
    }
    std::printf("\n%d test(s) failed.\n", failures);
    return 1;
}
