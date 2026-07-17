// Cross-validates GltrTrustRegionSolver against LstrsSolver on the same
// canonical 2D problems used by the other solver tests. Both solvers are
// meant to reach the exact global trust-region solution (LSTRS via the
// bordered-matrix secular equation, GLTR via a Krylov subspace that, for a
// 2D problem, spans the whole space within 2 iterations) so they should
// agree closely -- unlike the PCG comparisons, which only bound PCG's
// truncated-CG result from above.
#include "casscf/hessian_operator.hpp"
#include "casscf/gltr_trust_region_solver.hpp"
#include "casscf/lstrs_solver.hpp"

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

GltrConfig deterministic_config() {
    GltrConfig config;
    config.add_noise = false; // determinism for testing; see header doc comment
    config.max_iter = 50;
    config.tol = 1e-8;
    return config;
}

} // namespace

int main() {
    const double tol = 1e-4;

    // --- Case 1: positive definite, interior Newton step ---
    {
        Matrix H = (Matrix(2, 2) << 2.0, 0.0, 0.0, 4.0).finished();
        Vector g = (Vector(2) << -2.0, -4.0).finished();
        auto op = std::make_shared<DenseHessianOperator>(H);
        GltrTrustRegionSolver gltr(op, Vector::Ones(2), deterministic_config());

        auto result = gltr.solve(g, 5.0);
        expect_near(result.step(0), 1.0, tol, "interior case: GLTR step[0] == Newton step");
        expect_near(result.step(1), 1.0, tol, "interior case: GLTR step[1] == Newton step");
        if (result.reason != TerminationReason::SuccessInteriorSolution) {
            std::printf("FAIL: interior case: wrong termination reason\n");
            ++failures;
        } else {
            std::printf("PASS: interior case: termination reason is SuccessInteriorSolution\n");
        }
    }

    // --- Case 2: positive definite, boundary step -- compare against LSTRS ---
    {
        Matrix H = (Matrix(2, 2) << 2.0, 0.0, 0.0, 4.0).finished();
        Vector g = (Vector(2) << -2.0, -4.0).finished();
        auto op = std::make_shared<DenseHessianOperator>(H);
        GltrTrustRegionSolver gltr(op, Vector::Ones(2), deterministic_config());
        LstrsSolver lstrs(op);

        const double trust_radius = 1.0;
        auto r_gltr = gltr.solve(g, trust_radius);
        auto r_lstrs = lstrs.solve(g, trust_radius);
        expect_near(r_gltr.step.norm(), trust_radius, tol, "boundary case: ||GLTR step|| == radius");
        expect_near((r_gltr.step - r_lstrs.step).norm(), 0.0, tol, "boundary case: GLTR matches LSTRS");
    }

    // --- Case 3: indefinite Hessian -- compare against LSTRS's global solution ---
    {
        Matrix H = (Matrix(2, 2) << -1.0, 0.0, 0.0, 4.0).finished();
        Vector g = (Vector(2) << 1.0, 1.0).finished();
        auto op = std::make_shared<DenseHessianOperator>(H);
        GltrTrustRegionSolver gltr(op, Vector::Ones(2), deterministic_config());
        LstrsSolver lstrs(op);

        const double trust_radius = 2.0;
        auto r_gltr = gltr.solve(g, trust_radius);
        auto r_lstrs = lstrs.solve(g, trust_radius);
        // GLTR's own secular-equation bisection is only accurate to
        // 1e-4 * radius by construction (helper_PFCI.py:15191:
        // `if abs(err) < 1e-4 * radius: break`), so the reconstructed step
        // carries that much inherent imprecision relative to LSTRS's exact
        // answer -- looser tolerance than the well-conditioned boundary case.
        expect_near(r_gltr.step.norm(), trust_radius, 5e-4 * trust_radius, "indefinite case: ||GLTR step|| == radius");
        expect_near((r_gltr.step - r_lstrs.step).norm(), 0.0, 5e-4 * trust_radius, "indefinite case: GLTR matches LSTRS");
    }

    // --- Case 4: zero gradient ---
    {
        Matrix H = (Matrix(2, 2) << 2.0, 0.0, 0.0, 4.0).finished();
        Vector g = Vector::Zero(2);
        auto op = std::make_shared<DenseHessianOperator>(H);
        GltrTrustRegionSolver gltr(op, Vector::Ones(2), deterministic_config());

        auto result = gltr.solve(g, 1.0);
        expect_near(result.step.norm(), 0.0, tol, "zero gradient case: step is zero");
        if (result.reason != TerminationReason::SuccessGradientZero) {
            std::printf("FAIL: zero gradient case: wrong termination reason\n");
            ++failures;
        } else {
            std::printf("PASS: zero gradient case: termination reason is SuccessGradientZero\n");
        }
    }

    if (failures == 0) {
        std::printf("\nAll tests passed.\n");
        return 0;
    }
    std::printf("\n%d test(s) failed.\n", failures);
    return 1;
}
