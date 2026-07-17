// Cross-validates LstrsSolver against PcgTrustRegionSolver: both solve the
// same trust-region subproblem, so on the well-posed (positive-definite)
// cases they must agree on the step; on the indefinite case LSTRS finds the
// exact global solution while Steihaug-CG only guarantees a not-worse
// boundary point, so we check LSTRS achieves an equal-or-better (lower)
// model value instead of an identical step.
#include "casscf/hessian_operator.hpp"
#include "casscf/lstrs_solver.hpp"
#include "casscf/pcg_trust_region_solver.hpp"

#include <cmath>
#include <cstdio>
#include <memory>

using namespace casscf;

namespace {

int failures = 0;

void expect_le(double actual, double bound, double tol, const char* label) {
    if (actual > bound + tol) {
        std::printf("FAIL: %s -- expected <= %.10f, got %.10f\n", label, bound, actual);
        ++failures;
    } else {
        std::printf("PASS: %s (%.10f <= %.10f)\n", label, actual, bound);
    }
}

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
    const double tol = 1e-5;

    // --- Case 1: positive definite, interior Newton step ---
    {
        Matrix H = (Matrix(2, 2) << 2.0, 0.0, 0.0, 4.0).finished();
        Vector g = (Vector(2) << -2.0, -4.0).finished();
        auto dense_op = std::make_shared<DenseHessianOperator>(H);
        LstrsSolver lstrs(dense_op);
        PcgTrustRegionSolver pcg(dense_op, Vector::Ones(2), 1e-10, 100);

        auto r_lstrs = lstrs.solve(g, 5.0);
        auto r_pcg = pcg.solve(g, 5.0);
        expect_near((r_lstrs.step - r_pcg.step).norm(), 0.0, tol, "interior case: LSTRS matches PCG");
        expect_near(r_lstrs.step(0), 1.0, tol, "interior case: LSTRS step[0] == Newton step");
    }

    // --- Case 2: positive definite, boundary step ---
    // NOTE: PCG/Steihaug crosses the trust-region boundary on its very
    // first CG step here (the raw -gradient direction overshoots the
    // radius immediately), so its output is a steepest-descent-to-boundary
    // point, not the exact constrained optimum -- unlike case 1, PCG and
    // LSTRS are *not* expected to agree here. LSTRS solves the bordered
    // secular equation exactly, so it must reach an equal-or-better model
    // value than PCG's single-step truncation, same as the indefinite case.
    {
        Matrix H = (Matrix(2, 2) << 2.0, 0.0, 0.0, 4.0).finished();
        Vector g = (Vector(2) << -2.0, -4.0).finished();
        auto dense_op = std::make_shared<DenseHessianOperator>(H);
        LstrsSolver lstrs(dense_op);
        PcgTrustRegionSolver pcg(dense_op, Vector::Ones(2), 1e-10, 100);

        const double trust_radius = 1.0;
        auto r_lstrs = lstrs.solve(g, trust_radius);
        auto r_pcg = pcg.solve(g, trust_radius);
        expect_near(r_lstrs.step.norm(), trust_radius, 3e-3, "boundary case: ||LSTRS step|| == radius");
        expect_le(model_value(H, g, r_lstrs.step), model_value(H, g, r_pcg.step), 1e-6,
                  "boundary case: LSTRS model value <= PCG model value (global optimality)");
    }

    // --- Case 3: indefinite Hessian -- LSTRS must be globally optimal ---
    {
        Matrix H = (Matrix(2, 2) << -1.0, 0.0, 0.0, 4.0).finished();
        Vector g = (Vector(2) << 1.0, 1.0).finished();
        auto dense_op = std::make_shared<DenseHessianOperator>(H);
        LstrsSolver lstrs(dense_op);
        PcgTrustRegionSolver pcg(dense_op, Vector::Ones(2), 1e-10, 100);

        const double trust_radius = 2.0;
        auto r_lstrs = lstrs.solve(g, trust_radius);
        auto r_pcg = pcg.solve(g, trust_radius);
        // LSTRS's own convergence test is a *relative* 1e-3 tolerance on the
        // step norm (helper_PFCI.py:7269: abs((step10_norm-trust_radius)/trust_radius) <= 1e-3),
        // not an absolute one -- match that here rather than the tighter
        // absolute `tol` used for the well-conditioned cases above.
        expect_near(r_lstrs.step.norm(), trust_radius, 3e-3 * trust_radius, "indefinite case: ||LSTRS step|| == radius");
        expect_le(model_value(H, g, r_lstrs.step), model_value(H, g, r_pcg.step), 1e-8,
                  "indefinite case: LSTRS model value <= PCG model value (global optimality)");

        // KKT stationarity: (H + lambda*I) step = -g for lambda = -mu_min(H_bordered-derived
        // multiplier). Cheaper equivalent check here: the step must satisfy
        // H*step + g is parallel to step (both are proportional to -lambda*step
        // at a boundary stationary point), i.e. the residual (H*step+g) x step
        // (2D cross product) is ~0.
        const Vector residual = H * r_lstrs.step + g;
        const double cross = residual(0) * r_lstrs.step(1) - residual(1) * r_lstrs.step(0);
        expect_near(cross, 0.0, 1e-4, "indefinite case: LSTRS step satisfies KKT parallelity");
    }

    if (failures == 0) {
        std::printf("\nAll tests passed.\n");
        return 0;
    }
    std::printf("\n%d test(s) failed.\n", failures);
    return 1;
}
