// Cross-validates DavidsonDrivenLstrsSolver against LstrsSolver: given a
// guess-subspace cap that covers the whole problem (so Davidson's
// "iteration 1" reproduces the exact bordered eigenpairs at every trial
// alpha, no subspace-approximation error), both solvers solve the same
// trust-region subproblem via the same shared bisection core
// (solve_lstrs_bisection) and must agree, despite going through completely
// different eigenpair machinery (dense eigh vs. matrix-free Davidson).
#include "casscf/davidson_driven_lstrs_solver.hpp"
#include "casscf/hessian_operator.hpp"
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

class DenseGuessProviderWithGradient final : public HessianGuessProvider {
public:
    DenseGuessProviderWithGradient(Matrix hessian, Vector gradient)
        : hessian_(std::move(hessian)), gradient_(std::move(gradient)) {}

    void guess_block(const std::vector<int>& indices, Matrix& hessian_block, Vector& gradient_block) const override {
        const int m = static_cast<int>(indices.size());
        hessian_block = Matrix(m, m);
        gradient_block = Vector(m);
        for (int a = 0; a < m; ++a) {
            gradient_block(a) = gradient_(indices[a]);
            for (int b = 0; b < m; ++b)
                hessian_block(a, b) = hessian_(indices[a], indices[b]);
        }
    }

private:
    Matrix hessian_;
    Vector gradient_;
};

} // namespace

int main() {
    const double tol = 1e-4;

    // Indefinite Hessian (n_negative > 0), matching the regime where
    // production code actually picks the Davidson-driven path over GLTR
    // (select_trs_strategy).
    Matrix H = Matrix::Zero(4, 4);
    H.diagonal() << -1.0, 4.0, 2.0, 3.0;
    Vector g = (Vector(4) << 1.0, 1.0, 1.0, 1.0).finished();
    const double trust_radius = 1.0;

    auto dense_op = std::make_shared<DenseHessianOperator>(H);
    auto guess_provider = std::make_shared<DenseGuessProviderWithGradient>(H, g);

    // Force full guess-subspace coverage (see test_davidson_augmented_hessian_solver.cpp).
    DavidsonAugmentedHessianConfig davidson_config;
    davidson_config.large_problem_threshold = 0;
    davidson_config.large_problem_dim0 = 4;
    davidson_config.max_guess_dimension = 4;

    LstrsSolver lstrs(dense_op);
    DavidsonDrivenLstrsSolver davidson_lstrs(dense_op, guess_provider, H.diagonal(), {}, davidson_config);

    auto r_lstrs = lstrs.solve(g, trust_radius);
    auto r_davidson = davidson_lstrs.solve(g, trust_radius);

    // LSTRS's own convergence test is a *relative* 1e-3 tolerance on the
    // step norm (helper_PFCI.py:7269 / :11587-11588), not absolute -- same
    // characteristic hit in test_lstrs_solver.cpp's indefinite case.
    const double radius_tol = 3e-3 * trust_radius;
    expect_near(r_lstrs.step.norm(), trust_radius, radius_tol, "LSTRS reference: ||step|| == radius");
    expect_near(r_davidson.step.norm(), trust_radius, radius_tol, "Davidson-driven: ||step|| == radius");
    expect_near((r_lstrs.step - r_davidson.step).norm(), 0.0, tol, "Davidson-driven matches dense LSTRS reference");

    if (failures == 0) {
        std::printf("\nAll tests passed.\n");
        return 0;
    }
    std::printf("\n%d test(s) failed.\n", failures);
    return 1;
}
