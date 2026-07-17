// Exercises the Davidson subspace *expansion* loop (helper_PFCI.py:15608-16084),
// not just the "iteration 1" fast path the other Davidson tests hit. Uses a
// guess-subspace cap far smaller than the problem dimension, so the initial
// guess can't possibly contain a converged root and the solver must grow
// the subspace with preconditioned correction vectors across several
// Davidson iterations (and, across several LSTRS bisection steps, resume
// via restart=True). Cross-validated against LstrsSolver's dense/exact
// answer on the same problem.
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
    const int n = 15;

    Matrix H = Matrix::Zero(n, n);
    H.diagonal() << -2.0, -1.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0;
    Vector g = Vector::Ones(n);
    const double trust_radius = 1.0;

    auto dense_op = std::make_shared<DenseHessianOperator>(H);
    auto guess_provider = std::make_shared<DenseGuessProviderWithGradient>(H, g);

    // Force a guess subspace far smaller than the problem (dim1 = 2 while
    // n = 15), so the initial guess cannot contain a converged root and the
    // expansion loop must actually run.
    DavidsonAugmentedHessianConfig davidson_config;
    davidson_config.large_problem_threshold = 0;
    davidson_config.large_problem_dim0 = 2;
    davidson_config.max_guess_dimension = 2;

    LstrsSolver lstrs(dense_op);
    DavidsonDrivenLstrsSolver davidson_lstrs(dense_op, guess_provider, H.diagonal(), {}, davidson_config);

    auto r_lstrs = lstrs.solve(g, trust_radius);
    auto r_davidson = davidson_lstrs.solve(g, trust_radius);

    const double radius_tol = 3e-3 * trust_radius; // LSTRS's own relative convergence criterion
    expect_near(r_lstrs.step.norm(), trust_radius, radius_tol, "LSTRS reference: ||step|| == radius");
    expect_near(r_davidson.step.norm(), trust_radius, radius_tol, "Davidson-driven (expansion loop): ||step|| == radius");
    expect_near((r_lstrs.step - r_davidson.step).norm(), 0.0, tol,
                "Davidson-driven (expansion loop) matches dense LSTRS reference");

    if (failures == 0) {
        std::printf("\nAll tests passed.\n");
        return 0;
    }
    std::printf("\n%d test(s) failed.\n", failures);
    return 1;
}
