// Smoke test for DavidsonAugmentedHessianSolver::solve_iteration_one.
//
// Strategy: pick a guess-subspace cap >= the full problem dimension, so the
// Davidson guess subspace covers the whole space and the matrix-free
// residual check is exactly zero -- i.e. "iteration 1" must converge, and
// its result must exactly match a direct bordered_eigensolve() on the full
// problem. This exercises the full pipeline (index ranking, guess-block
// extraction, subspace bordered eigensolve, full-space embedding,
// matrix-free residual via BorderedHessianOperator, and the nroots==1/2
// convergence branches) against a reference that doesn't go through any of
// that machinery.
#include "casscf/bordered_eigensolve.hpp"
#include "casscf/davidson_augmented_hessian_solver.hpp"
#include "casscf/hessian_operator.hpp"

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

// Extracts dense sub-blocks directly from a fully-materialized reference
// Hessian/gradient -- standing in for the not-yet-ported intermediates-based
// build_orbital_hessian_guess.
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
    const double tol = 1e-6;

    Matrix H = Matrix::Zero(4, 4);
    H.diagonal() << 1.0, 2.0, 3.0, 4.0;
    Vector g = (Vector(4) << 1.0, 1.0, 1.0, 1.0).finished();
    const double alpha = 0.0;
    const double trust_radius = 0.5;

    auto dense_op = std::make_shared<DenseHessianOperator>(H);
    auto guess_provider = std::make_shared<DenseGuessProviderWithGradient>(H, g);

    DavidsonAugmentedHessianConfig config;
    // Force dim1 = 4 (== problem size) so the guess subspace covers
    // everything: dim1 = min(max(count, dim0), max_guess_dimension,
    // index_map_size). With this Hessian's diagonal all positive, count = 0,
    // so dim0 alone must reach 4 -- push the "large problem" branch's dim0
    // (normally 200) down to 4 rather than relying on the small-problem
    // branch's index_map_size/2 (which would give only 2).
    config.large_problem_threshold = 0;
    config.large_problem_dim0 = 4;
    config.max_guess_dimension = 4;

    DavidsonAugmentedHessianSolver solver(dense_op, guess_provider, config);

    // alpha_range <= 1e-5 selects the nroots==2 code path (helper_PFCI.py:15308-15313).
    const double alpha_range = 1e-6;
    auto result = solver.solve(g, H.diagonal(), alpha, alpha_range, trust_radius, /*restart=*/false);

    if (!result.converged) {
        std::printf("FAIL: solve() did not converge with a full-coverage guess subspace\n");
        return 1;
    }
    std::printf("PASS: converged with full-coverage guess subspace\n");

    expect_near(static_cast<double>(result.structure.dim1), 4.0, 0.0, "dim1 capped at full problem size");
    expect_near(result.structure.n_negative, 0.0, 0.0, "n_negative matches diagonal (none negative)");

    BorderedEigenPairs reference = bordered_eigensolve(g, H, alpha);
    expect_near(result.eigenvalues(0), reference.eigenvalues(0), tol, "lowest eigenvalue matches direct reference");
    expect_near(result.eigenvalues(1), reference.eigenvalues(1), tol, "second eigenvalue matches direct reference");

    // Eigenvectors are unique up to sign for non-degenerate eigenvalues;
    // check |cos(angle)| ~= 1 against the reference instead of exact equality.
    for (int col = 0; col < 2; ++col) {
        const double cos_angle = std::abs(result.eigenvectors.col(col).normalized().dot(
            reference.eigenvectors.col(col).normalized()));
        expect_near(cos_angle, 1.0, tol, col == 0 ? "root 0 eigenvector matches reference direction"
                                                   : "root 1 eigenvector matches reference direction");
    }

    if (failures == 0) {
        std::printf("\nAll tests passed.\n");
        return 0;
    }
    std::printf("\n%d test(s) failed.\n", failures);
    return 1;
}
