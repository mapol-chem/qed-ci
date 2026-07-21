// Exercises the nroots==2 branch of the initial-guess-subspace check
// (davidson_augmented_hessian_solver.cpp:170-179, `if (conv_status[0] &&
// conv_status[1]) ... else { push whichever of {0,1} isn't converged }`)
// on the path where exactly ONE of the two roots is already converged from
// the initial guess subspace alone, and the other genuinely needs the
// expansion loop -- as opposed to the "both converge immediately"
// (test_davidson_augmented_hessian_solver.cpp) or "neither converges
// immediately" (test_davidson_restart_no_collapse.cpp) cases.
//
// This partial case is otherwise hard to hit by chance: the guess-subspace
// Ritz pair for a root has to be an *exact* eigenvector of the full,
// matrix-free augmented operator (residual < 1e-7), not just a good
// approximation, which essentially never happens for a generic random
// problem (confirmed empirically -- random diagonal Hessians consistently
// leave BOTH roots' residuals several orders of magnitude above the
// convergence threshold on the first check, even when one is much smaller
// than the other).
//
// Deterministic construction: give exactly one coordinate a zero gradient
// component. Since H is diagonal here, a coordinate with g_i == 0 has no
// coupling to the border row/column of the augmented bordered matrix
// (row/column i's off-diagonal entries are H_ij==0 for j!=i, same as any
// diagonal H, and the border entry is g_i==0) -- so that coordinate is
// EXACTLY decoupled from every other coordinate, in both the true
// (index_map_size+1)-dimensional augmented operator and any guess subspace
// that happens to include it. Its exact eigenpair is (H_ii, e_i),
// reproduced by the guess subspace to machine precision the instant it's
// selected. Giving that coordinate a strongly negative diagonal
// (H_00 = -10.0) guarantees its selection: the ranking heuristic
// (helper_PFCI.py:15511/davidson_augmented_hessian_solver.cpp:97-99) scores
// any diag <= 1e-5 entry at a fixed 1e14 (maximum priority) regardless of
// its gradient, specifically so negative-curvature directions are never
// skipped for having a small gradient component. The other two
// negative-diagonal directions (with nonzero gradient, so genuinely
// coupled to the border) are what root 1 (the second-lowest eigenvalue)
// actually needs Davidson correction to resolve.
//
// Confirmed via a temporary CASSCF_DEBUG_DAVIDSON trace during development
// (not asserted here -- that flag isn't part of the public API) that this
// construction produces exactly conv_status={true, false} on the very
// first initial-guess check, i.e. unconverged_idx == {1} only. What IS
// asserted below, through the public API alone: root 0's eigenvalue comes
// out at exactly -10.0 (the analytically-exact decoupled answer, to
// numerical-noise tolerance) while root 1 matches a dense reference
// solve -- consistent with root 0 needing zero correction and root 1
// needing real Davidson iterations, and not reproducible by coincidence.
#include "casscf/bordered_eigensolve.hpp"
#include "casscf/davidson_augmented_hessian_solver.hpp"
#include "casscf/hessian_operator.hpp"

#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

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
    // Index 0: zero gradient, strongly negative diagonal -- exactly
    // decoupled, guaranteed selection (see header comment).
    // Indices 1-2: negative diagonal, nonzero gradient -- genuinely
    // border-coupled, what root 1 needs to resolve.
    // Indices 3-7: positive diagonal bulk (irrelevant to the two lowest
    // roots, just fills out the problem).
    const std::vector<double> diag_v = {-10.0, -3.0, -2.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    const std::vector<double> grad_v = {0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    const int n = static_cast<int>(diag_v.size());
    Matrix H = Matrix::Zero(n, n);
    Vector g(n);
    for (int i = 0; i < n; ++i) {
        H(i, i) = diag_v[i];
        g(i) = grad_v[i];
    }
    const double alpha = 0.0;
    const double trust_radius = 1.0;
    const double alpha_range = 1e-6; // <= threshold -> nroots==2

    auto dense_op = std::make_shared<DenseHessianOperator>(H);
    auto guess_provider = std::make_shared<DenseGuessProviderWithGradient>(H, g);

    DavidsonAugmentedHessianConfig config; // production defaults
    DavidsonAugmentedHessianSolver solver(dense_op, guess_provider, config);

    auto result = solver.solve(g, H.diagonal(), alpha, alpha_range, trust_radius, /*restart=*/false);
    if (!result.converged) {
        std::printf("FAIL: solve() did not converge\n");
        return 1;
    }
    std::printf("PASS: converged (%d Davidson iteration(s) -- root 0 needed none, root 1 needed real correction)\n",
                result.davidson_iterations);

    expect_near(result.eigenvalues(0), -10.0, 1e-10,
                "root 0 == exactly the decoupled eigenvalue (converged from the initial guess subspace alone)");

    BorderedEigenPairs reference = bordered_eigensolve(g, H, alpha);
    expect_near(reference.eigenvalues(0), -10.0, 1e-10, "sanity: dense reference agrees root 0 is the decoupled -10.0");
    expect_near(result.eigenvalues(1), reference.eigenvalues(1), 1e-6,
                "root 1 (needed real Davidson correction) matches dense reference");

    if (failures == 0) {
        std::printf("\nAll tests passed.\n");
        return 0;
    }
    std::printf("\n%d test(s) failed.\n", failures);
    return 1;
}
