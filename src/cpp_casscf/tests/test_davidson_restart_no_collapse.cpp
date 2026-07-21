// Exercises DavidsonAugmentedHessianSolver::solve()'s restart=true resume
// path (helper_PFCI.py:15551-15607) when the persisted subspace has NEVER
// hit its size cap -- i.e. `collapse_subspace_check_` is still false on
// entry (davidson_augmented_hessian_solver.cpp:225's `if
// (!collapse_subspace_check_)` branch, both in the resume setup and in
// run_expansion_loop's own dispatch at line ~301).
//
// This combination was previously untested: test_davidson_augmented_hessian_solver.cpp
// only calls solve() once (restart=false, converges before the expansion
// loop even runs); test_davidson_expansion_loop.cpp does drive
// restart=true resumes, but with an artificially tiny guess-subspace cap
// (max_guess_dimension=2 on a 15-dim problem) that forces the subspace to
// hit its cap almost immediately -- every restart=true call there resumes
// an ALREADY-collapsed subspace instead.
//
// Checked against real captured chemistry (see cpp_casscf/README.md,
// "digging into collapse_subspace/restart test coverage"): grepping
// `collapse subspace check` in Python's own stdout across every real
// dumped config we have (LiH/6-31G and LiH/6-311G (4,4)/(4,8) active
// spaces, and a harder stretched, 2-photon H2O/6-31G case, H_dim up to 97)
// shows `collapse_subspace_check` staying False for every single call --
// this port's default `DavidsonAugmentedHessianConfig` (matching Python's
// own hardcoded dim0/dim1/maxdim formulas, helper_PFCI.py:15530-15548)
// gives generous enough subspace-growth headroom relative to `dim1` that
// collapsing essentially never happens at these problem sizes, while
// restart=true IS the dominant call pattern (8/11 solve() calls in the LiH
// cases). So this test's problem is deliberately sized to be representative
// of that regime (dim1 = n/2, the *unmodified* production default), not an
// artificial worst case -- unlike test_davidson_expansion_loop.cpp's tiny
// forced cap.
//
// Strategy: a 20-dimensional diagonal Hessian, 3 negative-curvature
// directions, uniform gradient -- big enough that the initial guess
// subspace (dim1=10) can't converge on iteration 1 (so the expansion loop
// actually runs), small enough to solve exactly via a direct dense
// bordered_eigensolve() reference at each trial alpha. Three solve() calls
// on one shared solver instance mimic a real outer bisection sequence
// (restart=false establishing the persisted subspace, then two more trial
// alphas via restart=true) -- matching how DavidsonDrivenLstrsSolver
// actually drives this class in production.
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

void expect_true(bool cond, const char* label) {
    if (!cond) {
        std::printf("FAIL: %s\n", label);
        ++failures;
    } else {
        std::printf("PASS: %s\n", label);
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

// Solves at `alpha` via the shared solver instance, cross-checks against a
// fresh dense bordered_eigensolve() reference at that same alpha, and
// returns the result for further inspection.
DavidsonIterationResult check_call(DavidsonAugmentedHessianSolver& solver, const Vector& g, const Matrix& H,
                                    double alpha, double alpha_range, double trust_radius, bool restart,
                                    const char* label) {
    auto result = solver.solve(g, H.diagonal(), alpha, alpha_range, trust_radius, restart);
    if (!result.converged) {
        std::printf("FAIL: %s -- solve() did not converge\n", label);
        ++failures;
        return result;
    }
    BorderedEigenPairs reference = bordered_eigensolve(g, H, alpha);
    char buf0[128], buf1[128];
    std::snprintf(buf0, sizeof(buf0), "%s: root 0 eigenvalue matches dense reference", label);
    std::snprintf(buf1, sizeof(buf1), "%s: root 1 eigenvalue matches dense reference", label);
    expect_near(result.eigenvalues(0), reference.eigenvalues(0), 1e-6, buf0);
    expect_near(result.eigenvalues(1), reference.eigenvalues(1), 1e-6, buf1);
    return result;
}

} // namespace

int main() {
    const std::vector<double> diag_v = {-2.0, -1.5, -1.0, 3,  4,  5,  6,  7,  8,  9,
                                         10,   11,   12,   13, 14, 15, 16, 17, 18, 19};
    const int n = static_cast<int>(diag_v.size());
    Matrix H = Matrix::Zero(n, n);
    for (int i = 0; i < n; ++i) H(i, i) = diag_v[i];
    Vector g = Vector::Ones(n);
    const double trust_radius = 1.0;
    const double alpha_range = 1e-6; // <= threshold -> nroots==2 (both roots tracked)

    auto dense_op = std::make_shared<DenseHessianOperator>(H);
    auto guess_provider = std::make_shared<DenseGuessProviderWithGradient>(H, g);

    DavidsonAugmentedHessianConfig config; // production defaults, no artificial caps
    DavidsonAugmentedHessianSolver solver(dense_op, guess_provider, config);

    // Call 1: restart=false, establishes the persisted subspace. Must NOT
    // converge in the initial guess subspace alone (dim1=n/2=10 of 20
    // directions) -- confirmed empirically to need 2 Davidson iterations,
    // i.e. this problem genuinely exercises run_expansion_loop, not just
    // the "iteration 1" fast path test_davidson_augmented_hessian_solver.cpp
    // already covers.
    auto r1 = check_call(solver, g, H, /*alpha=*/0.0, alpha_range, trust_radius, /*restart=*/false, "call 1 (restart=false)");
    expect_true(r1.davidson_iterations >= 2, "call 1: expansion loop actually ran (>=2 Davidson iterations)");
    expect_near(static_cast<double>(r1.structure.dim1), 10.0, 0.0,
                "call 1: structure.dim1 == n/2 (production dim1 formula, and confirms the "
                "structure-propagation-through-run_expansion_loop fix)");
    expect_near(static_cast<double>(r1.structure.n_negative), 3.0, 0.0,
                "call 1: structure.n_negative matches the 3 negative-diagonal directions");

    // Calls 2/3: restart=true, resuming the SAME never-collapsed subspace
    // at new trial alphas -- the previously-untested combination. Both
    // converge in just 1 further Davidson iteration each (verified
    // empirically), confirming the resumed subspace is being reused
    // correctly, not silently rebuilt or corrupted.
    auto r2 = check_call(solver, g, H, /*alpha=*/-0.4, alpha_range, trust_radius, /*restart=*/true, "call 2 (restart=true)");
    expect_true(r2.davidson_iterations >= 1, "call 2: resumed subspace still required at least 1 more iteration");

    auto r3 = check_call(solver, g, H, /*alpha=*/-0.7, alpha_range, trust_radius, /*restart=*/true, "call 3 (restart=true)");
    expect_true(r3.davidson_iterations >= 1, "call 3: resumed subspace still required at least 1 more iteration");

    if (failures == 0) {
        std::printf("\nAll tests passed.\n");
        return 0;
    }
    std::printf("\n%d test(s) failed.\n", failures);
    return 1;
}
