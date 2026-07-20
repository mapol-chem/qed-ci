// Tests for LinearRMSolver (residual_minimization.py port) and
// linear_equation_solve (the Python's own driver loop around it,
// helper_PFCI.py:16342-16399). Two cases:
// 1. LinearRMSolver in isolation, driven directly against a small,
//    hand-constructed SPD matrix (not orbital_sigma3) -- checks the
//    solver's own subspace-extrapolation algorithm converges to the
//    correct answer, independent of anything orbital-space-specific.
// 2. The full linear_equation_solve wiring (real orbital_sigma3 as the
//    Hessian-vector product) on a small, well-conditioned orbital-space
//    problem -- checked via residual self-consistency
//    (orbital_sigma3(solution) + reduced_gradient ~= 0), the natural way
//    to validate a linear solver against an operator that's already
//    independently validated elsewhere (test_orbital_sigma.cpp) rather
//    than needing to know the solution in closed form.
#include "casscf/linear_equation_solve.hpp"
#include "casscf/linear_rm_solver.hpp"
#include "casscf/orbital_sigma.hpp"

#include <cmath>
#include <cstdio>
#include <random>

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

} // namespace

int main() {
    // --- Case 1: LinearRMSolver in isolation against a small, hand-built
    //     SPD system Hx = -b, diagonal-preconditioned exactly like
    //     linear_equation_solve's own driver loop. ---
    {
        Matrix H(4, 4);
        H << 4.0, 1.0, 0.5, 0.0, 1.0, 3.0, 0.2, 0.1, 0.5, 0.2, 5.0, 0.3, 0.0, 0.1, 0.3, 2.0;
        Vector b(4);
        b << 1.0, -2.0, 0.5, 3.0;
        Vector denom = H.diagonal();

        const Vector expected_x = H.ldlt().solve(-b);

        LinearRMSolver solver(b, /*max_subspace=*/10);
        Vector residual = b;
        const int max_iter = 30;
        bool converged = false;
        for (int i = 0; i < max_iter && !converged; ++i) {
            if (residual.norm() < 1e-10) {
                converged = true;
                break;
            }
            Vector trial_c = residual.cwiseQuotient(denom);
            const double norm = trial_c.norm();
            if (norm > 1e-12) trial_c /= norm;
            Vector sigma = H * trial_c;
            residual = solver.update_subspace_and_extrapolate(trial_c, sigma);
        }

        Vector x = solver.get_solution();
        expect_near((x - expected_x).norm(), 0.0, 1e-6, "case1: LinearRMSolver matches direct SPD solve");
        expect_near((H * x + b).norm(), 0.0, 1e-6, "case1: LinearRMSolver's answer satisfies Hx + b ~= 0");
    }

    // --- Case 2: full linear_equation_solve, real orbital_sigma3, checked
    //     via residual self-consistency. ---
    {
        Dimensions dims;
        dims.n_in_a = 1;
        dims.n_act_orb = 1;
        dims.n_virtual = 1;
        dims.nmo = 3;
        dims.n_occupied = 2;

        std::mt19937 rng(2024u);
        std::uniform_real_distribution<double> dist(-0.3, 0.3);

        Matrix U = Matrix::Identity(dims.nmo, dims.nmo);
        for (int i = 0; i < dims.nmo; ++i)
            for (int j = 0; j < dims.nmo; ++j)
                if (i != j) U(i, j) = 0.05 * dist(rng); // near-identity, well-conditioned

        Matrix A_tilde(dims.nmo, dims.nmo);
        for (int i = 0; i < dims.nmo; ++i)
            for (int j = 0; j < dims.nmo; ++j) A_tilde(i, j) = dist(rng);

        // Physically-symmetric-style G (same construction already used in
        // test_integral_transformer.cpp) -- a generic, well-conditioned,
        // non-degenerate operator rather than a fully-arbitrary random
        // tensor, so the linear solve has a reasonable chance of
        // converging within a small max_iter for this test.
        Matrix S(dims.nmo, dims.nmo);
        S << 1.0, 0.2, 0.3, 0.2, 0.9, 0.4, 0.3, 0.4, 1.1;
        Tensor4 G(dims.n_occupied, dims.n_occupied, dims.nmo, dims.nmo);
        for (int k = 0; k < dims.n_occupied; ++k)
            for (int l = 0; l < dims.n_occupied; ++l)
                for (int p = 0; p < dims.nmo; ++p)
                    for (int q = 0; q < dims.nmo; ++q) G(k, l, p, q) = S(k, p) * S(l, q);

        Vector reduced_gradient = Vector::Constant(dims.index_map_size(), 1, 0.05);
        Vector denom = Vector::Constant(dims.index_map_size(), 1, 2.0);

        LinearEquationSolveResult result =
            linear_equation_solve(U, A_tilde, G, reduced_gradient, denom, /*max_iter=*/20, /*conv_thresh=*/1e-8, dims);

        Vector residual = orbital_sigma3(U, A_tilde, G, result.solution, dims) + reduced_gradient;
        expect_near(result.converged, 1.0, 0, "case2: linear_equation_solve reports converged");
        expect_near(residual.norm(), 0.0, 1e-6,
                    "case2: solution satisfies orbital_sigma3(x) + reduced_gradient ~= 0");
    }

    if (failures == 0) {
        std::printf("\nAll tests passed.\n");
        return 0;
    }
    std::printf("\n%d test(s) failed.\n", failures);
    return 1;
}
