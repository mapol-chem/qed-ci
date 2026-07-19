// Tests for microiteration_energy.hpp (microiteration_exact_energy,
// microiteration_predicted_energy2 -- helper_PFCI.py:8800-8826, 9455-9467).
#include "casscf/microiteration_energy.hpp"
#include "casscf/orbital_sigma.hpp"

#include <cmath>
#include <cstdio>

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
    const double tol = 1e-9;

    // --- microiteration_exact_energy: hand-computed case, nmo=3,
    //     n_occupied=2. G chosen in separable form (G(K,L,R,S) = f(K,L),
    //     constant across R,S) so the double contraction reduces to
    //     something hand-summable: term2 = sum_{K,L} f(K,L) * colsum(K) *
    //     colsum(L), where colsum(K) = sum_R T(R,K). See this file's
    //     construction in cpp_casscf/README.md / commit message for the
    //     worked arithmetic (E = 13.31). ---
    {
        Dimensions dims;
        dims.nmo = 3;
        dims.n_occupied = 2;

        Matrix U(3, 3);
        U << 1.1, 0.2, 7.0,
             0.3, 1.4, 8.0,
             0.5, 0.6, 10.0;
        Matrix A(3, 3);
        A << 1.0, 0.5, 99.0,
             0.2, 0.3, 99.0,
             0.1, 0.4, 99.0;

        Tensor4 G(2, 2, 3, 3);
        for (int R = 0; R < 3; ++R) {
            for (int S = 0; S < 3; ++S) {
                G(0, 0, R, S) = 1.0;
                G(0, 1, R, S) = 2.0;
                G(1, 0, R, S) = 3.0;
                G(1, 1, R, S) = 4.0;
            }
        }

        const double result = microiteration_exact_energy(U, A, G, dims);
        expect_near(result, 13.31, 1e-9, "microiteration_exact_energy: hand-computed separable-G case");
    }

    // --- microiteration_predicted_energy2: thin wrapper around
    //     orbital_sigma3 (already independently cross-validated in
    //     test_orbital_sigma.cpp) -- checks the wiring (argument order,
    //     the 2*g.step + sigma.step combination) against the same formula
    //     computed inline. ---
    {
        Dimensions dims;
        dims.n_in_a = 1;
        dims.n_act_orb = 1;
        dims.n_virtual = 1;
        dims.nmo = 3;
        dims.n_occupied = 2;

        Matrix U(3, 3);
        U << 1.0, 0.1, -0.2,
             -0.1, 1.0, 0.3,
             0.2, -0.3, 1.0;
        Matrix A_tilde(3, 3);
        A_tilde << 0.5, 0.1, 0.2,
                   -0.1, 0.4, -0.2,
                   0.3, 0.1, 0.6;
        Tensor4 G(2, 2, 3, 3);
        double v = 0.05;
        for (int k = 0; k < 2; ++k)
            for (int l = 0; l < 2; ++l)
                for (int r = 0; r < 3; ++r)
                    for (int s = 0; s < 3; ++s) {
                        G(k, l, r, s) = v;
                        v += 0.01;
                    }

        const int n = dims.index_map_size();
        Vector reduced_gradient(n);
        Vector step(n);
        for (int i = 0; i < n; ++i) {
            reduced_gradient(i) = 0.1 * (i + 1);
            step(i) = 0.2 - 0.05 * i;
        }

        const double result = microiteration_predicted_energy2(U, reduced_gradient, A_tilde, G, step, dims);
        const Vector sigma_reduced = orbital_sigma3(U, A_tilde, G, step, dims);
        const double expected = 2.0 * reduced_gradient.dot(step) + sigma_reduced.dot(step);
        expect_near(result, expected, tol, "microiteration_predicted_energy2: matches inline 2*g.step + sigma.step");
    }

    if (failures == 0) {
        std::printf("\nAll tests passed.\n");
        return 0;
    }
    std::printf("\n%d test(s) failed.\n", failures);
    return 1;
}
