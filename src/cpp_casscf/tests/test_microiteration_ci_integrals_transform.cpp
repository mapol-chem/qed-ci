// Tests for microiteration_ci_integrals_transform (helper_PFCI.py:8689-8798).
#include "casscf/microiteration_ci_integrals_transform.hpp"

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

void expect_matrix_near(const Matrix& actual, const Matrix& expected, double tol, const char* label) {
    expect_near((actual - expected).norm(), 0.0, tol, label);
}

} // namespace

int main() {
    const double tol = 1e-9;

    Dimensions dims;
    dims.n_in_a = 1;
    dims.n_act_orb = 1;
    dims.n_virtual = 0;
    dims.nmo = 2;
    dims.n_occupied = 2;

    Matrix fock_core(2, 2);
    fock_core << 2.0, 0.5,
                 0.5, 3.0;

    // L(i,j,r,s) with i<n_occupied(=2), j<n_in_a(=1), r,s<nmo(=2).
    Tensor4 L(2, 1, 2, 2);
    L(0, 0, 0, 0) = 0.1;
    L(0, 0, 0, 1) = 0.2;
    L(0, 0, 1, 0) = 0.3;
    L(0, 0, 1, 1) = 0.4;
    L(1, 0, 0, 0) = 0.05;
    L(1, 0, 0, 1) = 0.06;
    L(1, 0, 1, 0) = 0.07;
    L(1, 0, 1, 1) = 0.08;

    const double E_core_ref = 5.0;

    // J/K built with the real two-electron-integral pair-swap symmetry
    // (J(p,q,r,s) = J(r,s,p,q)) via a symmetric-outer-product construction,
    // so the U == identity case below is exactly a no-op (see this file's
    // own derivation in the commit message / README for why that symmetry
    // is what makes the two active_twoeint correction terms cancel
    // exactly).
    Matrix Sm(2, 2);
    Sm << 1.0, 0.3,
          0.3, 1.2;
    Matrix Tm(2, 2);
    Tm << 0.6, 0.15,
          0.15, 0.9;
    Tensor4 J(2, 2, 2, 2), K(2, 2, 2, 2);
    for (int p = 0; p < 2; ++p)
        for (int q = 0; q < 2; ++q)
            for (int r = 0; r < 2; ++r)
                for (int s = 0; s < 2; ++s) {
                    J(p, q, r, s) = Sm(p, q) * Sm(r, s);
                    K(p, q, r, s) = Tm(p, q) * Tm(r, s);
                }

    Tensor4 active_twoeint_ref(1, 1, 1, 1);
    active_twoeint_ref(0, 0, 0, 0) = J(1, 1, 1, 1);

    Matrix d_cmo_ref(2, 2);
    d_cmo_ref << 0.7, 0.1,
                 0.1, 0.4;

    // --- Case 1: U == identity is a no-op on every output (E_core2 ==
    //     E_core_ref, active_fock_core == fock_core's active-active block,
    //     active_twoeint == active_twoeint_ref, d_cmo == d_cmo_ref). ---
    {
        Matrix U = Matrix::Identity(2, 2);
        MicroiterationCiIntegralsResult r =
            microiteration_ci_integrals_transform(U, E_core_ref, fock_core, L, J, K, active_twoeint_ref, d_cmo_ref, dims);

        expect_near(r.E_core2, E_core_ref, tol, "U==identity: E_core2 == E_core_ref (no-op)");
        Matrix expected_active_fock_core(1, 1);
        expected_active_fock_core << fock_core(1, 1);
        expect_matrix_near(r.active_fock_core, expected_active_fock_core, tol,
                            "U==identity: active_fock_core == fock_core's active-active block");
        expect_near(r.active_twoeint(0, 0, 0, 0), active_twoeint_ref(0, 0, 0, 0), tol,
                    "U==identity: active_twoeint == active_twoeint_ref (no-op, uses J's pair-swap symmetry)");
        expect_matrix_near(r.d_cmo, d_cmo_ref, tol, "U==identity: d_cmo == d_cmo_ref (no-op)");
    }

    // --- Case 2: nontrivial U, hand-computed E_core2 only (active_fock_core/
    //     active_twoeint's full formulas are too intricate to hand-verify
    //     independently here; see file header -- Case 1 above already
    //     exercises their code paths structurally). ---
    {
        Matrix T(2, 2);
        T << 0.1, 0.2,
             0.05, 0.15;
        Matrix U = T + Matrix::Identity(2, 2);

        MicroiterationCiIntegralsResult r =
            microiteration_ci_integrals_transform(U, E_core_ref, fock_core, L, J, K, active_twoeint_ref, d_cmo_ref, dims);

        // Hand-computed: E_core_ref + 4*sum_{r} fock_core(0,r)*T(r,0)
        //   + 2*sum_r T(r,0)*temp2(r,0), temp2(r,0) = sum_s
        //   (fock_core(r,s)+L(0,0,r,s))*T(s,0). See commit message for the
        //   full worked arithmetic -- result: 5.974.
        expect_near(r.E_core2, 5.974, 1e-9, "nontrivial U: hand-computed E_core2");
    }

    if (failures == 0) {
        std::printf("\nAll tests passed.\n");
        return 0;
    }
    std::printf("\n%d test(s) failed.\n", failures);
    return 1;
}
