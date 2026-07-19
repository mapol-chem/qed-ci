// Tests for internal_optimization.hpp (port of internal_transformation,
// internal_optimization_exact_energy, internal_optimization_predicted_energy,
// step_control -- helper_PFCI.py:6452-6877, 8018-8025) and
// calculate_ci_dependent_energy (intermediates.hpp, helper_PFCI.py:6047-6113),
// which internal_optimization_exact_energy calls.
//
// internal_transformation and internal_optimization_exact_energy are hand
// -verified against small, exactly hand-computable cases (paper arithmetic,
// not a re-derivation of the same C++ formula) rather than a brute-force
// mirror of the port, matching the style of test_orbital_rotation.cpp /
// test_pcg_trust_region.cpp.
#include "casscf/internal_optimization.hpp"
#include "casscf/intermediates.hpp"

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
    double err = (actual - expected).norm();
    expect_near(err, 0.0, tol, label);
}

} // namespace

int main() {
    const double tol = 1e-9;

    // --- internal_transformation: n=2 occupied block, J built from a
    //     symmetric outer product (J(p,q,r,s) = S(p,q)*S(r,s)) so it has the
    //     real chemical 8-fold permutational symmetry -- the K assignment
    //     K(i,j,k,l) = J_out(j,l,i,k) is only equal to the literal numpy
    //     `occupied_twoeint2.transpose(1,3,0,2)` (K[i,j,k,l] = J_out[k,i,l,j])
    //     when J_out has that symmetry, so this case is what actually
    //     exercises whether that equivalence holds. Reference computed here
    //     via the *literal* transpose formula, independent of how the port
    //     phrased it. ---
    {
        const int n = 2;
        Dimensions dims;
        dims.n_occupied = n;
        dims.nmo = n;

        Matrix U(2, 2);
        U << 0.8, -0.3,
             0.4, 0.9;

        Matrix S(2, 2);
        S << 1.0, 0.2,
             0.2, 1.0;

        Tensor4 occupied_J(n, n, n, n);
        for (int p = 0; p < n; ++p)
            for (int q = 0; q < n; ++q)
                for (int r = 0; r < n; ++r)
                    for (int s = 0; s < n; ++s) occupied_J(p, q, r, s) = S(p, q) * S(r, s);

        Matrix occupied_h1(2, 2);
        occupied_h1 << 1.0, 0.1,
                       0.1, 2.0;
        Matrix occupied_d_cmo(2, 2);
        occupied_d_cmo << 0.5, 0.05,
                          0.05, 0.3;

        InternalTransformationResult result =
            internal_transformation(U, occupied_h1, occupied_d_cmo, occupied_J, dims);

        // Reference: quarter-transform J one leg at a time (literal
        // translation of the four chained einsums), then the literal
        // transpose(1,3,0,2) for K: K_ref[i,j,k,l] = J_out[k,i,l,j].
        Tensor4 t1(n, n, n, n), t2(n, n, n, n);
        for (int k = 0; k < n; ++k)
            for (int l = 0; l < n; ++l)
                for (int m = 0; m < n; ++m)
                    for (int s = 0; s < n; ++s) {
                        double acc = 0.0;
                        for (int nn = 0; nn < n; ++nn) acc += occupied_J(k, l, m, nn) * U(nn, s);
                        t1(k, l, m, s) = acc;
                    }
        for (int k = 0; k < n; ++k)
            for (int l = 0; l < n; ++l)
                for (int r = 0; r < n; ++r)
                    for (int s = 0; s < n; ++s) {
                        double acc = 0.0;
                        for (int m = 0; m < n; ++m) acc += t1(k, l, m, s) * U(m, r);
                        t2(k, l, r, s) = acc;
                    }
        for (int k = 0; k < n; ++k)
            for (int q = 0; q < n; ++q)
                for (int r = 0; r < n; ++r)
                    for (int s = 0; s < n; ++s) {
                        double acc = 0.0;
                        for (int l = 0; l < n; ++l) acc += t2(k, l, r, s) * U(l, q);
                        t1(k, q, r, s) = acc;
                    }
        Tensor4 J_ref(n, n, n, n);
        for (int p = 0; p < n; ++p)
            for (int q = 0; q < n; ++q)
                for (int r = 0; r < n; ++r)
                    for (int s = 0; s < n; ++s) {
                        double acc = 0.0;
                        for (int k = 0; k < n; ++k) acc += t1(k, q, r, s) * U(k, p);
                        J_ref(p, q, r, s) = acc;
                    }
        Tensor4 K_ref(n, n, n, n);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                for (int k = 0; k < n; ++k)
                    for (int l = 0; l < n; ++l) K_ref(i, j, k, l) = J_ref(k, i, l, j);

        double j_err = 0.0, k_err = 0.0;
        for (int p = 0; p < n; ++p)
            for (int q = 0; q < n; ++q)
                for (int r = 0; r < n; ++r)
                    for (int s = 0; s < n; ++s) {
                        j_err += std::abs(result.J(p, q, r, s) - J_ref(p, q, r, s));
                        k_err += std::abs(result.K(p, q, r, s) - K_ref(p, q, r, s));
                    }
        expect_near(j_err, 0.0, tol, "internal_transformation: J matches literal quarter-transform reference");
        expect_near(k_err, 0.0, tol,
                    "internal_transformation: K matches literal transpose(1,3,0,2) reference (via J symmetry)");

        Matrix h1_ref = U.transpose() * occupied_h1 * U;
        Matrix d_cmo_ref = U.transpose() * occupied_d_cmo * U;
        expect_matrix_near(result.h1, h1_ref, tol, "internal_transformation: h1 == U^T occupied_h1 U");
        expect_matrix_near(result.d_cmo1, d_cmo_ref, tol, "internal_transformation: d_cmo1 == U^T occupied_d_cmo U");
    }

    // --- calculate_ci_dependent_energy: N_p=1, davidson_roots=1, num_det=1,
    //     hand-computed on paper (see test file's git history / commit
    //     message for the worked arithmetic). ---
    {
        Matrix eigenvecs(1, 2);
        eigenvecs << 0.6, 0.8;
        Matrix occupied_d_cmo(1, 1);
        occupied_d_cmo << 0.4;
        Vector weight(1);
        weight << 1.0;

        double result = calculate_ci_dependent_energy(eigenvecs, occupied_d_cmo, weight,
                                                        /*N_p=*/1, /*num_det=*/1, /*omega=*/0.2,
                                                        /*d_exp=*/1.5, /*n_in_a=*/1);
        expect_near(result, 0.34050505876, 1e-9, "calculate_ci_dependent_energy: hand-computed N_p=1 case");
    }

    // --- calculate_ci_dependent_energy: N_p == 0 short-circuits to exactly
    //     zero regardless of other inputs (helper_PFCI.py:6060-6061). ---
    {
        Matrix eigenvecs(1, 1);
        eigenvecs << 1.0;
        Matrix occupied_d_cmo(1, 1);
        occupied_d_cmo << 99.0;
        Vector weight(1);
        weight << 1.0;
        double result = calculate_ci_dependent_energy(eigenvecs, occupied_d_cmo, weight, 0, 1, 0.5, 1.0, 1);
        expect_near(result, 0.0, 0.0, "calculate_ci_dependent_energy: N_p == 0 returns exactly 0");
    }

    // --- internal_optimization_exact_energy: n_in_a=1, n_act_orb=1,
    //     n_occupied=2, N_p=0 (so ci_dependent_energy == 0, isolating this
    //     function's own arithmetic), hand-computed on paper. Exercises all
    //     three accept/reject paths off the same base numbers. ---
    {
        Dimensions dims;
        dims.n_in_a = 1;
        dims.n_act_orb = 1;
        dims.n_occupied = 2;

        Matrix occupied_h1(2, 2);
        occupied_h1 << 1.0, 0.0,
                       0.0, 2.0;
        Matrix occupied_d_cmo(2, 2);
        occupied_d_cmo << 0.0, 0.0,
                          0.0, 0.3;

        Matrix S(2, 2);
        S << 1.0, 0.2,
             0.2, 1.0;
        Matrix T2(2, 2);
        T2 << 0.5, 0.1,
              0.1, 0.4;
        Tensor4 occupied_J(2, 2, 2, 2), occupied_K(2, 2, 2, 2);
        for (int p = 0; p < 2; ++p)
            for (int q = 0; q < 2; ++q)
                for (int r = 0; r < 2; ++r)
                    for (int s = 0; s < 2; ++s) {
                        occupied_J(p, q, r, s) = S(p, q) * S(r, s);
                        occupied_K(p, q, r, s) = T2(p, q) * T2(r, s);
                    }

        StateAverageData sad;
        sad.D_tu_avg = Matrix(1, 1);
        sad.D_tu_avg << 0.7;
        sad.D_tuvw_avg = Tensor4(1, 1, 1, 1);
        sad.D_tuvw_avg(0, 0, 0, 0) = 0.9;
        sad.Dpe_tu_avg = Matrix(1, 1);
        sad.Dpe_tu_avg << 0.6;
        sad.weight = Vector(1);
        sad.weight << 1.0;
        sad.N_p = 0;
        sad.num_det = 1;
        sad.omega = 0.1;
        sad.Enuc = 5.0;
        sad.d_c = 0.05;
        sad.d_exp = 0.0;

        Matrix eigenvecs(1, 1);
        eigenvecs << 1.0;

        const double sum_energy_expected = 11.8697507; // see file header / worked derivation

        // Case A: E0 chosen so energy_change < 0 -> accepted regardless of
        // hard_case.
        {
            InternalOptimizationEnergyResult r = internal_optimization_exact_energy(
                /*E0=*/12.0, eigenvecs, occupied_h1, occupied_d_cmo, occupied_J, occupied_K,
                /*hard_case=*/0, sad, dims);
            expect_near(r.energy_change, sum_energy_expected - 12.0, 1e-6,
                        "internal_optimization_exact_energy: energy_change (accept case)");
            if (r.accepted) {
                std::printf("PASS: internal_optimization_exact_energy: accepted on negative energy_change\n");
            } else {
                std::printf("FAIL: internal_optimization_exact_energy: accepted on negative energy_change\n");
                ++failures;
            }
            expect_matrix_near(r.occupied_fock_core.block(1, 1, 1, 1), Matrix::Constant(1, 1, 3.8), 1e-6,
                                "internal_optimization_exact_energy: committed fock_active == 3.8 (accept case)");
            expect_near(r.E_core, 3.75, 1e-6, "internal_optimization_exact_energy: committed E_core (accept case)");
            expect_near(r.gkl2(0, 0), 3.3, 1e-6, "internal_optimization_exact_energy: committed gkl2 (accept case)");
        }

        // Case B: E0 chosen so energy_change > 0, hard_case != 2 -> rejected,
        // no commit (result.E_core stays at its default 0.0).
        {
            InternalOptimizationEnergyResult r = internal_optimization_exact_energy(
                /*E0=*/11.0, eigenvecs, occupied_h1, occupied_d_cmo, occupied_J, occupied_K,
                /*hard_case=*/0, sad, dims);
            expect_near(r.energy_change, sum_energy_expected - 11.0, 1e-6,
                        "internal_optimization_exact_energy: energy_change (reject case)");
            if (!r.accepted) {
                std::printf("PASS: internal_optimization_exact_energy: rejected on positive energy_change, hard_case != 2\n");
            } else {
                std::printf("FAIL: internal_optimization_exact_energy: rejected on positive energy_change, hard_case != 2\n");
                ++failures;
            }
            expect_near(r.E_core, 0.0, 0.0,
                        "internal_optimization_exact_energy: no commit on reject (E_core left default)");
        }

        // Case C: same positive energy_change as Case B, but hard_case == 2
        // -> accepted anyway (helper_PFCI.py:6815).
        {
            InternalOptimizationEnergyResult r = internal_optimization_exact_energy(
                /*E0=*/11.0, eigenvecs, occupied_h1, occupied_d_cmo, occupied_J, occupied_K,
                /*hard_case=*/2, sad, dims);
            if (r.accepted) {
                std::printf("PASS: internal_optimization_exact_energy: accepted on hard_case == 2 despite positive energy_change\n");
            } else {
                std::printf("FAIL: internal_optimization_exact_energy: accepted on hard_case == 2 despite positive energy_change\n");
                ++failures;
            }
            expect_near(r.E_core, 3.75, 1e-6, "internal_optimization_exact_energy: committed E_core (hard_case==2 case)");
        }
    }

    // --- internal_optimization_predicted_energy: plain quadratic form. ---
    {
        Vector gradient_ai(2);
        gradient_ai << 0.3, -0.1;
        Matrix hessian_ai(2, 2);
        hessian_ai << 2.0, 0.5,
                      0.5, 1.5;
        Vector Rai(2);
        Rai << 0.2, 0.4;

        double result = internal_optimization_predicted_energy(gradient_ai, hessian_ai, Rai);
        // 2*(0.3*0.2 + -0.1*0.4) + [0.2 0.4] * hessian_ai * [0.2;0.4]
        double expected = 2.0 * (0.3 * 0.2 + -0.1 * 0.4) + Rai.dot(hessian_ai * Rai);
        expect_near(result, expected, tol, "internal_optimization_predicted_energy: quadratic form");
    }

    // --- step_control: threshold behavior, helper_PFCI.py:8018-8025. ---
    {
        expect_near(step_control(0.1, 1.0), 0.7, tol, "step_control: ratio in [0,0.25] shrinks by 0.7");
        expect_near(step_control(0.25, 1.0), 0.7, tol, "step_control: ratio == 0.25 boundary shrinks");
        expect_near(step_control(0.0, 1.0), 0.7, tol, "step_control: ratio == 0.0 boundary shrinks");
        expect_near(step_control(0.5, 1.0), 1.0, tol, "step_control: ratio in (0.25,0.75] unchanged");
        expect_near(step_control(0.75, 1.0), 1.0, tol, "step_control: ratio == 0.75 boundary unchanged (not > 0.75)");
        expect_near(step_control(0.9, 0.5), 0.6, tol, "step_control: ratio > 0.75 grows by 1.2, uncapped");
        expect_near(step_control(0.9, 1.0), 0.75, tol, "step_control: ratio > 0.75 growth capped at 0.75");
        expect_near(step_control(-0.5, 1.0), 1.0, tol,
                    "step_control: negative ratio fails the ratio >= 0 guard, left unchanged");
    }

    if (failures == 0) {
        std::printf("\nAll tests passed.\n");
        return 0;
    }
    std::printf("\n%d test(s) failed.\n", failures);
    return 1;
}
