// Tests for CasscfMicroiterationOptimizationStep (the real
// MicroiterationOptimizationStep, port of microiteration_optimization6,
// helper_PFCI.py:10908-12423). No real CiStateAverageSolver exists yet, so --
// same spirit as test_internal_optimization_step.cpp -- it's mocked, and the
// test problem is chosen to be exactly hand-solvable end to end: an all-zero
// context/RDM/constants problem makes build_intermediates return exactly
// zero A/G/fock_core/E_core/active_fock_core/active_twoeint/L (purely linear
// contractions of zero inputs), so zero_energy == 0, gradient_tilde == 0, and
// the inner loop's gradient_norm < 1e-7 check fires immediately every outer
// pass without ever computing a trust-region step -- exercising
// build_intermediates -> zero_energy -> build_gradient/build_hessian_diagonal
// -> inner-loop small-gradient break -> the accepted_count==0 reference-point
// fallback -> commit_ci_solver_inputs -> the injected CI solver, end to end,
// without needing to hand-derive a real GLTR/Davidson/PCG step.
//
// With every pass's current_energy identically 0.0, the outer loop's own
// small-energy-change convergence check (microiteration >= 2 required) fires
// deterministically on the third pass (microiteration == 2) -- so case 1
// (a generous max_microiterations) verifies that natural convergence path,
// and case 2 (max_microiterations == 1) verifies the separate microiteration
// cap by cutting the run short before that convergence check could ever
// fire.
#include "casscf/microiteration_optimization_step.hpp"

#include <cmath>
#include <cstdio>
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

void expect_matrix_near(const Matrix& actual, const Matrix& expected, double tol, const char* label) {
    expect_near((actual - expected).norm(), 0.0, tol, label);
}

class MockCiSolver final : public CiStateAverageSolver {
public:
    CiStateAverageResult solve(const Matrix& eigenvecs_guess, bool use_staged_inputs = false,
                                std::optional<double> davidson_threshold_override = std::nullopt,
                                std::optional<int> davidson_maxiter_override = std::nullopt) override {
        (void)use_staged_inputs;
        (void)davidson_threshold_override;
        (void)davidson_maxiter_override;
        ++call_count;
        CiStateAverageResult result;
        result.eigenvectors = eigenvecs_guess; // pass-through, arbitrary for this test
        result.eigenvalues = Vector::Zero(1);
        result.avg_energy = 0.0;
        result.D_tu_avg = Matrix::Zero(1, 1);
        result.D_tuvw_avg = Tensor4(1, 1, 1, 1);
        result.D_tuvw_avg.setZero();
        result.Dpe_tu_avg = Matrix::Zero(1, 1);
        // helper_PFCI.py:11003's own current_residual=1 sentinel -- this mock
        // has no real Davidson residual to report, so it holds the total_norm
        // break (helper_PFCI.py:11243-11256) at its Python-matching initial
        // value instead of defaulting to 0.0, which would make that break
        // fire a pass early and change what this test is actually exercising
        // (the small-energy-change break in Step 2, not the total-norm break).
        result.residual_norm = 1.0;
        return result;
    }

    int call_count = 0;
};

// --- QN/BFGS wiring test -------------------------------------------------
//
// Records every davidson_threshold_override this class passes, instead of
// hand-deriving a real GLTR/Davidson step (this system is deliberately NOT
// hand-solvable the way the all-zero problem above is): the override value
// is exactly 1e-9 while qn_count == 0 and exactly 0.1*||reduced_gradient||
// once qn_count > 0 (helper_PFCI.py:12399-12404) -- a call-by-call trace of
// this value is therefore a direct, black-box witness of when qn_count
// transitions from 0 to > 0, without needing access to the class's private
// qn_optimization/qn_count/BfgsOperator state.
class RecordingCiSolver final : public CiStateAverageSolver {
public:
    CiStateAverageResult solve(const Matrix& eigenvecs_guess, bool use_staged_inputs = false,
                                std::optional<double> davidson_threshold_override = std::nullopt,
                                std::optional<int> davidson_maxiter_override = std::nullopt) override {
        (void)use_staged_inputs;
        recorded_thresholds.push_back(davidson_threshold_override.value_or(-1.0));
        recorded_maxiters.push_back(davidson_maxiter_override.value_or(-1));
        ++call_count;
        CiStateAverageResult result;
        result.eigenvectors = eigenvecs_guess;
        result.eigenvalues = Vector::Zero(1);
        result.avg_energy = 0.0;
        result.D_tu_avg = Matrix::Zero(1, 1);
        result.D_tuvw_avg = Tensor4(1, 1, 1, 1);
        result.D_tuvw_avg.setZero();
        result.Dpe_tu_avg = Matrix::Zero(1, 1);
        result.residual_norm = 1.0; // see MockCiSolver's own comment above
        return result;
    }

    int call_count = 0;
    std::vector<double> recorded_thresholds;
    std::vector<int> recorded_maxiters;
};

// dims: n_in_a=1, n_act_orb=1, n_virtual=1 (nmo=3, n_occupied=2) --
// index_map_size == 3 (one inactive-active, one inactive-virtual, one
// active-virtual rotation parameter), small enough to keep GLTR/Davidson
// cheap but big enough that n_negative == 0 is plausible (not asserted
// directly -- see below).
CasscfContext make_small_nonzero_gradient_context() {
    CasscfContext context;
    // J == K == 0 and N_p == 0 (see constants below) together zero out
    // every two-electron and photon-coupling contribution to A/gradient_tilde,
    // leaving A_ri == 2*F_ri == 2*H_spatial2[r,i] (since D_tu_avg == 0 too,
    // from the mocked CI solver's all-zero RDMs) -- a small, deliberately
    // tiny off-diagonal H_spatial2 is therefore the only source of a
    // nonzero gradient in this test, and its magnitude directly controls
    // the resulting gradient norm.
    context.H_spatial2 = Matrix(3, 3);
    context.H_spatial2 << 0.0, 1.0e-4, 2.0e-4, 1.0e-4, 0.5, 3.0e-4, 2.0e-4, 3.0e-4, 1.0;
    context.d_cmo = Matrix::Zero(3, 3);
    context.U_total = Matrix::Identity(3, 3);
    context.J = Tensor4(2, 2, 3, 3);
    context.J.setZero();
    context.K = Tensor4(2, 2, 3, 3);
    context.K.setZero();
    context.occupied_h1 = Matrix::Zero(2, 2);
    context.occupied_d_cmo = Matrix::Zero(2, 2);
    context.occupied_fock_core = Matrix::Zero(2, 2);
    context.occupied_J = Tensor4(2, 2, 2, 2);
    context.occupied_J.setZero();
    context.occupied_K = Tensor4(2, 2, 2, 2);
    context.occupied_K.setZero();
    context.D_tu_avg = Matrix::Zero(1, 1);
    context.D_tuvw_avg = Tensor4(1, 1, 1, 1);
    context.D_tuvw_avg.setZero();
    context.Dpe_tu_avg = Matrix::Zero(1, 1);
    return context;
}

CasscfContext make_all_zero_context() {
    CasscfContext context;
    context.H_spatial2 = Matrix::Zero(2, 2);
    context.d_cmo = Matrix::Zero(2, 2);
    context.U_total = Matrix::Identity(2, 2);
    context.J = Tensor4(2, 2, 2, 2);
    context.J.setZero();
    context.K = Tensor4(2, 2, 2, 2);
    context.K.setZero();
    context.occupied_h1 = Matrix::Zero(2, 2);
    context.occupied_d_cmo = Matrix::Zero(2, 2);
    context.occupied_fock_core = Matrix::Zero(2, 2);
    context.occupied_J = Tensor4(2, 2, 2, 2);
    context.occupied_J.setZero();
    context.occupied_K = Tensor4(2, 2, 2, 2);
    context.occupied_K.setZero();
    context.D_tu_avg = Matrix::Zero(1, 1);
    context.D_tuvw_avg = Tensor4(1, 1, 1, 1);
    context.D_tuvw_avg.setZero();
    context.Dpe_tu_avg = Matrix::Zero(1, 1);
    return context;
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

    CasscfPhysicalConstants constants;
    constants.N_p = 0; // isolates this test from calculate_ci_dependent_energy /
                        // calculate_off_diagonal_photon_constant, both exactly 0 at N_p == 0
    constants.num_det = 1;
    constants.omega = 0.1;
    constants.Enuc = 0.0;
    constants.d_c = 0.0;
    constants.d_exp = 0.0;
    constants.weight = Vector::Constant(1, 1.0);

    // --- Case 1: natural convergence. current_energy is identically 0.0
    //     every pass, so the outer loop's small-energy-change check fires as
    //     soon as it's eligible (microiteration >= 2, i.e. on the third
    //     pass, microiteration == 2) -- the CI solver is called once per
    //     pass that actually runs to completion (microiteration 0 and 1),
    //     not on the pass that breaks early. ---
    {
        MockCiSolver ci_solver;
        CasscfMicroiterationOptimizationStep step(dims, constants, ci_solver, /*max_microiterations=*/20);

        CasscfContext context = make_all_zero_context();
        Matrix eigenvecs = Matrix::Zero(1, 1);
        step.run(context, /*U=*/Matrix::Identity(2, 2), eigenvecs, /*convergence_threshold=*/1e-4);

        expect_near(ci_solver.call_count, 2, 0,
                    "case1: CI solver called exactly twice (microiteration 0 and 1; "
                    "microiteration 2 breaks before reaching the CI solve)");
        expect_matrix_near(step.last_U2(), Matrix::Identity(2, 2), tol,
                            "case1: U2 stays identity (gradient is zero every pass, no step ever accepted)");
        expect_near(context.E_core, 0.0, tol, "case1: committed E_core == 0");
        expect_near(context.E_core2, 0.0, tol, "case1: committed E_core2 == 0");
        expect_matrix_near(context.occupied_fock_core, Matrix::Zero(2, 2), tol,
                            "case1: committed occupied_fock_core == 0");
        expect_matrix_near(context.gkl2, Matrix::Zero(1, 1), tol, "case1: committed gkl2 == 0");
    }

    // --- Case 2: same all-zero problem, but max_microiterations == 1 cuts
    //     the run short before the natural (microiteration >= 2) convergence
    //     check could ever fire -- verifies the separate microiteration cap
    //     path terminates the outer loop on its own. ---
    {
        MockCiSolver ci_solver;
        CasscfMicroiterationOptimizationStep step(dims, constants, ci_solver, /*max_microiterations=*/1);

        CasscfContext context = make_all_zero_context();
        Matrix eigenvecs = Matrix::Zero(1, 1);
        step.run(context, /*U=*/Matrix::Identity(2, 2), eigenvecs, /*convergence_threshold=*/1e-4);

        expect_near(ci_solver.call_count, 1, 0,
                    "case2: CI solver called exactly once before the microiteration cap fires");
        expect_matrix_near(step.last_U2(), Matrix::Identity(2, 2), tol,
                            "case2: U2 stays identity");
    }

    // --- Case 3: QN activation + wiring. A small, genuinely nonzero (but
    //     deliberately tiny) gradient keeps microiteration 0 in the
    //     gradient-small Newton-fallback regime (1e-7 < ||g|| <= 1e-3, see
    //     RecordingCiSolver's own comment) -- hard_case is FORCED to 2
    //     there, so the very first inner-loop trial is unconditionally
    //     accepted regardless of energy_change's sign, deterministically
    //     without needing to hand-derive a real GLTR/Davidson step. The
    //     resulting step is small enough (well under the 0.05 QN-activation
    //     threshold) that QN activates on this same first inner iteration --
    //     confirmed empirically (not just assumed): activation happens
    //     BEFORE this pass's own CI-solve tail call
    //     (helper_PFCI.py:12222-12228 sits well before the CI-solve block),
    //     so BOTH of this run's CI-solve calls (pass 0's own tail call
    //     included) observe qn_count > 0, not just pass 1's. Both recorded
    //     thresholds are therefore asserted to be the post-activation
    //     override, not the pre-activation base value -- there is no pass in
    //     THIS construction where the base 1e-9/5 threshold would ever be
    //     used, since the gradient is small enough to trigger both the
    //     Newton-fallback's forced accept AND QN activation on the very
    //     first inner iteration ever attempted. (should_activate_qn's own
    //     truth table, including cases that do NOT activate, is covered
    //     directly and exactly in test_quasi_newton_decisions.cpp -- this
    //     test's job is to confirm the surrounding wiring calls it
    //     correctly, not to re-derive its logic.) max_microiterations == 2
    //     keeps the run short enough that the outer loop's own
    //     small-energy-change convergence check (only eligible at
    //     microiteration >= 2) can never fire and cut the run short before
    //     both passes complete.
    //
    //     This deliberately does NOT hand-verify the GLTR step used on pass
    //     1 (the QN branch's own "exact Hessian at reference" GLTR solve) --
    //     unlike the all-zero case above, this problem was chosen to be
    //     *reachable* by the QN wiring, not hand-solvable end to end. What
    //     it verifies instead is the wiring itself: (a) QN activates on the
    //     pass this construction predicts, evidenced by the CI-solve
    //     threshold-override trace departing from the pre-QN 1e-9/5 pair
    //     (helper_PFCI.py:12399-12404); (b) the BfgsOperator reference-point
    //     construction (via top-of-loop should_reset_bfgs_reference_point)
    //     and the QN branch's own should_reset_bfgs_reference dispatch to
    //     the exact-Hessian-at-reference GLTR sub-branch both run without
    //     crashing or producing NaN/garbage; (c) the resulting U2 is still a
    //     genuine rotation (orthogonal) after two passes through this new
    //     code path.
    {
        Dimensions dims3;
        dims3.n_in_a = 1;
        dims3.n_act_orb = 1;
        dims3.n_virtual = 1;
        dims3.nmo = 3;
        dims3.n_occupied = 2;

        CasscfPhysicalConstants constants3;
        constants3.N_p = 0;
        constants3.num_det = 1;
        constants3.omega = 0.1;
        constants3.Enuc = 0.0;
        constants3.d_c = 0.0;
        constants3.d_exp = 0.0;
        constants3.weight = Vector::Constant(1, 1.0);

        RecordingCiSolver ci_solver;
        CasscfMicroiterationOptimizationStep step(dims3, constants3, ci_solver, /*max_microiterations=*/2);

        CasscfContext context = make_small_nonzero_gradient_context();
        Matrix eigenvecs = Matrix::Zero(1, 1);
        step.run(context, /*U=*/Matrix::Identity(3, 3), eigenvecs, /*convergence_threshold=*/1e-4);

        expect_near(ci_solver.call_count, 2, 0, "case3: CI solver called exactly twice (both passes complete)");
        if (static_cast<int>(ci_solver.recorded_thresholds.size()) == 2) {
            // Both calls, not just the second: activation happens inside
            // pass 0's own inner loop, before that same pass's CI-solve
            // tail call -- see this case's own comment above.
            for (int i = 0; i < 2; ++i) {
                const bool is_qn_override = ci_solver.recorded_thresholds[i] > 0.0 &&
                                             std::abs(ci_solver.recorded_thresholds[i] - 1e-9) > 1e-12;
                if (is_qn_override) {
                    std::printf("PASS: case3: pass %d uses the post-activation QN override threshold "
                                "(%.6e != 1e-9)\n",
                                i, ci_solver.recorded_thresholds[i]);
                } else {
                    std::printf("FAIL: case3: pass %d threshold %.6e -- expected the QN override (!= 1e-9); QN "
                                "did not activate as this test's construction requires\n",
                                i, ci_solver.recorded_thresholds[i]);
                    ++failures;
                }
                expect_near(ci_solver.recorded_maxiters[i], 10000, 0,
                            "case3: pass uses the QN override davidson maxiter");
            }
        } else {
            std::printf("FAIL: case3: expected exactly 2 recorded CI-solve calls, got %zu\n",
                        ci_solver.recorded_thresholds.size());
            ++failures;
        }

        const Matrix& U2 = step.last_U2();
        expect_matrix_near(U2.transpose() * U2, Matrix::Identity(3, 3), 1e-8,
                            "case3: U2 remains a genuine (orthogonal) rotation after QN activation + one QN pass");
        expect_near((U2 - Matrix::Identity(3, 3)).norm() > 1e-8 ? 1.0 : 0.0, 1.0, 0.0,
                    "case3: U2 differs from identity (a rotation was actually applied both passes)");
    }

    if (failures == 0) {
        std::printf("\nAll tests passed.\n");
        return 0;
    }
    std::printf("\n%d test(s) failed.\n", failures);
    return 1;
}
