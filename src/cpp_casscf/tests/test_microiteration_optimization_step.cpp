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
    CiStateAverageResult solve(const Matrix& eigenvecs_guess, bool use_staged_inputs = false) override {
        (void)use_staged_inputs;
        ++call_count;
        CiStateAverageResult result;
        result.eigenvectors = eigenvecs_guess; // pass-through, arbitrary for this test
        result.eigenvalues = Vector::Zero(1);
        result.avg_energy = 0.0;
        result.D_tu_avg = Matrix::Zero(1, 1);
        result.D_tuvw_avg = Tensor4(1, 1, 1, 1);
        result.D_tuvw_avg.setZero();
        result.Dpe_tu_avg = Matrix::Zero(1, 1);
        return result;
    }

    int call_count = 0;
};

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

    if (failures == 0) {
        std::printf("\nAll tests passed.\n");
        return 0;
    }
    std::printf("\n%d test(s) failed.\n", failures);
    return 1;
}
