// Tests for CasscfInternalOptimizationStep (the real InternalOptimizationStep,
// port of internal_optimization3, helper_PFCI.py:6847-7961). No real
// CiStateAverageSolver/IntegralTransformer exists yet (see
// cpp_casscf/README.md, "What's still open"), so -- same spirit as
// test_macroiteration_driver.cpp -- these are mocked, and the test problem
// itself is chosen to be exactly hand-solvable end to end rather than
// requiring a real chemistry dataset: an all-zero RDM/integral problem
// makes build_intermediates_internal/build_gradient_and_hessian return
// exactly zero A/G/gradient/hessian (both are purely linear contractions of
// their inputs, confirmed by inspection of intermediates.cpp -- no additive
// bias term), so LstrsSolver trivially returns a zero step, the trial
// rotation is the identity, and internal_optimization_exact_energy's
// energy_change comes out to exactly 0.0 (recomputing the same sum_energy
// formula against unchanged inputs), which satisfies its accept condition
// (energy_change <= 0.0) on every microiteration. This exercises the full
// pipeline wiring -- intermediates -> LstrsSolver -> internal_transformation
// -> internal_optimization_exact_energy -> CI-solver commit -> convergence
// check -- end to end, without needing to hand-derive a real LSTRS
// bisection step.
#include "casscf/internal_optimization_step.hpp"

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
    explicit MockCiSolver(bool converged) : converged_(converged) {}

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
        result.ci_diagonalization_converged = converged_;
        return result;
    }

    int call_count = 0;

private:
    bool converged_;
};

class RecordingIntegralTransformer final : public IntegralTransformer {
public:
    void transform_internal_rotation(const Matrix& U_delta) override {
        ++internal_rotation_call_count;
        last_U_delta = U_delta;
    }
    void transform_macroiteration(const Matrix&) override { ++macroiteration_call_count; }

    int internal_rotation_call_count = 0;
    int macroiteration_call_count = 0;
    Matrix last_U_delta;
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
    constants.Enuc = 5.0;
    constants.d_c = 0.05;
    constants.d_exp = 0.0;
    constants.weight = Vector::Constant(1, 1.0);

    // --- Case 1: converges in a single microiteration (gradient is exactly
    //     zero from the start, and the mocked CI solver reports converged
    //     immediately). ---
    {
        MockCiSolver ci_solver(/*converged=*/true);
        RecordingIntegralTransformer integral_transformer;
        CasscfInternalOptimizationStep step(dims, constants, ci_solver, integral_transformer);

        CasscfContext context = make_all_zero_context();
        Matrix eigenvecs = Matrix::Zero(1, 1);
        step.run(context, /*E0=*/0.0, eigenvecs);

        expect_near(integral_transformer.internal_rotation_call_count, 1, 0,
                    "case1: transform_internal_rotation called exactly once");
        expect_near(ci_solver.call_count, 1, 0, "case1: CI solver called exactly once (converges in 1 microiteration)");
        expect_matrix_near(context.U_total, Matrix::Identity(2, 2), tol,
                            "case1: U_total stays identity (U1 == identity, since the step was zero)");
        expect_near(context.E_core, 0.0, tol, "case1: committed E_core == 0");
        expect_matrix_near(context.occupied_fock_core, Matrix::Zero(2, 2), tol,
                            "case1: committed occupied_fock_core == 0");
    }

    // --- Case 2: same all-zero problem, but the mocked CI solver never
    //     reports convergence -- gradient stays exactly zero (trivially
    //     < 1e-4) every iteration, so the only way the loop can terminate
    //     is the microiteration cap, set low here to keep the test fast. ---
    {
        MockCiSolver ci_solver(/*converged=*/false);
        RecordingIntegralTransformer integral_transformer;
        const int max_microiterations = 3;
        CasscfInternalOptimizationStep step(dims, constants, ci_solver, integral_transformer, max_microiterations);

        CasscfContext context = make_all_zero_context();
        Matrix eigenvecs = Matrix::Zero(1, 1);
        step.run(context, /*E0=*/0.0, eigenvecs);

        // microiteration runs 0,1,2,3 (four total) before the cap fires at
        // microiteration == max_microiterations == 3; every iteration is
        // accepted (energy_change == 0.0 exactly each time), so the CI
        // solver is called once per iteration including the last.
        expect_near(ci_solver.call_count, max_microiterations + 1, 0,
                    "case2: CI solver called once per microiteration through the cap");
        expect_near(integral_transformer.internal_rotation_call_count, 1, 0,
                    "case2: transform_internal_rotation still called exactly once, at the cap");
    }

    if (failures == 0) {
        std::printf("\nAll tests passed.\n");
        return 0;
    }
    std::printf("\n%d test(s) failed.\n", failures);
    return 1;
}
