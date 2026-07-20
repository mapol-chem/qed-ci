// Tests for MacroiterationDriver::run -- the orchestration shape ported
// from the macroiteration while-loop at helper_PFCI.py:2394-3060. Not a
// port of any specific Python test: the four collaborators
// (CiStateAverageSolver, InternalOptimizationStep, MicroiterationOptimizationStep,
// IntegralTransformer) are mocked so the loop shape -- call ordering, the
// convergence latch, the n_in_a==0 skip, the restart branch, and the
// H_spatial2/d_cmo/U_total bookkeeping -- can be checked independently of
// any real CASSCF intermediates.
#include "casscf/macroiteration_driver.hpp"
#include "casscf/orbital_rotation.hpp"

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

void expect_true(bool cond, const char* label) {
    if (!cond) {
        std::printf("FAIL: %s\n", label);
        ++failures;
    } else {
        std::printf("PASS: %s\n", label);
    }
}

void expect_matrix_near(const Matrix& actual, const Matrix& expected, double tol, const char* label) {
    expect_near((actual - expected).norm(), 0.0, tol, label);
}

// Returns a fixed sequence of avg_energies (one per solve() call) and
// otherwise-arbitrary eigenvalues/eigenvectors.
class ScriptedCiSolver final : public CiStateAverageSolver {
public:
    explicit ScriptedCiSolver(std::vector<double> avg_energies) : avg_energies_(std::move(avg_energies)) {}

    CiStateAverageResult solve(const Matrix& eigenvecs_guess, bool use_staged_inputs = false) override {
        (void)use_staged_inputs;
        last_eigenvecs_guess = eigenvecs_guess;
        CiStateAverageResult result;
        result.avg_energy = avg_energies_.at(call_count);
        result.eigenvalues = Vector::Constant(1, result.avg_energy);
        result.eigenvectors = eigenvecs_guess; // pass-through, arbitrary for this test
        ++call_count;
        return result;
    }

    int call_count = 0;
    Matrix last_eigenvecs_guess;

private:
    std::vector<double> avg_energies_;
};

class RecordingInternalStep final : public InternalOptimizationStep {
public:
    void run(CasscfContext& context, double E0, Matrix& eigenvecs) override {
        ++call_count;
        last_context = &context;
        last_E0 = E0;
        last_eigenvecs = eigenvecs;
    }

    int call_count = 0;
    CasscfContext* last_context = nullptr;
    double last_E0 = 0.0;
    Matrix last_eigenvecs;
};

// Returns a scripted sequence of U2 matrices (one per run() call, cycling
// the last entry if more calls happen than scripted values).
class ScriptedMicroiterationStep final : public MicroiterationOptimizationStep {
public:
    explicit ScriptedMicroiterationStep(std::vector<Matrix> u2_sequence) : u2_sequence_(std::move(u2_sequence)) {}

    void run(CasscfContext& /*context*/, const Matrix& U, Matrix& eigenvecs, double convergence_threshold) override {
        last_U0.push_back(U);
        last_eigenvecs = eigenvecs;
        last_convergence_threshold = convergence_threshold;
        const size_t idx = call_count < u2_sequence_.size() ? call_count : u2_sequence_.size() - 1;
        u2_ = u2_sequence_.at(idx);
        ++call_count;
    }

    const Matrix& last_U2() const override { return u2_; }

    int call_count = 0;
    std::vector<Matrix> last_U0;
    Matrix last_eigenvecs;
    double last_convergence_threshold = 0.0;

private:
    std::vector<Matrix> u2_sequence_;
    Matrix u2_;
};

class RecordingIntegralTransformer final : public IntegralTransformer {
public:
    void transform_internal_rotation(const Matrix& U_delta) override {
        ++internal_rotation_call_count;
        last_internal_rotation_U_delta = U_delta;
    }

    void transform_macroiteration(const Matrix& U_total) override {
        ++macroiteration_call_count;
        last_macroiteration_U_total = U_total;
    }

    int internal_rotation_call_count = 0;
    int macroiteration_call_count = 0;
    Matrix last_internal_rotation_U_delta;
    Matrix last_macroiteration_U_total;
};

Dimensions make_dims(int n_in_a, int n_act_orb, int n_virtual) {
    Dimensions dims;
    dims.n_in_a = n_in_a;
    dims.n_act_orb = n_act_orb;
    dims.n_virtual = n_virtual;
    dims.n_occupied = n_in_a + n_act_orb;
    dims.nmo = dims.n_occupied + n_virtual;
    return dims;
}

} // namespace

int main() {
    const double tol = 1e-10;

    // --- Case 1: convergence after two CI calls, no restart triggered.
    //     Checks the break sits before internal_step/microiteration_step
    //     for the converging iteration, and that call counts land where
    //     the Python's loop shape implies. ---
    {
        Dimensions dims = make_dims(1, 1, 1);
        MacroiterationDriverConfig config;
        config.dims = dims;
        config.max_macroiterations = 50;

        ScriptedCiSolver ci_solver({-10.5, -10.5}); // second call: |diff| == 0 < energy_convergence
        RecordingInternalStep internal_step;
        ScriptedMicroiterationStep microiteration_step({Matrix::Identity(dims.nmo, dims.nmo)});
        RecordingIntegralTransformer integral_transformer;

        CasscfContext context;
        context.H_spatial2 = Matrix::Zero(dims.nmo, dims.nmo);
        context.d_cmo = Matrix::Zero(dims.nmo, dims.nmo);

        MacroiterationDriver driver(config, ci_solver, internal_step, microiteration_step, integral_transformer);
        MacroiterationResult result = driver.run(Matrix::Identity(1, 1), -10.0, context);

        expect_true(result.converged, "case1: converged");
        expect_near(result.macroiterations_run, 2, 0, "case1: macroiterations_run");
        expect_near(result.avg_energy, -10.5, tol, "case1: final avg_energy");
        expect_near(ci_solver.call_count, 2, 0, "case1: CI solver called twice");
        // internal_step runs once per macroiteration>0 iteration that doesn't
        // break -- macroiteration==1 runs it, macroiteration==2 breaks first.
        expect_near(internal_step.call_count, 1, 0, "case1: internal_step called once");
        // microiteration_step runs once per iteration that isn't broken out
        // of -- macroiteration 0 and 1, not 2.
        expect_near(microiteration_step.call_count, 2, 0, "case1: microiteration_step called twice");
        expect_near(integral_transformer.macroiteration_call_count, 2, 0, "case1: transform_macroiteration called twice");
        expect_near(integral_transformer.internal_rotation_call_count, 0, 0, "case1: no restart triggered");
    }

    // --- Case 2: n_in_a == 0 -- internal_step must never run, even past
    //     macroiteration 0 (helper_PFCI.py:2804: `if macroiteration > 0 and
    //     self.n_in_a > 0`). ---
    {
        Dimensions dims = make_dims(0, 2, 1);
        MacroiterationDriverConfig config;
        config.dims = dims;
        config.max_macroiterations = 3; // never converges -- exhausts the cap

        ScriptedCiSolver ci_solver({-5.0, -6.0});
        RecordingInternalStep internal_step;
        ScriptedMicroiterationStep microiteration_step({Matrix::Identity(dims.nmo, dims.nmo)});
        RecordingIntegralTransformer integral_transformer;

        CasscfContext context;
        context.H_spatial2 = Matrix::Zero(dims.nmo, dims.nmo);
        context.d_cmo = Matrix::Zero(dims.nmo, dims.nmo);

        MacroiterationDriver driver(config, ci_solver, internal_step, microiteration_step, integral_transformer);
        MacroiterationResult result = driver.run(Matrix::Identity(1, 1), -4.0, context);

        expect_true(!result.converged, "case2: did not converge (exhausted cap)");
        expect_near(result.macroiterations_run, 3, 0, "case2: macroiterations_run == max_macroiterations");
        expect_near(internal_step.call_count, 0, 0, "case2: internal_step never called (n_in_a == 0)");
    }

    // --- Case 3: H_spatial2/d_cmo/U_total bookkeeping under a single
    //     iteration, no restart -- exact check against U2^T H U2 / U_total == U2. ---
    {
        Dimensions dims = make_dims(1, 1, 1);
        MacroiterationDriverConfig config;
        config.dims = dims;
        config.max_macroiterations = 1;

        // Symmetric, so R = 0.5*(U2 - U2^T) == 0 identically and no restart
        // fires -- this case is only about the H_spatial2/d_cmo/U_total
        // bookkeeping, not the restart branch (that's Case 4).
        Matrix U2(3, 3);
        U2 << 1.0, 0.1, 0.2,
              0.1, 1.0, 0.3,
              0.2, 0.3, 1.0;

        ScriptedCiSolver ci_solver({});
        RecordingInternalStep internal_step;
        ScriptedMicroiterationStep microiteration_step({U2});
        RecordingIntegralTransformer integral_transformer;

        Matrix H_spatial2(3, 3);
        H_spatial2 << 1.0, 0.2, 0.1,
                      0.2, 2.0, 0.3,
                      0.1, 0.3, 3.0;
        Matrix d_cmo = Matrix::Identity(3, 3) * 0.5;

        CasscfContext context;
        context.H_spatial2 = H_spatial2;
        context.d_cmo = d_cmo;

        MacroiterationDriver driver(config, ci_solver, internal_step, microiteration_step, integral_transformer);
        MacroiterationResult result = driver.run(Matrix::Identity(1, 1), -1.0, context);

        expect_matrix_near(context.U_total, U2, tol, "case3: U_total == U2 (no restart, single iteration)");
        expect_matrix_near(integral_transformer.last_macroiteration_U_total, U2, tol,
                            "case3: transform_macroiteration saw U_total == U2");
        expect_near(integral_transformer.internal_rotation_call_count, 0, 0, "case3: no restart triggered");
        expect_near(microiteration_step.call_count, 1, 0, "case3: microiteration_step called once");
    }

    // --- Case 4: restart branch -- microiteration_step's first U2 has a
    //     large Rai block, forcing the internal-rotation correction. Checks
    //     the correction U_delta matches build_unitary_matrix independently
    //     computed on the same Rai/Rvi/Rva split, that transform_internal_rotation
    //     fires exactly once, that microiteration_step is re-invoked with the
    //     complementary-block U_delta, and that the final U_total is the
    //     product of both corrections and the (identity) second U2. ---
    {
        Dimensions dims = make_dims(1, 1, 1);
        MacroiterationDriverConfig config;
        config.dims = dims;
        config.max_macroiterations = 1;
        config.internal_rotation_restart_threshold = 1e-4;

        // R = 0.5*(U2 - U2^T); want R(n_in_a=1, 0) = R(1,0) to exceed 1e-4.
        // U2(1,0) = 0.5, U2(0,1) = -0.5 -> R(1,0) = 0.5*(0.5 - (-0.5)) = 0.5.
        Matrix U2_first = Matrix::Identity(3, 3);
        U2_first(1, 0) = 0.5;
        U2_first(0, 1) = -0.5;
        Matrix U2_second = Matrix::Identity(3, 3);

        ScriptedCiSolver ci_solver({});
        RecordingInternalStep internal_step;
        ScriptedMicroiterationStep microiteration_step({U2_first, U2_second});
        RecordingIntegralTransformer integral_transformer;

        CasscfContext context;
        context.H_spatial2 = Matrix::Identity(3, 3);
        context.d_cmo = Matrix::Identity(3, 3);

        MacroiterationDriver driver(config, ci_solver, internal_step, microiteration_step, integral_transformer);
        MacroiterationResult result = driver.run(Matrix::Identity(1, 1), -1.0, context);

        expect_near(microiteration_step.call_count, 2, 0, "case4: microiteration_step called twice (restart)");
        expect_near(integral_transformer.internal_rotation_call_count, 1, 0,
                    "case4: transform_internal_rotation called once");

        Matrix Rai(1, 1);
        Rai(0, 0) = 0.5;
        Matrix Rvi_zero = Matrix::Zero(1, 1);
        Matrix Rva_zero = Matrix::Zero(1, 1);
        Matrix expected_U_delta1 = build_unitary_matrix(Rai, Rvi_zero, Rva_zero, dims);
        expect_matrix_near(integral_transformer.last_internal_rotation_U_delta, expected_U_delta1, tol,
                            "case4: transform_internal_rotation saw the Rai-only U_delta");

        // Second microiteration_step call must be seeded with the
        // complementary-block U_delta (Rai == 0, Rvi/Rva from R), not U0.
        Matrix Rai_zero = Matrix::Zero(1, 1);
        Matrix expected_U_delta2 = build_unitary_matrix(Rai_zero, Rvi_zero, Rva_zero, dims); // Rvi, Rva are also 0 here
        expect_matrix_near(microiteration_step.last_U0.at(1), expected_U_delta2, tol,
                            "case4: second microiteration_step call seeded with complementary-block U_delta");

        Matrix expected_U_total = expected_U_delta1 * expected_U_delta2 * U2_second;
        expect_matrix_near(context.U_total, expected_U_total, tol,
                            "case4: U_total == U_delta1 * U_delta2 * final U2");
    }

    if (failures == 0) {
        std::printf("\nAll tests passed.\n");
        return 0;
    }
    std::printf("\n%d test(s) failed.\n", failures);
    return 1;
}
