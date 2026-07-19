// Tests for CasscfCiSetup + CasscfCiStateAverageSolver (the real
// CiStateAverageSolver, the CI diagonalization + weighted state-average
// energy + RDM build that runs at the top of each macroiteration after the
// first, helper_PFCI.py:2424-2553). Like test_integral_transformer.cpp,
// this runs against the REAL compiled ci_solver.c backend (a genuine
// Davidson CI diagonalization), not a mock or hand-derived reference for
// the C library's own internals -- but the *problem* itself is chosen to
// be exactly hand-solvable, so the C++ wiring (array marshaling, gkl2/
// occupied_fock_core/occupied_twoeint construction, constint/constdouble
// layout, RDM accumulation/symmetrization) can be checked against an
// independently-derivable expected answer.
//
// Setup: 2 active orbitals, n_act_a == 1 (1 alpha + 1 beta electron --
// confirmed empirically that CasscfCiConfig::n_act_a sets BOTH the alpha
// and beta electron count via num_det = num_alpha^2, i.e. this models a
// closed-shell S_z == 0 active space, 2 electrons total, not 1), a
// DIAGONAL one-electron Hamiltonian (H_spatial2 = diag(0, 1)), and all
// two-electron integrals (J, K) exactly zero -- a genuinely
// non-interacting active space, so each CI determinant's energy is just
// (electron count) * (orbital energy), with no coupling between
// determinants. N_p == 0 (as in this project's other all-zero-style
// tests) turns off every photon-related energy term.
#include "casscf/ci_setup.hpp"
#include "casscf/ci_state_average_solver.hpp"

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

Dimensions make_dims() {
    Dimensions dims;
    dims.n_in_a = 0;
    dims.n_act_orb = 2;
    dims.n_virtual = 0;
    dims.nmo = 2;
    dims.n_occupied = 2;
    return dims;
}

CasscfPhysicalConstants make_constants(int davidson_roots, double weight_value) {
    CasscfPhysicalConstants constants;
    constants.N_p = 0; // isolates this test from photon-basis energy terms, same trick
                        // as test_internal_optimization_step.cpp / test_microiteration_optimization_step.cpp
    constants.num_det = 4;
    constants.omega = 0.0;
    constants.Enuc = 0.0;
    constants.d_c = 0.0;
    constants.d_exp = 0.0;
    constants.weight = Vector::Constant(davidson_roots, weight_value);
    return constants;
}

CasscfContext make_context(const Matrix& H_spatial2, const Dimensions& dims) {
    CasscfContext context;
    context.H_spatial2 = H_spatial2;
    context.d_cmo = Matrix::Zero(dims.nmo, dims.nmo);
    context.J = Tensor4(dims.n_occupied, dims.n_occupied, dims.nmo, dims.nmo);
    context.J.setZero();
    context.K = Tensor4(dims.n_occupied, dims.n_occupied, dims.nmo, dims.nmo);
    context.K.setZero();
    context.E_core = 0.0;
    return context;
}

} // namespace

int main() {
    const double tol = 1e-8;

    Dimensions dims = make_dims();
    Matrix H_spatial2 = Matrix::Zero(dims.nmo, dims.nmo);
    H_spatial2(0, 0) = 0.0;
    H_spatial2(1, 1) = 1.0;

    // --- Case 1: 1 root. Non-interacting Hamiltonian's ground state puts
    //     both electrons in orbital 0 (energy 2*0.0 == 0.0), 1-RDM ==
    //     diag(2, 0) (2 electrons in orbital 0, 0 in orbital 1). ---
    {
        CasscfCiConfig config;
        config.n_act_a = 1;
        config.davidson_roots = 1;
        config.davidson_threshold = 1e-10;
        config.davidson_indim = 2;
        config.davidson_maxdim = 4;
        config.davidson_maxiter = 200;
        config.target_spin = -1.0;

        CasscfPhysicalConstants constants = make_constants(1, 1.0);
        CasscfContext context = make_context(H_spatial2, dims);

        CasscfCiSetup setup(dims, config, constants, H_spatial2, context.J, context.K, /*E_core=*/0.0);
        expect_near(setup.H_dim(), 4, 0, "case1: H_dim == num_alpha^2 * (N_p+1) == 2^2*1");

        CasscfCiStateAverageSolver solver(dims, config, constants, setup, context);
        Matrix eigenvecs_guess = Matrix::Zero(1, setup.H_dim());
        eigenvecs_guess(0, 0) = 1.0;

        CiStateAverageResult result = solver.solve(eigenvecs_guess);

        expect_near(result.ci_diagonalization_converged, 1.0, 0, "case1: Davidson converged");
        expect_near(result.eigenvalues(0), 0.0, tol, "case1: ground-state eigenvalue == 0.0");
        expect_near(result.avg_energy, 0.0, tol, "case1: avg_energy == 0.0");

        Matrix expected_D_tu(2, 2);
        expected_D_tu << 2.0, 0.0, 0.0, 0.0;
        expect_matrix_near(result.D_tu_avg, expected_D_tu, tol, "case1: D_tu_avg == diag(2, 0)");
    }

    // --- Case 2: 2 equally-weighted roots. Ground state (both electrons in
    //     orbital 0, energy 0.0) and first excited state (both electrons in
    //     orbital 1, energy 2*1.0 == 2.0) -- avg_energy == 0.5*0 + 0.5*2 ==
    //     1.0, state-averaged 1-RDM == diag(1, 1) (average of diag(2,0) and
    //     diag(0,2)). ---
    {
        CasscfCiConfig config;
        config.n_act_a = 1;
        config.davidson_roots = 2;
        config.davidson_threshold = 1e-10;
        config.davidson_indim = 2;
        config.davidson_maxdim = 2;
        config.davidson_maxiter = 200;
        config.target_spin = -1.0;

        CasscfPhysicalConstants constants = make_constants(2, 0.5);
        CasscfContext context = make_context(H_spatial2, dims);

        CasscfCiSetup setup(dims, config, constants, H_spatial2, context.J, context.K, /*E_core=*/0.0);

        CasscfCiStateAverageSolver solver(dims, config, constants, setup, context);
        Matrix eigenvecs_guess = Matrix::Zero(2, setup.H_dim());
        eigenvecs_guess(0, 0) = 1.0;
        eigenvecs_guess(1, setup.H_dim() - 1) = 1.0;

        CiStateAverageResult result = solver.solve(eigenvecs_guess);

        expect_near(result.ci_diagonalization_converged, 1.0, 0, "case2: Davidson converged");
        expect_near(result.eigenvalues(0), 0.0, tol, "case2: ground-state eigenvalue == 0.0");
        expect_near(result.eigenvalues(1), 2.0, tol, "case2: excited-state eigenvalue == 2.0");
        expect_near(result.avg_energy, 1.0, tol, "case2: avg_energy == 0.5*0 + 0.5*2 == 1.0");

        Matrix expected_D_tu = Matrix::Identity(2, 2);
        expect_matrix_near(result.D_tu_avg, expected_D_tu, tol, "case2: D_tu_avg == diag(1, 1)");
    }

    if (failures == 0) {
        std::printf("\nAll tests passed.\n");
        return 0;
    }
    std::printf("\n%d test(s) failed.\n", failures);
    return 1;
}
