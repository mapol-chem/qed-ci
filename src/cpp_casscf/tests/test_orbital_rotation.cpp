// Tests for build_unitary_matrix (port of helper_PFCI.py:6131-6164). Not a
// port of any specific Python test -- these check the mathematical
// properties the function must satisfy (orthogonality, identity on zero
// input, exact agreement with the closed-form 2D rotation, and that the
// small-angle series branch agrees with the general formula) rather than
// reproducing a Python-side numeric fixture.
#include "casscf/orbital_rotation.hpp"

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

void expect_orthogonal(const Matrix& U, double tol, const char* label) {
    Matrix should_be_identity = U.transpose() * U;
    double err = (should_be_identity - Matrix::Identity(U.rows(), U.rows())).norm();
    expect_near(err, 0.0, tol, label);
}

} // namespace

int main() {
    const double tol = 1e-9;

    // --- Case 1: all rotation blocks zero -> U_delta is the identity ---
    {
        Dimensions dims;
        dims.n_in_a = 2;
        dims.n_act_orb = 2;
        dims.n_virtual = 2;
        dims.nmo = 6;
        dims.n_occupied = 4;

        Matrix Rai = Matrix::Zero(dims.n_act_orb, dims.n_in_a);
        Matrix Rvi = Matrix::Zero(dims.n_virtual, dims.n_in_a);
        Matrix Rva = Matrix::Zero(dims.n_virtual, dims.n_act_orb);

        Matrix U = build_unitary_matrix(Rai, Rvi, Rva, dims);
        double err = (U - Matrix::Identity(dims.nmo, dims.nmo)).norm();
        expect_near(err, 0.0, tol, "zero rotation: U_delta == identity");
    }

    // --- Case 2: single active-inactive angle reduces to the closed-form
    //     2x2 rotation matrix (R = theta * [[0,-1],[1,0]] is exactly
    //     solvable: exp(R) = [[cos theta, -sin theta],[sin theta, cos
    //     theta]], regardless of the degenerate eigenbasis build_unitary_matrix
    //     happens to pick for -R^2 = theta^2 * I) ---
    {
        Dimensions dims;
        dims.n_in_a = 1;
        dims.n_act_orb = 1;
        dims.n_virtual = 0;
        dims.nmo = 2;
        dims.n_occupied = 2;

        const double theta = 0.7;
        Matrix Rai(1, 1);
        Rai(0, 0) = theta;
        Matrix Rvi(0, 1);
        Matrix Rva(0, 1);

        Matrix U = build_unitary_matrix(Rai, Rvi, Rva, dims);
        expect_near(U(0, 0), std::cos(theta), tol, "2x2 rotation: U(0,0) == cos(theta)");
        expect_near(U(0, 1), -std::sin(theta), tol, "2x2 rotation: U(0,1) == -sin(theta)");
        expect_near(U(1, 0), std::sin(theta), tol, "2x2 rotation: U(1,0) == sin(theta)");
        expect_near(U(1, 1), std::cos(theta), tol, "2x2 rotation: U(1,1) == cos(theta)");
    }

    // --- Case 3: tiny angle exercises the small-angle series fallback
    //     (tau < 1e-15 branch never actually triggers for a nonzero theta,
    //     but this checks the direct trig path stays accurate down to a
    //     very small rotation, and separately that a genuinely zero
    //     direction -- see Case 4 -- hits the series exactly) ---
    {
        Dimensions dims;
        dims.n_in_a = 1;
        dims.n_act_orb = 1;
        dims.n_virtual = 0;
        dims.nmo = 2;
        dims.n_occupied = 2;

        const double theta = 1e-8;
        Matrix Rai(1, 1);
        Rai(0, 0) = theta;
        Matrix Rvi(0, 1);
        Matrix Rva(0, 1);

        Matrix U = build_unitary_matrix(Rai, Rvi, Rva, dims);
        expect_near(U(0, 0), std::cos(theta), tol, "tiny rotation: U(0,0) == cos(theta)");
        expect_near(U(1, 0), std::sin(theta), tol, "tiny rotation: U(1,0) == sin(theta)");
    }

    // --- Case 4: a fully decoupled virtual block (Rvi == Rva == 0, but
    //     nonzero Rai) forces exact zero eigenvalues of -R^2 restricted to
    //     the virtual subspace, hitting the tau < 1e-15 series branch for
    //     real (not just numerically small) tau. The virtual block of
    //     U_delta must come out as the identity, and the in_a/act block
    //     must match the same Rai run through the n_virtual == 0 case. ---
    {
        Dimensions dims_with_virtual;
        dims_with_virtual.n_in_a = 2;
        dims_with_virtual.n_act_orb = 2;
        dims_with_virtual.n_virtual = 3;
        dims_with_virtual.nmo = 7;
        dims_with_virtual.n_occupied = 4;

        Matrix Rai(2, 2);
        Rai << 0.3, -0.1,
               0.2, 0.4;
        Matrix Rvi = Matrix::Zero(dims_with_virtual.n_virtual, dims_with_virtual.n_in_a);
        Matrix Rva = Matrix::Zero(dims_with_virtual.n_virtual, dims_with_virtual.n_act_orb);

        Matrix U = build_unitary_matrix(Rai, Rvi, Rva, dims_with_virtual);

        Matrix virtual_block = U.block(dims_with_virtual.n_occupied, dims_with_virtual.n_occupied,
                                        dims_with_virtual.n_virtual, dims_with_virtual.n_virtual);
        double virtual_err = (virtual_block - Matrix::Identity(dims_with_virtual.n_virtual,
                                                                 dims_with_virtual.n_virtual)).norm();
        expect_near(virtual_err, 0.0, tol, "decoupled virtual block: identity");

        double off_block_err = U.block(0, dims_with_virtual.n_occupied, dims_with_virtual.n_occupied,
                                        dims_with_virtual.n_virtual).norm();
        expect_near(off_block_err, 0.0, tol, "decoupled virtual block: no in_a/act <-> virtual mixing");

        Dimensions dims_no_virtual;
        dims_no_virtual.n_in_a = 2;
        dims_no_virtual.n_act_orb = 2;
        dims_no_virtual.n_virtual = 0;
        dims_no_virtual.nmo = 4;
        dims_no_virtual.n_occupied = 4;
        Matrix Rvi_empty(0, 2);
        Matrix Rva_empty(0, 2);
        Matrix U_reference = build_unitary_matrix(Rai, Rvi_empty, Rva_empty, dims_no_virtual);

        Matrix in_act_block = U.block(0, 0, 4, 4);
        double reference_err = (in_act_block - U_reference).norm();
        expect_near(reference_err, 0.0, tol, "decoupled virtual block: in_a/act block matches n_virtual=0 case");
    }

    // --- Case 5: general random-ish rotation -- U_delta must be orthogonal
    //     (exp of a real antisymmetric generator is always in O(n), and in
    //     fact SO(n) since it's a matrix exponential / connected to the
    //     identity) ---
    {
        Dimensions dims;
        dims.n_in_a = 2;
        dims.n_act_orb = 2;
        dims.n_virtual = 2;
        dims.nmo = 6;
        dims.n_occupied = 4;

        Matrix Rai(2, 2);
        Rai << 0.10, -0.05,
               0.20, 0.15;
        Matrix Rvi(2, 2);
        Rvi << -0.08, 0.12,
               0.03, -0.11;
        Matrix Rva(2, 2);
        Rva << 0.25, 0.02,
               -0.06, 0.18;

        Matrix U = build_unitary_matrix(Rai, Rvi, Rva, dims);
        expect_orthogonal(U, 1e-8, "general rotation: U_delta^T U_delta == I");
        expect_near(U.determinant(), 1.0, 1e-6, "general rotation: det(U_delta) == 1");
    }

    if (failures == 0) {
        std::printf("\nAll tests passed.\n");
        return 0;
    }
    std::printf("\n%d test(s) failed.\n", failures);
    return 1;
}
