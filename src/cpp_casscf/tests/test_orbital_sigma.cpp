// Tests for orbital_sigma3 (port of orbital_sigma3 -> build_sigma_reduced7,
// helper_PFCI.py:8285-8316, 8533-8686). No real captured Python data exists
// for this function yet (see orbital_sigma.hpp's doc comment), so this
// cross-validates the production implementation (in orbital_sigma.cpp,
// which reads directly off the raw G tensor and combines two of the
// Python's terms where they're provably additive over a disjoint-but-
// complementary range split) against a second, independently written
// reference implementation below that:
//   (a) materializes G_ij/G_ti/G_tu as literal separate matrices (matching
//       the Python's own intermediate arrays, rather than reading G
//       directly), and
//   (b) re-derives every step (including the ones orbital_sigma.cpp
//       combined) via a different algebraic route -- plain Eigen matrix
//       products instead of explicit index loops, and keeping the two
//       "combinable" terms of the d < n_in_a case genuinely separate.
// If a transcription or derivation error exists in either implementation,
// the two are unlikely to agree on random small problems by coincidence.
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

// Independent reference implementation -- see file header.
Vector reference_orbital_sigma3(const Matrix& U, const Matrix& A_tilde, const Tensor4& G, const Vector& R_reduced,
                                 const Dimensions& dims) {
    const int nmo = dims.nmo;
    const int n_occ = dims.n_occupied;
    const int n_in_a = dims.n_in_a;
    const int n_act = dims.n_act_orb;
    const auto index_map = build_index_map(dims);

    Matrix R_total = Matrix::Zero(nmo, n_occ);
    for (std::size_t j = 0; j < index_map.size(); ++j)
        R_total(index_map[j].first, index_map[j].second) = R_reduced(static_cast<int>(j));

    // temp1 (helper_PFCI.py:8566-8576), same formula as the production
    // code -- not the part under independent re-check here (see file
    // header: the G-block contraction and the A3_tilde terms are).
    Matrix temp1(nmo, n_occ);
    for (int r = 0; r < nmo; ++r) {
        for (int k = 0; k < n_occ; ++k) {
            double val = 0.0;
            for (int p = 0; p < nmo; ++p) val += U(r, p) * R_total(p, k);
            for (int l = 0; l < n_occ; ++l) val -= U(r, l) * R_total(k, l);
            temp1(r, k) = val;
        }
    }

    // Literal G_ij/G_ti/G_tu materialization, matching microiteration_optimization6's
    // own G1 = G.transpose(3,1,2,0) then block-slice-and-reshape
    // (helper_PFCI.py:11033-11045):
    //   G_ij[(a,b),(c,d)] = G[d,b,c,a],           a,c < nmo; b,d < n_in_a
    //   G_ti[(a,b),(c,d)] = G[d,n_in_a+b,c,a],    a,c < nmo; b < n_act; d < n_in_a
    //   G_tu[(a,b),(c,d)] = G[n_in_a+d,n_in_a+b,c,a], a,c < nmo; b,d < n_act
    Matrix G_ij(nmo * n_in_a, nmo * n_in_a);
    for (int a = 0; a < nmo; ++a)
        for (int b = 0; b < n_in_a; ++b)
            for (int c = 0; c < nmo; ++c)
                for (int d = 0; d < n_in_a; ++d) G_ij(a * n_in_a + b, c * n_in_a + d) = G(d, b, c, a);

    Matrix G_ti(nmo * n_act, nmo * n_in_a);
    for (int a = 0; a < nmo; ++a)
        for (int b = 0; b < n_act; ++b)
            for (int c = 0; c < nmo; ++c)
                for (int d = 0; d < n_in_a; ++d) G_ti(a * n_act + b, c * n_in_a + d) = G(d, n_in_a + b, c, a);

    Matrix G_tu(nmo * n_act, nmo * n_act);
    for (int a = 0; a < nmo; ++a)
        for (int b = 0; b < n_act; ++b)
            for (int c = 0; c < nmo; ++c)
                for (int d = 0; d < n_act; ++d) G_tu(a * n_act + b, c * n_act + d) = G(n_in_a + d, n_in_a + b, c, a);

    Vector R1_i(nmo * n_in_a);
    for (int a = 0; a < nmo; ++a)
        for (int b = 0; b < n_in_a; ++b) R1_i(a * n_in_a + b) = temp1(a, b);
    Vector R1_a(nmo * n_act);
    for (int a = 0; a < nmo; ++a)
        for (int b = 0; b < n_act; ++b) R1_a(a * n_act + b) = temp1(a, n_in_a + b);

    // sigma_i = R1_i @ G_ij + R1_a @ G_ti  (row-vector @ matrix ==
    // matrix^T @ column-vector).
    const Vector sigma_i = G_ij.transpose() * R1_i + G_ti.transpose() * R1_a;
    // sigma_a = R1_i @ G_ti.T + R1_a @ G_tu.
    const Vector sigma_a = G_ti * R1_i + G_tu.transpose() * R1_a;

    Matrix W(nmo, n_occ);
    for (int c = 0; c < nmo; ++c) {
        for (int d = 0; d < n_in_a; ++d) W(c, d) = sigma_i(c * n_in_a + d);
        for (int dd = 0; dd < n_act; ++dd) W(c, n_in_a + dd) = sigma_a(c * n_act + dd);
    }

    // Step 4 (helper_PFCI.py:8612-8618), re-derived via matrix algebra
    // instead of explicit loops: M4 = U^T @ W, main term is M4 itself; the
    // r < n_occupied correction term works out to M4(k, r).
    const Matrix M4 = U.transpose() * W;
    Matrix sigma_total(nmo, n_occ);
    for (int r = 0; r < nmo; ++r)
        for (int k = 0; k < n_occ; ++k) sigma_total(r, k) = M4(r, k) - (r < n_occ ? M4(k, r) : 0.0);

    // Step 5 (helper_PFCI.py:8620-8680), also re-derived via matrix
    // algebra: TA = A3_tilde @ R_total gives both the main term (TA(r,k))
    // and, for r < n_occupied, its "swapped" correction (TA(k,r)); TC/TD
    // are the two more asymmetric corrections.
    const Matrix A3_tilde = A_tilde + A_tilde.transpose();
    const Matrix TA = A3_tilde * R_total;                                         // (nmo, n_occ)
    const Matrix TC = R_total * A3_tilde.leftCols(n_occ).transpose();             // (nmo, nmo)
    const Matrix TD = A3_tilde.leftCols(n_occ) * R_total.transpose();             // (nmo, nmo)
    for (int r = 0; r < nmo; ++r) {
        for (int k = 0; k < n_occ; ++k) {
            double val = sigma_total(r, k);
            val -= 0.5 * TA(r, k);
            if (r < n_occ) val += 0.5 * TA(k, r);
            val -= 0.5 * TC(r, k);
            val += 0.5 * TD(r, k);
            sigma_total(r, k) = val;
        }
    }

    Vector sigma_reduced(static_cast<int>(index_map.size()));
    for (std::size_t j = 0; j < index_map.size(); ++j)
        sigma_reduced(static_cast<int>(j)) = sigma_total(index_map[j].first, index_map[j].second);
    return sigma_reduced;
}

Tensor4 random_tensor4(int d0, int d1, int d2, int d3, std::mt19937& rng) {
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    Tensor4 t(d0, d1, d2, d3);
    for (int i = 0; i < d0; ++i)
        for (int j = 0; j < d1; ++j)
            for (int k = 0; k < d2; ++k)
                for (int l = 0; l < d3; ++l) t(i, j, k, l) = dist(rng);
    return t;
}

Matrix random_matrix(int rows, int cols, std::mt19937& rng) {
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    Matrix m(rows, cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j) m(i, j) = dist(rng);
    return m;
}

void run_case(const Dimensions& dims, unsigned seed, const char* label) {
    std::mt19937 rng(seed);
    const Matrix U = random_matrix(dims.nmo, dims.nmo, rng);
    const Matrix A_tilde = random_matrix(dims.nmo, dims.nmo, rng);
    const Tensor4 G = random_tensor4(dims.n_occupied, dims.n_occupied, dims.nmo, dims.nmo, rng);
    const Vector R_reduced = random_matrix(dims.index_map_size(), 1, rng);

    const Vector production = orbital_sigma3(U, A_tilde, G, R_reduced, dims);
    const Vector reference = reference_orbital_sigma3(U, A_tilde, G, R_reduced, dims);

    expect_near((production - reference).norm(), 0.0, 1e-10, label);

    // FAST path (OrbitalSigmaOperator / orbital_sigma3_fast) vs the legacy
    // loop production version -- same math, BLAS-backed evaluation with the G
    // blocks precomputed once. Checked both via the one-shot convenience
    // wrapper and via a reused operator applied to two different vectors
    // (the reuse path is the one production actually exercises: build once,
    // apply many times per solve).
    const Vector fast = orbital_sigma3_fast(U, A_tilde, G, R_reduced, dims);
    expect_near((fast - production).norm(), 0.0, 1e-10, label);

    OrbitalSigmaOperator op(U, A_tilde, G, dims);
    expect_near((op.apply(R_reduced) - production).norm(), 0.0, 1e-10, label);
    // A second, independent vector through the SAME operator -- guards
    // against the operator accidentally caching anything vector-dependent.
    std::mt19937 rng2(seed ^ 0xabcdu);
    const Vector R2 = random_matrix(dims.index_map_size(), 1, rng2);
    expect_near((op.apply(R2) - orbital_sigma3(U, A_tilde, G, R2, dims)).norm(), 0.0, 1e-10, label);
}

} // namespace

int main() {
    // --- Case 1: small, all three blocks (inactive/active/virtual) nonempty. ---
    {
        Dimensions dims;
        dims.n_in_a = 2;
        dims.n_act_orb = 2;
        dims.n_virtual = 2;
        dims.nmo = 6;
        dims.n_occupied = 4;
        run_case(dims, 12345u, "case1: production matches independent reference (n_in_a=2,n_act=2,n_virtual=2)");
    }

    // --- Case 2: n_in_a == 0 (exercises the d < n_in_a loop range being
    //     empty -- the whole G_ij/inactive-block machinery degenerates). ---
    {
        Dimensions dims;
        dims.n_in_a = 0;
        dims.n_act_orb = 2;
        dims.n_virtual = 2;
        dims.nmo = 4;
        dims.n_occupied = 2;
        run_case(dims, 999u, "case2: production matches independent reference (n_in_a=0)");
    }

    // --- Case 3: n_virtual == 0 (nmo == n_occupied). ---
    {
        Dimensions dims;
        dims.n_in_a = 1;
        dims.n_act_orb = 2;
        dims.n_virtual = 0;
        dims.nmo = 3;
        dims.n_occupied = 3;
        run_case(dims, 42u, "case3: production matches independent reference (n_virtual=0)");
    }

    // --- Case 4: minimal nontrivial dims (n_in_a=1, n_act=1, n_virtual=1). ---
    {
        Dimensions dims;
        dims.n_in_a = 1;
        dims.n_act_orb = 1;
        dims.n_virtual = 1;
        dims.nmo = 3;
        dims.n_occupied = 2;
        run_case(dims, 7u, "case4: production matches independent reference (minimal dims)");
    }

    // --- Case 5: a larger case (nmo=20, n_occ=8), representative of the
    //     performance regime the fast path exists for -- exercises the
    //     BLAS-backed matmuls at a nontrivial size and confirms the fast/loop
    //     agreement holds beyond the tiny hand cases above. ---
    {
        Dimensions dims;
        dims.n_in_a = 4;
        dims.n_act_orb = 4;
        dims.n_virtual = 12;
        dims.nmo = 20;
        dims.n_occupied = 8;
        run_case(dims, 20240720u, "case5: fast/loop agree at a larger size (nmo=20, n_occ=8)");
    }

    if (failures == 0) {
        std::printf("\nAll tests passed.\n");
        return 0;
    }
    std::printf("\n%d test(s) failed.\n", failures);
    return 1;
}
