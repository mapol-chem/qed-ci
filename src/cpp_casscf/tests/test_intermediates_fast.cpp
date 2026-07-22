// Cross-validates build_intermediates_fast (BLAS-backed) against the legacy
// explicit-loop build_intermediates, which is the correctness oracle -- the
// same arrangement test_orbital_sigma.cpp uses for orbital_sigma3 vs
// OrbitalSigmaOperator.
//
// The fast path is a term-by-term transformation of the legacy loops (not an
// independent re-derivation), so agreement here is checking the transforms --
// index packing, storage order, GEMM operand construction -- not the physics.
// That makes the storage-order assumptions the main risk: the fast path maps
// the trailing (r,s) plane of a RowMajor Tensor4 directly as a matrix and
// reshapes D_tuvw_avg's buffer as (n_act^2, n_act^2) with no copy. A wrong
// assumption there transposes a slab, which random asymmetric inputs detect
// immediately -- hence every input below is filled with distinct random values
// and NOT symmetrized, so that A(r,s) != A(s,r) and any accidental transpose
// shows up.
//
// Several dimension shapes are exercised because a number of the packings
// differ only in stride arithmetic (n_in_a vs n_act vs n_occupied), and a
// shape where two of those happen to be equal would hide an index swap.
#include "casscf/intermediates.hpp"

#include <cmath>
#include <cstdio>
#include <random>

using namespace casscf;

namespace {

int failures = 0;

// Largest elementwise deviation between two same-shaped tensors/matrices.
double max_abs_diff(const Tensor4& a, const Tensor4& b) {
    double worst = 0.0;
    const Eigen::Index n = a.size();
    for (Eigen::Index i = 0; i < n; ++i) worst = std::max(worst, std::abs(a.data()[i] - b.data()[i]));
    return worst;
}

double max_abs_diff(const Matrix& a, const Matrix& b) { return (a - b).cwiseAbs().maxCoeff(); }

void expect_below(double actual, double tol, const char* label) {
    if (!(actual <= tol)) {
        std::printf("FAIL: %s -- max|diff| = %.3e exceeds %.3e\n", label, actual, tol);
        ++failures;
    } else {
        std::printf("PASS: %s (max|diff| = %.3e)\n", label, actual);
    }
}

void run_case(int n_in_a, int n_act, int n_virt, unsigned seed, const char* name) {
    Dimensions dims;
    dims.n_in_a = n_in_a;
    dims.n_act_orb = n_act;
    dims.n_occupied = n_in_a + n_act;
    dims.nmo = n_in_a + n_act + n_virt;

    const int nmo = dims.nmo;
    const int n_occ = dims.n_occupied;

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    auto fill_m = [&](Matrix& m) {
        for (int i = 0; i < m.rows(); ++i)
            for (int j = 0; j < m.cols(); ++j) m(i, j) = dist(rng);
    };
    auto fill_t = [&](Tensor4& t) {
        for (Eigen::Index i = 0; i < t.size(); ++i) t.data()[i] = dist(rng);
    };

    Matrix H_spatial2(nmo, nmo), d_cmo(nmo, nmo);
    fill_m(H_spatial2);
    fill_m(d_cmo);

    // J/K are (n_occupied, n_occupied, nmo, nmo) -- see intermediates.hpp's
    // note that these are occupied-restricted in the first two axes only.
    Tensor4 J(n_occ, n_occ, nmo, nmo), K(n_occ, n_occ, nmo, nmo);
    fill_t(J);
    fill_t(K);

    Matrix D_tu_avg(n_act, n_act), Dpe_tu_avg(n_act, n_act);
    fill_m(D_tu_avg);
    fill_m(Dpe_tu_avg);
    Tensor4 D_tuvw_avg(n_act, n_act, n_act, n_act);
    fill_t(D_tuvw_avg);

    // omega deliberately nonzero so the -sqrt(omega/2) photon terms (which
    // vanish identically at omega == 0) are actually exercised.
    const double omega = 0.37;
    const double off_diagonal_constant = -0.21;

    const FullBlockIntermediates legacy = build_intermediates(
        H_spatial2, d_cmo, J, K, D_tu_avg, D_tuvw_avg, Dpe_tu_avg, off_diagonal_constant, omega, dims);
    const FullBlockIntermediates fast = build_intermediates_fast(
        H_spatial2, d_cmo, J, K, D_tu_avg, D_tuvw_avg, Dpe_tu_avg, off_diagonal_constant, omega, dims);

    // Tolerance is loose relative to the ~1e-16 elementwise agreement actually
    // observed: the two paths sum the same terms in different orders, so exact
    // equality is not expected, but anything above this is a real discrepancy
    // rather than reassociation noise.
    const double tol = 1e-11;
    std::printf("-- case %s (n_in_a=%d, n_act=%d, n_virt=%d, nmo=%d)\n", name, n_in_a, n_act, n_virt, nmo);
    expect_below(max_abs_diff(legacy.A, fast.A), tol, "A");
    expect_below(max_abs_diff(legacy.G, fast.G), tol, "G");
    expect_below(max_abs_diff(legacy.fock_core, fast.fock_core), tol, "fock_core");
    expect_below(max_abs_diff(legacy.L, fast.L), tol, "L");
    expect_below(max_abs_diff(legacy.active_fock_core, fast.active_fock_core), tol, "active_fock_core");
    expect_below(max_abs_diff(legacy.active_twoeint, fast.active_twoeint), tol, "active_twoeint");
    expect_below(std::abs(legacy.E_core - fast.E_core), tol, "E_core");
}

} // namespace

int main() {
    // Shapes chosen so no two of n_in_a / n_act / n_virt coincide within a
    // case (an index swap between them would otherwise be invisible), plus
    // the degenerate n_in_a == 0 case, where the inactive blocks and every
    // n_in_a-strided packing become empty.
    run_case(2, 3, 4, 12345u, "asymmetric");
    run_case(3, 2, 5, 999u, "act<inactive");
    run_case(1, 4, 2, 4242u, "single-inactive");
    run_case(0, 3, 3, 777u, "no-inactive");
    run_case(4, 5, 6, 31337u, "larger");

    if (failures == 0) {
        std::printf("\nAll build_intermediates_fast cross-validation checks passed.\n");
        return 0;
    }
    std::printf("\n%d check(s) FAILED.\n", failures);
    return 1;
}
