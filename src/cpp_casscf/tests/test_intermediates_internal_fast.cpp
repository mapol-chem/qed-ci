// Cross-validates build_intermediates_internal_fast (BLAS-backed) against the
// legacy explicit-loop build_intermediates_internal, which is the correctness
// oracle -- same arrangement as test_intermediates_fast.cpp does for the
// full-block twin, and test_orbital_sigma.cpp for orbital_sigma3.
//
// The fast path is a term-by-term transformation of the legacy loops, so
// agreement here checks the transforms (index packing, storage order, GEMM
// operand construction), not the physics. Storage-order assumptions are the
// main risk -- the fast path maps the trailing (r,s) plane of a RowMajor
// Tensor4 directly as a matrix and reshapes D_tuvw_avg's buffer as
// (n_act^2, n_act^2) with no copy -- so every input below is filled with
// distinct random values and deliberately NOT symmetrized: any accidental
// transpose then shows up immediately.
//
// The term this test exists to protect most specifically is `A`'s
// active-active two-electron contraction. build_intermediates uses
// "vwrt,tuvw->ru" (r third in J) while build_intermediates_internal uses
// "rtvw,tuvw->ru" (r LEADING in occupied_J) -- genuinely different index
// patterns, forced by the two functions' different J shapes. The fast path
// therefore contracts `P * Q` where the full-block twin needs
// `P.transpose() * Q`. Getting that backwards is the most plausible way to
// break this function, and an asymmetric occupied_J makes it visible.
#include "casscf/intermediates.hpp"

#include <cmath>
#include <cstdio>
#include <random>

using namespace casscf;

namespace {

int failures = 0;

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

void run_case(int n_in_a, int n_act, unsigned seed, const char* name) {
    Dimensions dims;
    dims.n_in_a = n_in_a;
    dims.n_act_orb = n_act;
    dims.n_occupied = n_in_a + n_act;
    // nmo is not used by build_intermediates_internal (rot_dim == n_occupied),
    // but is set to something distinct from n_occupied so that any accidental
    // use of dims.nmo in the fast path would produce a shape mismatch rather
    // than silently agreeing.
    dims.nmo = n_in_a + n_act + 3;

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

    Matrix occupied_fock_core(n_occ, n_occ), occupied_d_cmo(n_occ, n_occ);
    fill_m(occupied_fock_core);
    fill_m(occupied_d_cmo);

    // Fully occupied-restricted (n_occupied)^4 blocks -- unlike the full-block
    // twin's J/K, which are occupied-restricted in the first two axes only.
    Tensor4 occupied_J(n_occ, n_occ, n_occ, n_occ), occupied_K(n_occ, n_occ, n_occ, n_occ);
    fill_t(occupied_J);
    fill_t(occupied_K);

    Matrix D_tu_avg(n_act, n_act), Dpe_tu_avg(n_act, n_act);
    fill_m(D_tu_avg);
    fill_m(Dpe_tu_avg);
    Tensor4 D_tuvw_avg(n_act, n_act, n_act, n_act);
    fill_t(D_tuvw_avg);

    // omega deliberately nonzero so the -sqrt(omega/2) photon terms (which
    // vanish identically at omega == 0) are actually exercised.
    const double omega = 0.37;
    const double off_diagonal_constant = -0.21;

    const SmallBlockIntermediates legacy =
        build_intermediates_internal(occupied_fock_core, occupied_d_cmo, occupied_J, occupied_K, D_tu_avg,
                                     D_tuvw_avg, Dpe_tu_avg, off_diagonal_constant, omega, dims);
    const SmallBlockIntermediates fast =
        build_intermediates_internal_fast(occupied_fock_core, occupied_d_cmo, occupied_J, occupied_K, D_tu_avg,
                                          D_tuvw_avg, Dpe_tu_avg, off_diagonal_constant, omega, dims);

    // Loose relative to the ~1e-16 elementwise agreement actually observed:
    // the two paths sum the same terms in different orders, so exact equality
    // is not expected, but anything above this is a real discrepancy rather
    // than reassociation noise.
    const double tol = 1e-11;
    std::printf("-- case %s (n_in_a=%d, n_act=%d, n_occupied=%d)\n", name, n_in_a, n_act, n_occ);
    expect_below(max_abs_diff(legacy.A, fast.A), tol, "A");
    expect_below(max_abs_diff(legacy.G, fast.G), tol, "G");
}

} // namespace

int main() {
    // Shapes chosen so n_in_a != n_act within a case (an index swap between
    // the two would otherwise be invisible), plus the degenerate n_in_a == 0
    // case, where the inactive blocks and every n_in_a-strided packing become
    // empty -- including the Lact map, whose leading offset is n_in_a-scaled.
    run_case(2, 3, 12345u, "asymmetric");
    run_case(3, 2, 999u, "act<inactive");
    run_case(1, 4, 4242u, "single-inactive");
    run_case(0, 3, 777u, "no-inactive");
    run_case(4, 5, 31337u, "larger");
    run_case(5, 8, 24680u, "cas-8-12-shaped");

    if (failures == 0) {
        std::printf("\nAll build_intermediates_internal_fast cross-validation checks passed.\n");
        return 0;
    }
    std::printf("\n%d check(s) FAILED.\n", failures);
    return 1;
}
