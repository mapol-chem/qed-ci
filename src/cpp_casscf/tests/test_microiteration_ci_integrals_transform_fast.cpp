// Cross-validates microiteration_ci_integrals_transform_fast (BLAS-backed,
// symmetry-reduced) against the legacy explicit-loop
// microiteration_ci_integrals_transform, which is the correctness oracle.
//
// IMPORTANT -- this test differs from test_intermediates_fast.cpp and
// test_intermediates_internal_fast.cpp in one deliberate way. Those fill their
// J/K with independent random numbers, because those fast paths are exact
// algebraic rearrangements valid for arbitrary inputs. This one is NOT: its
// active_twoeint transforms exploit the physical permutational symmetries of
// the ERIs (see the header for which, and for where they were verified on real
// dumps). Feeding it independently-random J/K would make the two paths
// genuinely disagree -- a spurious failure, not a bug.
//
// So the inputs here are built the way real ones are: from a single random
// (nmo)^4 tensor g that is explicitly symmetrized to full 8-fold ERI symmetry,
// then sliced into the code's own two conventions
//     J(k,l,r,s) = (rs|kl) = g(r,s,k,l)
//     K(k,l,r,s) = (rk|sl) = g(r,k,s,l)
// This is stronger than symmetrizing J and K independently: it guarantees they
// are mutually consistent, i.e. derived from ONE integral set, exactly as the
// real ones are. Everything else (U, fock_core, L, d_cmo_ref,
// active_twoeint_ref) stays unsymmetrized random, so any accidental transpose
// in the non-ERI parts still shows up immediately.
//
// L is built from J/K by its real defining formula (4K - K^transpose - J)
// rather than filled randomly, for the same consistency reason.
#include "casscf/microiteration_ci_integrals_transform.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace casscf;

namespace {

int failures = 0;

double max_abs_diff(const Tensor4& a, const Tensor4& b) {
    double worst = 0.0;
    for (Eigen::Index i = 0; i < a.size(); ++i) worst = std::max(worst, std::abs(a.data()[i] - b.data()[i]));
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

    // Random (nmo)^4 tensor symmetrized to full 8-fold ERI symmetry:
    //   g(p,q,r,s) == g(q,p,r,s) == g(p,q,s,r) == g(r,s,p,q)
    std::vector<double> g(static_cast<std::size_t>(nmo) * nmo * nmo * nmo);
    auto gidx = [&](int p, int q, int r, int s) {
        return ((static_cast<std::size_t>(p) * nmo + q) * nmo + r) * nmo + s;
    };
    for (auto& x : g) x = dist(rng);
    std::vector<double> gs(g.size());
    for (int p = 0; p < nmo; ++p)
        for (int q = 0; q < nmo; ++q)
            for (int r = 0; r < nmo; ++r)
                for (int s = 0; s < nmo; ++s)
                    gs[gidx(p, q, r, s)] =
                        (g[gidx(p, q, r, s)] + g[gidx(q, p, r, s)] + g[gidx(p, q, s, r)] + g[gidx(q, p, s, r)] +
                         g[gidx(r, s, p, q)] + g[gidx(s, r, p, q)] + g[gidx(r, s, q, p)] + g[gidx(s, r, q, p)]) /
                        8.0;

    // J(k,l,r,s) = (rs|kl), K(k,l,r,s) = (rk|sl) -- the conventions this code
    // uses (optimization_note.md: J_rs^kl = (rs|kl), K_rs^kl = (rk|sl)).
    Tensor4 J(n_occ, n_occ, nmo, nmo), K(n_occ, n_occ, nmo, nmo);
    for (int k = 0; k < n_occ; ++k)
        for (int l = 0; l < n_occ; ++l)
            for (int r = 0; r < nmo; ++r)
                for (int s = 0; s < nmo; ++s) {
                    J(k, l, r, s) = gs[gidx(r, s, k, l)];
                    K(k, l, r, s) = gs[gidx(r, k, s, l)];
                }

    // L = 4K - K^T(last two axes) - J, its real defining formula.
    Tensor4 L(n_occ, n_in_a, nmo, nmo);
    for (int p = 0; p < n_occ; ++p)
        for (int j = 0; j < n_in_a; ++j)
            for (int r = 0; r < nmo; ++r)
                for (int s = 0; s < nmo; ++s) L(p, j, r, s) = 4.0 * K(p, j, r, s) - K(p, j, s, r) - J(p, j, r, s);

    auto fill_m = [&](Matrix& m) {
        for (int i = 0; i < m.rows(); ++i)
            for (int j = 0; j < m.cols(); ++j) m(i, j) = dist(rng);
    };

    // U is a general random matrix, not orthogonal: nothing in either path
    // requires orthogonality, so leaving it general is the stronger test.
    Matrix U(nmo, nmo), d_cmo_ref(nmo, nmo);
    fill_m(U);
    fill_m(d_cmo_ref);

    // fock_core is symmetric in reality; keep it so, since the fast path's
    // term-1 congruence transform is where a wrong transpose would hide, and
    // an artificially asymmetric fock_core would flag a difference that cannot
    // occur with real input.
    Matrix fock_core(nmo, nmo);
    fill_m(fock_core);
    fock_core = ((fock_core + fock_core.transpose()) * 0.5).eval();

    Tensor4 active_twoeint_ref(n_act, n_act, n_act, n_act);
    for (Eigen::Index i = 0; i < active_twoeint_ref.size(); ++i) active_twoeint_ref.data()[i] = dist(rng);

    const double E_core_ref = 0.731;

    const MicroiterationCiIntegralsResult legacy = microiteration_ci_integrals_transform(
        U, E_core_ref, fock_core, L, J, K, active_twoeint_ref, d_cmo_ref, dims);
    const MicroiterationCiIntegralsResult fast = microiteration_ci_integrals_transform_fast(
        U, E_core_ref, fock_core, L, J, K, active_twoeint_ref, d_cmo_ref, dims);

    // Loose relative to the ~1e-13 absolute agreement actually observed. These
    // quantities are sums of O(nmo^2 * n_act^2) products of O(1) randoms, so
    // their magnitudes run to ~1e2-1e3 here; the tolerance is set against that
    // scale, not against 1.
    const double tol = 1e-9;
    std::printf("-- case %s (n_in_a=%d, n_act=%d, n_virt=%d, nmo=%d)\n", name, n_in_a, n_act, n_virt, nmo);
    expect_below(std::abs(legacy.E_core2 - fast.E_core2), tol, "E_core2");
    expect_below(max_abs_diff(legacy.active_fock_core, fast.active_fock_core), tol, "active_fock_core");
    expect_below(max_abs_diff(legacy.active_twoeint, fast.active_twoeint), tol, "active_twoeint");
    expect_below(max_abs_diff(legacy.d_cmo, fast.d_cmo), tol, "d_cmo");
}

} // namespace

int main() {
    // No two of n_in_a / n_act / n_virt coincide within a case, so an index
    // swap between them cannot hide; plus n_in_a == 0, which empties the
    // inactive density matrix D and every n_in_a-strided loop.
    run_case(2, 3, 4, 12345u, "asymmetric");
    run_case(3, 2, 5, 999u, "act<inactive");
    run_case(1, 4, 2, 4242u, "single-inactive");
    run_case(0, 3, 3, 777u, "no-inactive");
    run_case(4, 5, 3, 31337u, "larger");

    if (failures == 0) {
        std::printf("\nAll microiteration_ci_integrals_transform_fast cross-validation checks passed.\n");
        return 0;
    }
    std::printf("\n%d check(s) FAILED.\n", failures);
    return 1;
}
