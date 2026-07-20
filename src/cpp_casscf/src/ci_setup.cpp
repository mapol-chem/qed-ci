#include "casscf/ci_setup.hpp"

#include "casscf/ci_orbital_backend.hpp"

#include <algorithm>
#include <numeric>

namespace casscf {
namespace {

// C++17 has no std::comb (C++20 doesn't add one either, only <ranges>-style
// utilities elsewhere) -- math.comb(n, k), computed with the usual
// multiplicative recurrence rather than a raw factorial to avoid overflow
// at these (orbital-count) sizes.
long long binomial_coefficient(int n, int k) {
    if (k < 0 || k > n) return 0;
    k = std::min(k, n - k);
    long long result = 1;
    for (int i = 0; i < k; ++i) {
        result = result * (n - i) / (i + 1);
    }
    return result;
}

} // namespace

ActiveBlockIntermediates compute_active_block_intermediates(const Matrix& H_spatial2, const Tensor4& J,
                                                               const Tensor4& K, const Dimensions& dims) {
    const int nmo = dims.nmo;
    const int n_in_a = dims.n_in_a;
    const int n_act = dims.n_act_orb;

    // fock_core(r,s) = H_spatial2(r,s) + sum_{j<n_in_a}[2*J(j,j,r,s) - K(j,j,r,s)]
    // helper_PFCI.py:2455-2461 (the per-macroiteration block) / 5754-5760
    // (build_intermediates's own identical internal computation, see
    // ci_setup.hpp's doc comment for why this is transcribed fresh here
    // rather than reusing build_intermediates).
    Matrix fock_core = H_spatial2;
    for (int r = 0; r < nmo; ++r) {
        for (int s = 0; s < nmo; ++s) {
            double acc = 0.0;
            for (int j = 0; j < n_in_a; ++j) acc += 2.0 * J(j, j, r, s) - K(j, j, r, s);
            fock_core(r, s) += acc;
        }
    }

    ActiveBlockIntermediates result;
    result.fock_core = fock_core;
    {
        double e_core = 0.0;
        for (int j = 0; j < n_in_a; ++j) e_core += H_spatial2(j, j) + fock_core(j, j);
        result.E_core = e_core;
    }
    result.active_fock_core = fock_core.block(n_in_a, n_in_a, n_act, n_act);
    result.active_twoeint = Tensor4(n_act, n_act, n_act, n_act);
    for (int t = 0; t < n_act; ++t)
        for (int u = 0; u < n_act; ++u)
            for (int v = 0; v < n_act; ++v)
                for (int w = 0; w < n_act; ++w)
                    result.active_twoeint(t, u, v, w) = J(n_in_a + t, n_in_a + u, n_in_a + v, n_in_a + w);
    return result;
}

OccupiedCiBlocks build_occupied_ci_blocks(const ActiveBlockIntermediates& active, const Dimensions& dims) {
    const int n_in_a = dims.n_in_a;
    const int n_act = dims.n_act_orb;
    const int n_occ = dims.n_occupied;

    OccupiedCiBlocks result;
    result.occupied_fock_core = Matrix::Zero(n_occ, n_occ);
    result.occupied_fock_core.block(n_in_a, n_in_a, n_act, n_act) = active.active_fock_core;

    result.occupied_twoeint = Tensor4(n_occ, n_occ, n_occ, n_occ);
    result.occupied_twoeint.setZero();
    for (int t = 0; t < n_act; ++t)
        for (int u = 0; u < n_act; ++u)
            for (int v = 0; v < n_act; ++v)
                for (int w = 0; w < n_act; ++w)
                    result.occupied_twoeint(n_in_a + t, n_in_a + u, n_in_a + v, n_in_a + w) =
                        active.active_twoeint(t, u, v, w);
    return result;
}

CasscfCiSetup::CasscfCiSetup(Dimensions dims, CasscfCiConfig config, CasscfPhysicalConstants constants,
                               const Matrix& H_spatial2, const Tensor4& J, const Tensor4& K, double E_core) {
    const int n_act_a = config.n_act_a;
    const int n_act_orb = dims.n_act_orb;
    const int n_in_a = dims.n_in_a;
    const int nmo = dims.nmo;
    const int N_p = constants.N_p;

    num_alpha_ = static_cast<int>(binomial_coefficient(n_act_orb, n_act_a));
    const int num_det = num_alpha_ * num_alpha_;
    h_dim_ = num_det * (N_p + 1);
    indim_ = config.davidson_indim * config.davidson_roots;
    maxdim_ = config.davidson_maxdim * config.davidson_roots;

    // helper_PFCI.py:1507-1526 -- table/table_creation/table_annihilation/
    // b_array/Y allocation formulas.
    table_.assign(static_cast<size_t>(num_alpha_) * (n_act_a * (n_act_orb - n_act_a) + n_act_a) * 4, 0);
    const int num_links1 = n_act_orb - n_act_a + 1;
    const long long rows1 = binomial_coefficient(n_act_orb, n_act_a - 1) * num_links1;
    table_creation_.assign(static_cast<size_t>(rows1) * 3, 0);
    const long long rows2 = static_cast<long long>(num_alpha_) * n_act_a;
    table_annihilation_.assign(static_cast<size_t>(rows2) * 3, 0);
    b_array_.assign(static_cast<size_t>(num_alpha_) * n_act_orb * n_act_orb * 2, 0);
    Y_.assign(static_cast<size_t>(n_act_a) * (n_act_orb - n_act_a + 1) * 3, 0);

    // get_graph(N, n_o, Y) -- helper_PFCI.py:1527 (c_graph(n_act_a, n_act_orb, self.Y)).
    get_graph(static_cast<size_t>(n_act_a), static_cast<size_t>(n_act_orb), Y_.data());

    ActiveBlockIntermediates active = compute_active_block_intermediates(H_spatial2, J, K, dims);
    OccupiedCiBlocks occupied = build_occupied_ci_blocks(active, dims);
    RowMajorMatrix occupied_fock_core_rm = occupied.occupied_fock_core;

    // get_string -- helper_PFCI.py:1570-1589. H_diag is a genuine output
    // parameter but deliberately discarded here -- see this class's own
    // header doc comment for why self.H_diag is dead weight for this port.
    Vector H_diag_unused = Vector::Zero(h_dim_);
    get_string(occupied_fock_core_rm.data(), occupied.occupied_twoeint.data(), H_diag_unused.data(),
                b_array_.data(), table_.data(), table_creation_.data(), table_annihilation_.data(), N_p,
                num_alpha_, nmo, n_act_a, n_act_orb, n_in_a, E_core, constants.omega, constants.Enuc,
                constants.d_c, config.target_spin);

    // build_H_diag_cas_spin -- helper_PFCI.py:1549-1565 -- only needed here
    // to derive index_Hdiag (see this class's header doc comment: this is
    // the ONE-TIME, pre-optimization H_diag3 that index_Hdiag is frozen
    // against for the whole run, not the per-macroiteration one
    // CasscfCiStateAverageSolver rebuilds fresh every call).
    Vector H_diag3_initial = Vector::Zero(h_dim_);
    build_H_diag_cas_spin(occupied_fock_core_rm.data(), occupied.occupied_twoeint.data(),
                            H_diag3_initial.data(), N_p, num_alpha_, nmo, n_act_a, n_act_orb, n_in_a, E_core,
                            constants.omega, constants.Enuc, constants.d_c, Y_.data(), config.target_spin);

    // helper_PFCI.py:1567 -- self.index_Hdiag = self.H_diag3.argsort().
    index_Hdiag_.resize(h_dim_);
    std::iota(index_Hdiag_.begin(), index_Hdiag_.end(), 0);
    std::sort(index_Hdiag_.begin(), index_Hdiag_.end(),
              [&](int a, int b) { return H_diag3_initial(a) < H_diag3_initial(b); });

    // build_S_diag -- helper_PFCI.py:1592-1613.
    S_diag_ = Vector::Zero(h_dim_);
    build_S_diag(S_diag_.data(), num_alpha_, nmo, n_act_a, n_act_orb, n_in_a, /*shift=*/0.0);
    S_diag_projection_ = Vector::Zero(h_dim_);
    const double spin_shift = config.target_spin * (config.target_spin + 1.0);
    build_S_diag(S_diag_projection_.data(), num_alpha_, nmo, n_act_a, n_act_orb, n_in_a, spin_shift);
}

} // namespace casscf
