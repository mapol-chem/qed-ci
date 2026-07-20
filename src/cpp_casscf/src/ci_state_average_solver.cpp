#include "casscf/ci_state_average_solver.hpp"

#include "casscf/ci_orbital_backend.hpp"

#include <cstdint>
#include <utility>
#include <vector>

namespace casscf {
namespace {

// ci_solver.h's C signatures don't mark these read-only ("const") array
// parameters const -- confirmed by reading the actual bodies that none of
// table/table_creation/table_annihilation/b_array/Y/S_diag/
// S_diag_projection/index_Hdiag are ever written by get_roots/
// build_H_diag_cas_spin/build_active_rdm/build_active_photon_electron_one_rdm
// (only eigenvals/eigenvecs/D_tu/D_tuvw/Dpe_tu/constint[8]/constdouble[4]
// are genuine outputs). Safe to strip const at this FFI boundary rather
// than needing a fresh copy of CasscfCiSetup's persistent storage on every
// solve() call.
int32_t* mutable_ptr(const std::vector<int32_t>& v) { return const_cast<int32_t*>(v.data()); }
double* mutable_ptr(const Vector& v) { return const_cast<double*>(v.data()); }

} // namespace

CasscfCiStateAverageSolver::CasscfCiStateAverageSolver(Dimensions dims, CasscfCiConfig config,
                                                          CasscfPhysicalConstants constants,
                                                          CasscfCiSetup& setup, CasscfContext& context)
    : dims_(dims), config_(std::move(config)), constants_(std::move(constants)), setup_(&setup),
      context_(&context) {}

CiStateAverageResult CasscfCiStateAverageSolver::solve(const Matrix& eigenvecs_guess) {
    CasscfContext& context = *context_;
    const Dimensions& dims = dims_;
    const int n_act = dims.n_act_orb;
    const int n_in_a = dims.n_in_a;
    const int nmo = dims.nmo;
    const int H_dim = setup_->H_dim();
    const int davidson_roots = config_.davidson_roots;

    // helper_PFCI.py:2424-2488: active_fock_core/active_twoeint/gkl2/
    // occupied_fock_core/occupied_J recomputed fresh from
    // self.H_spatial2/self.J/self.K every macroiteration -- NOT read from
    // context's occupied_*/gkl2 staging fields (see this class's header
    // doc comment).
    ActiveBlockIntermediates active = compute_active_block_intermediates(context.H_spatial2, context.J,
                                                                            context.K, dims);
    OccupiedCiBlocks occupied = build_occupied_ci_blocks(active, dims);

    // gkl2(k,l) = active_fock_core(k,l) - 0.5 * sum_j active_twoeint(k,j,j,l)
    // helper_PFCI.py:2487-2488 -- NOT zero-padded to occupied size (unlike
    // occupied_fock_core): get_roots's h1e argument is genuinely
    // (n_act_orb, n_act_orb)-shaped here, confirmed against get_roots's own
    // documented argtypes (h1e = self.gkl2, (n_act_orb, n_act_orb)) --
    // distinct from build_H_diag_cas_spin's h1e, which IS the occupied-sized
    // occupied_fock_core. Easy to conflate the two since both are called
    // "h1e"-like inputs in the Python and this project's own earlier
    // CasscfContext.gkl2/occupied_fock_core fields already establish they're
    // separate quantities for exactly this reason.
    Matrix gkl2(n_act, n_act);
    for (int k = 0; k < n_act; ++k) {
        for (int l = 0; l < n_act; ++l) {
            double acc = 0.0;
            for (int j = 0; j < n_act; ++j) acc += active.active_twoeint(k, j, j, l);
            gkl2(k, l) = active.active_fock_core(k, l) - 0.5 * acc;
        }
    }
    RowMajorMatrix gkl2_rm = gkl2;

    // helper_PFCI.py:2494-2510: H_diag3 rebuilt fresh every macroiteration
    // (unlike index_Hdiag, which CasscfCiSetup froze once -- see that
    // class's header doc comment).
    RowMajorMatrix occupied_fock_core_rm = occupied.occupied_fock_core;
    Vector H_diag3 = Vector::Zero(H_dim);
    build_H_diag_cas_spin(occupied_fock_core_rm.data(), occupied.occupied_twoeint.data(), H_diag3.data(),
                            constants_.N_p, setup_->num_alpha(), nmo, config_.n_act_a, n_act, n_in_a,
                            context.E_core, constants_.omega, constants_.Enuc, constants_.d_c,
                            mutable_ptr(setup_->Y()), config_.target_spin);

    // helper_PFCI.py:1913-1932, 2511-2518: constint/constdouble. Built
    // fully fresh here every call rather than mirroring the Python's own
    // "patch 4 fields, leave the rest from setup" micro-optimization --
    // produces identical values (everything else is unchanged input this
    // class already has fresh access to every call) with less state to
    // track.
    const double d_diag = 2.0 * context.d_cmo.topLeftCorner(n_in_a, n_in_a).trace();

    std::vector<int32_t> constint = {config_.n_act_a,
                                       n_act,
                                       n_in_a,
                                       nmo,
                                       constants_.N_p,
                                       setup_->indim(),
                                       setup_->maxdim(),
                                       davidson_roots,
                                       config_.davidson_maxiter};
    Vector constdouble(6);
    constdouble(0) = constants_.Enuc;
    constdouble(1) = config_.ignore_dse_terms ? 0.0 : constants_.d_c;
    constdouble(2) = constants_.omega;
    constdouble(3) = constants_.d_exp - d_diag;
    constdouble(4) = config_.davidson_threshold;
    constdouble(5) = context.E_core;

    RowMajorMatrix occupied_d_cmo_rm = context.d_cmo.topLeftCorner(dims.n_occupied, dims.n_occupied);
    Vector eigenvals = Vector::Zero(davidson_roots);
    RowMajorMatrix eigenvecs_rm = eigenvecs_guess;

    get_roots(gkl2_rm.data(), occupied.occupied_twoeint.data(), occupied_d_cmo_rm.data(), H_diag3.data(),
               mutable_ptr(setup_->S_diag()), mutable_ptr(setup_->S_diag_projection()), eigenvals.data(),
               eigenvecs_rm.data(), mutable_ptr(setup_->table()), mutable_ptr(setup_->table_creation()),
               mutable_ptr(setup_->table_annihilation()), mutable_ptr(setup_->b_array()), constint.data(),
               constdouble.data(), mutable_ptr(setup_->index_Hdiag()), /*casscf=*/true, config_.target_spin);

    CiStateAverageResult result;
    result.eigenvalues = eigenvals;
    result.eigenvectors = eigenvecs_rm;
    // helper_PFCI.py:7699/7712-7728-ish: constint[8] == 0 after the call
    // means the Davidson diagonalization converged.
    result.ci_diagonalization_converged = (constint[8] == 0);

    // helper_PFCI.py:2543-2545: avg_energy = sum_i weight[i] * eigenvals[i].
    double avg_energy = 0.0;
    for (int i = 0; i < davidson_roots; ++i) avg_energy += constants_.weight(i) * eigenvals(i);
    result.avg_energy = avg_energy;

    // helper_PFCI.py:8027-8085 (build_state_average_rdms): accumulate
    // weighted RDMs over the just-solved eigenvecs_rm (NOT the original
    // guess), then symmetrize D_tuvw_avg over its (t, u) pair.
    RowMajorMatrix D_tu_avg_rm = RowMajorMatrix::Zero(n_act, n_act);
    RowMajorMatrix Dpe_tu_avg_rm = RowMajorMatrix::Zero(n_act, n_act);
    Tensor4 D_tuvw_avg(n_act, n_act, n_act, n_act);
    D_tuvw_avg.setZero();
    const int num_photon = constants_.N_p + 1;
    for (int i = 0; i < davidson_roots; ++i) {
        build_active_rdm(eigenvecs_rm.data(), D_tu_avg_rm.data(), D_tuvw_avg.data(), mutable_ptr(setup_->table()),
                           config_.n_act_a, n_act, num_photon, i, i, constants_.weight(i));
        build_active_photon_electron_one_rdm(eigenvecs_rm.data(), Dpe_tu_avg_rm.data(),
                                                mutable_ptr(setup_->table()), config_.n_act_a, n_act, num_photon,
                                                i, i, constants_.weight(i));
    }
    for (int t = 0; t < n_act; ++t) {
        for (int u = t; u < n_act; ++u) {
            for (int v = 0; v < n_act; ++v) {
                for (int w = 0; w < n_act; ++w) {
                    const double dum = D_tuvw_avg(t, u, v, w) + D_tuvw_avg(u, t, v, w);
                    D_tuvw_avg(t, u, v, w) = dum / 2.0;
                    D_tuvw_avg(u, t, v, w) = dum / 2.0;
                }
            }
        }
    }
    result.D_tu_avg = D_tu_avg_rm;
    result.D_tuvw_avg = D_tuvw_avg;
    result.Dpe_tu_avg = Dpe_tu_avg_rm;

    return result;
}

} // namespace casscf
