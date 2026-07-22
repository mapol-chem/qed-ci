#include "casscf/intermediates.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace casscf {

SmallBlockIntermediates build_intermediates_internal(const Matrix& occupied_fock_core,
                                                       const Matrix& occupied_d_cmo,
                                                       const Tensor4& occupied_J,
                                                       const Tensor4& occupied_K,
                                                       const Matrix& D_tu_avg,
                                                       const Tensor4& D_tuvw_avg,
                                                       const Matrix& Dpe_tu_avg,
                                                       double off_diagonal_constant,
                                                       double omega, const Dimensions& dims) {
    const int n_occupied = dims.n_occupied;
    const int n_in_a = dims.n_in_a;
    const int n_act = dims.n_act_orb;
    const int rot_dim = n_occupied; // helper_PFCI.py:5579

    SmallBlockIntermediates result;
    result.A = Matrix::Zero(n_occupied, n_occupied);
    result.G = Tensor4(n_occupied, n_occupied, rot_dim, rot_dim);
    result.G.setZero();
    Matrix& A = result.A;
    Tensor4& G = result.G;

    // L(p,j,r,s) = 4*K(p,j,r,s) - K(p,j,s,r) - J(p,j,r,s), helper_PFCI.py:5580-5587
    Tensor4 L(n_occupied, n_in_a, rot_dim, rot_dim);
    for (int p = 0; p < n_occupied; ++p)
        for (int j = 0; j < n_in_a; ++j)
            for (int r = 0; r < rot_dim; ++r)
                for (int s = 0; s < rot_dim; ++s)
                    L(p, j, r, s) = 4.0 * occupied_K(p, j, r, s) - occupied_K(p, j, s, r) - occupied_J(p, j, r, s);

    // fock_general(r,s) = occupied_fock_core(r,s)
    //   + sum_tu D_tu_avg(t,u)*occupied_J(n_in_a+t,n_in_a+u,r,s)
    //   - 0.5*sum_tu D_tu_avg(t,u)*occupied_K(n_in_a+t,n_in_a+u,r,s)
    // helper_PFCI.py:5581, 5588-5607
    Matrix fock_general = occupied_fock_core;
    for (int r = 0; r < rot_dim; ++r) {
        for (int s = 0; s < rot_dim; ++s) {
            double acc = 0.0;
            for (int t = 0; t < n_act; ++t)
                for (int u = 0; u < n_act; ++u)
                    acc += D_tu_avg(t, u) * occupied_J(n_in_a + t, n_in_a + u, r, s) -
                           0.5 * D_tu_avg(t, u) * occupied_K(n_in_a + t, n_in_a + u, r, s);
            fock_general(r, s) += acc;
        }
    }

    // A[:, :n_in_a] = 2*(fock_general[:, :n_in_a] + occupied_d_cmo[:, :n_in_a]*off_diagonal_constant)
    // helper_PFCI.py:5609-5613
    for (int r = 0; r < n_occupied; ++r)
        for (int j = 0; j < n_in_a; ++j)
            A(r, j) = 2.0 * (fock_general(r, j) + occupied_d_cmo(r, j) * off_diagonal_constant);

    // A[:, n_in_a:n_occupied] = einsum("rt,tu->ru", occupied_fock_core[:, act], D_tu_avg) --
    // a plain matrix product. helper_PFCI.py:5615-5619
    {
        Matrix term = occupied_fock_core.block(0, n_in_a, n_occupied, n_act) * D_tu_avg;
        for (int r = 0; r < n_occupied; ++r)
            for (int u = 0; u < n_act; ++u) A(r, n_in_a + u) = term(r, u);
    }

    // A[:, n_in_a:n_occupied] += einsum("rtvw,tuvw->ru", occupied_J[:,act,act,act], D_tuvw_avg)
    // helper_PFCI.py:5620-5633
    for (int r = 0; r < n_occupied; ++r) {
        for (int u = 0; u < n_act; ++u) {
            double acc = 0.0;
            for (int t = 0; t < n_act; ++t)
                for (int v = 0; v < n_act; ++v)
                    for (int w = 0; w < n_act; ++w)
                        acc += occupied_J(r, n_in_a + t, n_in_a + v, n_in_a + w) * D_tuvw_avg(t, u, v, w);
            A(r, n_in_a + u) += acc;
        }
    }

    // A[:, n_in_a:n_occupied] += -sqrt(omega/2)*einsum("rt,tu->ru", occupied_d_cmo[:,act], Dpe_tu_avg)
    // helper_PFCI.py:5634-5638 -- plain matrix product
    {
        Matrix term = occupied_d_cmo.block(0, n_in_a, n_occupied, n_act) * Dpe_tu_avg;
        const double pref = -std::sqrt(omega / 2.0);
        for (int r = 0; r < n_occupied; ++r)
            for (int u = 0; u < n_act; ++u) A(r, n_in_a + u) += pref * term(r, u);
    }

    // G[:n_in_a,:n_in_a,:,:] = 2*fock_general*delta(i,j) + 2*L[:n_in_a,:,:,:]
    //                        + 2*off_diagonal_constant*occupied_d_cmo*delta(i,j)
    // helper_PFCI.py:5660-5670
    for (int i = 0; i < n_in_a; ++i) {
        for (int j = 0; j < n_in_a; ++j) {
            for (int r = 0; r < rot_dim; ++r) {
                for (int s = 0; s < rot_dim; ++s) {
                    double val = 2.0 * L(i, j, r, s);
                    if (i == j) {
                        val += 2.0 * fock_general(r, s);
                        val += 2.0 * off_diagonal_constant * occupied_d_cmo(r, s);
                    }
                    G(i, j, r, s) = val;
                }
            }
        }
    }

    // G[n_in_a:n_occupied,:n_in_a,:,:] = einsum("tv,vjrs->tjrs", D_tu_avg, L[n_in_a:,:,:,:])
    // helper_PFCI.py:5672-5676
    for (int t = 0; t < n_act; ++t) {
        for (int j = 0; j < n_in_a; ++j) {
            for (int r = 0; r < rot_dim; ++r) {
                for (int s = 0; s < rot_dim; ++s) {
                    double acc = 0.0;
                    for (int v = 0; v < n_act; ++v) acc += D_tu_avg(t, v) * L(n_in_a + v, j, r, s);
                    G(n_in_a + t, j, r, s) = acc;
                }
            }
        }
    }

    // G[:n_in_a, n_in_a:n_occupied, :, :] = G[n_in_a:n_occupied, :n_in_a, :, :].transpose(1,0,3,2)
    // helper_PFCI.py:5677-5679
    for (int j = 0; j < n_in_a; ++j)
        for (int t = 0; t < n_act; ++t)
            for (int r = 0; r < rot_dim; ++r)
                for (int s = 0; s < rot_dim; ++s) G(j, n_in_a + t, r, s) = G(n_in_a + t, j, s, r);

    // G[n_in_a:, n_in_a:, :, :] = einsum("rs,tu->turs", occupied_fock_core, D_tu_avg)
    //                            + einsum("vwrs,tuvw->turs", occupied_J[act,act,:,:], D_tuvw_avg)
    //                            + 2*einsum("vwrs,tvuw->turs", occupied_K[act,act,:,:], D_tuvw_avg)
    //                            - sqrt(omega/2)*einsum("rs,tu->turs", occupied_d_cmo, Dpe_tu_avg)
    // helper_PFCI.py:5681-5714. Note the third term's einsum label "tvuw" on
    // D_tuvw_avg -- einsum labels are purely positional per operand, so this
    // means literally D_tuvw_avg(t,v,u,w) (arguments in that order), not the
    // canonical D_tuvw_avg(t,u,v,w) used everywhere else in this function.
    {
        const double pref_pe = -std::sqrt(omega / 2.0);
        for (int t = 0; t < n_act; ++t) {
            for (int u = 0; u < n_act; ++u) {
                for (int r = 0; r < rot_dim; ++r) {
                    for (int s = 0; s < rot_dim; ++s) {
                        double val = occupied_fock_core(r, s) * D_tu_avg(t, u);
                        val += pref_pe * occupied_d_cmo(r, s) * Dpe_tu_avg(t, u);
                        double acc_J = 0.0, acc_K = 0.0;
                        for (int v = 0; v < n_act; ++v) {
                            for (int w = 0; w < n_act; ++w) {
                                acc_J += occupied_J(n_in_a + v, n_in_a + w, r, s) * D_tuvw_avg(t, u, v, w);
                                acc_K += occupied_K(n_in_a + v, n_in_a + w, r, s) * D_tuvw_avg(t, v, u, w);
                            }
                        }
                        val += acc_J + 2.0 * acc_K;
                        G(n_in_a + t, n_in_a + u, r, s) = val;
                    }
                }
            }
        }
    }

    return result;
}

double calculate_off_diagonal_photon_constant(const Matrix& eigenvecs, const Vector& weight,
                                               int N_p, int num_det, double omega) {
    double off_diagonal_constant = 0.0;
    if (N_p == 0) return off_diagonal_constant; // helper_PFCI.py:6113-6114

    const int np1 = N_p + 1;
    const int davidson_roots = static_cast<int>(eigenvecs.rows());

    auto block = [&](int i, int m) { return eigenvecs.row(i).segment(m * num_det, num_det); };

    for (int i = 0; i < davidson_roots; ++i) {
        for (int m = 0; m < np1; ++m) {
            if (m > 0 && m < N_p) {
                off_diagonal_constant += -weight(i) * std::sqrt(m * omega / 2.0) * block(i, m).dot(block(i, m - 1));
                off_diagonal_constant += -weight(i) * std::sqrt((m + 1) * omega / 2.0) * block(i, m).dot(block(i, m + 1));
            } else if (m == N_p) {
                off_diagonal_constant += -weight(i) * std::sqrt(m * omega / 2.0) * block(i, m).dot(block(i, m - 1));
            } else { // m == 0 (N_p > 0 here, guaranteed by the early return above)
                off_diagonal_constant += -weight(i) * std::sqrt((m + 1) * omega / 2.0) * block(i, m).dot(block(i, m + 1));
            }
        }
    }
    return off_diagonal_constant;
}

double calculate_ci_dependent_energy(const Matrix& eigenvecs, const Matrix& occupied_d_cmo,
                                      const Vector& weight, int N_p, int num_det, double omega,
                                      double d_exp, int n_in_a) {
    // helper_PFCI.py:6053
    const double d_diag = 2.0 * occupied_d_cmo.topLeftCorner(n_in_a, n_in_a).trace();
    if (N_p == 0) return 0.0; // helper_PFCI.py:6060-6061 (continue every m -> nothing accumulates)

    const int np1 = N_p + 1;
    const int davidson_roots = static_cast<int>(eigenvecs.rows());
    double off_diagonal_constant_energy = 0.0;
    double photon_energy = 0.0;

    auto block = [&](int i, int m) { return eigenvecs.row(i).segment(m * num_det, num_det); };

    for (int i = 0; i < davidson_roots; ++i) {
        for (int m = 0; m < np1; ++m) {
            if (m > 0 && m < N_p) {
                off_diagonal_constant_energy +=
                    weight(i) * std::sqrt(m * omega / 2.0) * (d_exp - d_diag) * block(i, m).dot(block(i, m - 1));
                off_diagonal_constant_energy +=
                    weight(i) * std::sqrt((m + 1) * omega / 2.0) * (d_exp - d_diag) * block(i, m).dot(block(i, m + 1));
            } else if (m == N_p) {
                off_diagonal_constant_energy +=
                    weight(i) * std::sqrt(m * omega / 2.0) * (d_exp - d_diag) * block(i, m).dot(block(i, m - 1));
            } else { // m == 0
                off_diagonal_constant_energy +=
                    weight(i) * std::sqrt((m + 1) * omega / 2.0) * (d_exp - d_diag) * block(i, m).dot(block(i, m + 1));
            }
            photon_energy += weight(i) * m * omega * block(i, m).dot(block(i, m));
        }
    }
    return off_diagonal_constant_energy + photon_energy;
}

FullBlockIntermediates build_intermediates(const Matrix& H_spatial2, const Matrix& d_cmo,
                                            const Tensor4& J, const Tensor4& K,
                                            const Matrix& D_tu_avg, const Tensor4& D_tuvw_avg,
                                            const Matrix& Dpe_tu_avg, double off_diagonal_constant,
                                            double omega, const Dimensions& dims) {
    const int nmo = dims.nmo;
    const int n_occupied = dims.n_occupied;
    const int n_in_a = dims.n_in_a;
    const int n_act = dims.n_act_orb;
    const int rot_dim = nmo; // helper_PFCI.py:5749-5750, full_space == True is the only active case

    FullBlockIntermediates result;
    result.A = Matrix::Zero(rot_dim, rot_dim);
    result.G = Tensor4(n_occupied, n_occupied, rot_dim, rot_dim);
    result.G.setZero();

    // fock_core(r,s) = H_spatial2(r,s) + 2*sum_{j<n_in_a} J(j,j,r,s) - sum_{j<n_in_a} K(j,j,r,s)
    // helper_PFCI.py:5754-5760
    Matrix fock_core = H_spatial2;
    for (int r = 0; r < nmo; ++r) {
        for (int s = 0; s < nmo; ++s) {
            double acc = 0.0;
            for (int j = 0; j < n_in_a; ++j) acc += 2.0 * J(j, j, r, s) - K(j, j, r, s);
            fock_core(r, s) += acc;
        }
    }
    result.fock_core = fock_core;

    Matrix& A = result.A;
    Tensor4& G = result.G;

    // L(p,j,r,s) = 4*K(p,j,r,s) - K(p,j,s,r) - J(p,j,r,s), for p in [0,n_occupied), j in [0,n_in_a)
    // helper_PFCI.py:5783-5791 (J, K here are (n_occupied,n_occupied,nmo,nmo) -- see header doc)
    Tensor4 L(n_occupied, n_in_a, rot_dim, rot_dim);
    for (int p = 0; p < n_occupied; ++p)
        for (int j = 0; j < n_in_a; ++j)
            for (int r = 0; r < rot_dim; ++r)
                for (int s = 0; s < rot_dim; ++s)
                    L(p, j, r, s) = 4.0 * K(p, j, r, s) - K(p, j, s, r) - J(p, j, r, s);
    result.L = L;

    // helper_PFCI.py:5955-5969 -- the remaining reference-point side
    // outputs (E_core, active_fock_core, active_twoeint), computed
    // directly from H_spatial2/fock_core/J, all already in scope above.
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

    // fock_general(r,s) = fock_core(r,s)
    //   + sum_tu D_tu_avg(t,u)*(J(n_in_a+t,n_in_a+u,r,s) - 0.5*K(n_in_a+t,n_in_a+u,r,s))
    // helper_PFCI.py:5793-5811
    Matrix fock_general = fock_core;
    for (int r = 0; r < rot_dim; ++r) {
        for (int s = 0; s < rot_dim; ++s) {
            double acc = 0.0;
            for (int t = 0; t < n_act; ++t)
                for (int u = 0; u < n_act; ++u)
                    acc += D_tu_avg(t, u) * J(n_in_a + t, n_in_a + u, r, s) -
                           0.5 * D_tu_avg(t, u) * K(n_in_a + t, n_in_a + u, r, s);
            fock_general(r, s) += acc;
        }
    }

    // A[:, :n_in_a] = 2*(fock_general[:, :n_in_a] + d_cmo[:rot_dim, :n_in_a]*off_diagonal_constant)
    // helper_PFCI.py:5814-5817
    for (int r = 0; r < rot_dim; ++r)
        for (int j = 0; j < n_in_a; ++j) A(r, j) = 2.0 * (fock_general(r, j) + d_cmo(r, j) * off_diagonal_constant);

    // A[:, n_in_a:n_occupied] = einsum("rt,tu->ru", fock_core[:rot_dim,act], D_tu_avg) -- matrix product
    // helper_PFCI.py:5820-5825
    {
        Matrix term = fock_core.block(0, n_in_a, rot_dim, n_act) * D_tu_avg;
        for (int r = 0; r < rot_dim; ++r)
            for (int u = 0; u < n_act; ++u) A(r, n_in_a + u) = term(r, u);
    }

    // A[:, n_in_a:n_occupied] += einsum("vwrt,tuvw->ru",
    //     J[n_in_a:n_occupied, n_in_a:n_occupied, :rot_dim, n_in_a:n_occupied], D_tuvw_avg)
    // helper_PFCI.py:5830-5840. A genuinely different index pattern from
    // build_intermediates_internal's analogous term ("rtvw,tuvw->ru") --
    // verified term by term, not the same formula under different names.
    for (int r = 0; r < rot_dim; ++r) {
        for (int u = 0; u < n_act; ++u) {
            double acc = 0.0;
            for (int v = 0; v < n_act; ++v)
                for (int w = 0; w < n_act; ++w)
                    for (int t = 0; t < n_act; ++t)
                        acc += J(n_in_a + v, n_in_a + w, r, n_in_a + t) * D_tuvw_avg(t, u, v, w);
            A(r, n_in_a + u) += acc;
        }
    }

    // A[:, n_in_a:n_occupied] += -sqrt(omega/2)*einsum("rt,tu->ru", d_cmo[:rot_dim,act], Dpe_tu_avg)
    // helper_PFCI.py:5844-5849 -- matrix product
    {
        Matrix term = d_cmo.block(0, n_in_a, rot_dim, n_act) * Dpe_tu_avg;
        const double pref = -std::sqrt(omega / 2.0);
        for (int r = 0; r < rot_dim; ++r)
            for (int u = 0; u < n_act; ++u) A(r, n_in_a + u) += pref * term(r, u);
    }

    // temp2(r,s) = fock_general(r,s) + off_diagonal_constant*d_cmo(r,s), helper_PFCI.py:5850-5853
    Matrix temp2 = fock_general;
    for (int r = 0; r < rot_dim; ++r)
        for (int s = 0; s < rot_dim; ++s) temp2(r, s) += off_diagonal_constant * d_cmo(r, s);

    // G[:n_in_a,:n_in_a,:,:] = 2*temp2*delta(i,j) + 2*L[:n_in_a,:,:,:], helper_PFCI.py:5857-5863
    for (int i = 0; i < n_in_a; ++i)
        for (int j = 0; j < n_in_a; ++j)
            for (int r = 0; r < rot_dim; ++r)
                for (int s = 0; s < rot_dim; ++s)
                    G(i, j, r, s) = 2.0 * L(i, j, r, s) + (i == j ? 2.0 * temp2(r, s) : 0.0);

    // G[n_in_a:n_occupied,:n_in_a,:,:] = einsum("tv,vjrs->tjrs", D_tu_avg, L[n_in_a:,:,:,:])
    // helper_PFCI.py:5870-5875 (L's axis0 is n_occupied-sized, so L[n_in_a:,...] has axis0
    // size n_act -- matches D_tu_avg's contracted axis)
    for (int t = 0; t < n_act; ++t) {
        for (int j = 0; j < n_in_a; ++j) {
            for (int r = 0; r < rot_dim; ++r) {
                for (int s = 0; s < rot_dim; ++s) {
                    double acc = 0.0;
                    for (int v = 0; v < n_act; ++v) acc += D_tu_avg(t, v) * L(n_in_a + v, j, r, s);
                    G(n_in_a + t, j, r, s) = acc;
                }
            }
        }
    }

    // G[:n_in_a, n_in_a:n_occupied, :, :] = G[n_in_a:n_occupied, :n_in_a, :, :].transpose(1,0,3,2)
    // helper_PFCI.py:5879-5881
    for (int j = 0; j < n_in_a; ++j)
        for (int t = 0; t < n_act; ++t)
            for (int r = 0; r < rot_dim; ++r)
                for (int s = 0; s < rot_dim; ++s) G(j, n_in_a + t, r, s) = G(n_in_a + t, j, s, r);

    // G[n_in_a:, n_in_a:, :, :] = einsum("rs,tu->turs", fock_core[:rot_dim,:rot_dim], D_tu_avg)
    //                            + einsum("vwrs,tuvw->turs", J[act,act,:rot_dim,:rot_dim], D_tuvw_avg)
    //                            + 2*einsum("vwrs,tvuw->turs", K[act,act,:rot_dim,:rot_dim], D_tuvw_avg)
    //                            - sqrt(omega/2)*einsum("rs,tu->turs", d_cmo[:rot_dim,:rot_dim], Dpe_tu_avg)
    // helper_PFCI.py:5886-5928 -- structurally identical to
    // build_intermediates_internal's G[n_in_a:,n_in_a:,:,:] block (same
    // einsum strings, different-sized/full-space inputs).
    {
        const double pref_pe = -std::sqrt(omega / 2.0);
        for (int t = 0; t < n_act; ++t) {
            for (int u = 0; u < n_act; ++u) {
                for (int r = 0; r < rot_dim; ++r) {
                    for (int s = 0; s < rot_dim; ++s) {
                        double val = fock_core(r, s) * D_tu_avg(t, u);
                        val += pref_pe * d_cmo(r, s) * Dpe_tu_avg(t, u);
                        double acc_J = 0.0, acc_K = 0.0;
                        for (int v = 0; v < n_act; ++v) {
                            for (int w = 0; w < n_act; ++w) {
                                acc_J += J(n_in_a + v, n_in_a + w, r, s) * D_tuvw_avg(t, u, v, w);
                                acc_K += K(n_in_a + v, n_in_a + w, r, s) * D_tuvw_avg(t, v, u, w);
                            }
                        }
                        val += acc_J + 2.0 * acc_K;
                        G(n_in_a + t, n_in_a + u, r, s) = val;
                    }
                }
            }
        }
    }

    return result;
}

// ---------------------------------------------------------------------------
// FAST path: build_intermediates_fast. Same math as build_intermediates above,
// with the four contraction-shaped terms re-expressed as BLAS-backed Eigen
// matrix products and the elementwise ones as (r,s)-slab operations. See
// intermediates.hpp's doc comment for the term-by-term correspondence.
//
// Every slab map below relies on Tensor4 being RowMajor (tensor_types.hpp:17):
// for a (d0,d1,nmo,nmo) tensor the trailing (r,s) plane at fixed (i,j) is
// nmo*nmo contiguous doubles, so it maps directly as a RowMajorMatrix with no
// copy. Getting that storage order wrong would silently transpose slabs, which
// is exactly what the cross-validation test guards.
// ---------------------------------------------------------------------------

namespace {

// (r,s) slab of a (d0,d1,nmo,nmo) RowMajor tensor at fixed (i,j), as a matrix.
inline Eigen::Map<const RowMajorMatrix> slab(const Tensor4& T, int i, int j, int nmo) {
    return Eigen::Map<const RowMajorMatrix>(T.data() + (static_cast<std::ptrdiff_t>(i) * T.dimension(1) + j) *
                                                           static_cast<std::ptrdiff_t>(nmo) * nmo,
                                            nmo, nmo);
}

inline Eigen::Map<RowMajorMatrix> slab_mut(Tensor4& T, int i, int j, int nmo) {
    return Eigen::Map<RowMajorMatrix>(T.data() + (static_cast<std::ptrdiff_t>(i) * T.dimension(1) + j) *
                                                     static_cast<std::ptrdiff_t>(nmo) * nmo,
                                      nmo, nmo);
}

} // namespace

FullBlockIntermediates build_intermediates_fast(const Matrix& H_spatial2, const Matrix& d_cmo,
                                                 const Tensor4& J, const Tensor4& K,
                                                 const Matrix& D_tu_avg, const Tensor4& D_tuvw_avg,
                                                 const Matrix& Dpe_tu_avg, double off_diagonal_constant,
                                                 double omega, const Dimensions& dims) {
    const int nmo = dims.nmo;
    const int n_occupied = dims.n_occupied;
    const int n_in_a = dims.n_in_a;
    const int n_act = dims.n_act_orb;
    const int rot_dim = nmo; // as in build_intermediates: full_space == True is the only active case
    const int nmo2 = nmo * nmo;
    const int nact2 = n_act * n_act;

    FullBlockIntermediates result;
    result.A = Matrix::Zero(rot_dim, rot_dim);
    result.G = Tensor4(n_occupied, n_occupied, rot_dim, rot_dim);
    result.G.setZero();

    // Row-major copies of the two matrices that get combined with tensor
    // slabs below (Matrix is ColMajor; mixing storage orders in a binary
    // expression is avoided here entirely rather than relied upon).
    const RowMajorMatrix H_rm = H_spatial2;
    const RowMajorMatrix d_cmo_rm = d_cmo;

    // fock_core, legacy lines 239-245: sum of n_in_a diagonal slabs.
    RowMajorMatrix fock_core_rm = H_rm;
    for (int j = 0; j < n_in_a; ++j) fock_core_rm += 2.0 * slab(J, j, j, nmo) - slab(K, j, j, nmo);
    const Matrix fock_core = fock_core_rm;
    result.fock_core = fock_core;

    Matrix& A = result.A;
    Tensor4& G = result.G;

    // L, legacy lines 254-258: elementwise per (p,j) slab. The transpose is of
    // a distinct buffer from the destination, so there is no aliasing.
    Tensor4 L(n_occupied, n_in_a, rot_dim, rot_dim);
    for (int p = 0; p < n_occupied; ++p) {
        for (int j = 0; j < n_in_a; ++j) {
            const auto Kpj = slab(K, p, j, nmo);
            slab_mut(L, p, j, nmo) = 4.0 * Kpj - Kpj.transpose() - slab(J, p, j, nmo);
        }
    }
    result.L = L;

    // E_core / active_fock_core / active_twoeint, legacy lines 264-275.
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

    // Active-active (v,w) blocks of J and K, packed as (n_act^2, nmo^2) with
    // row (v*n_act + w) -- the shared operand of every contraction below.
    // For fixed v the n_act rows are already contiguous in the source tensor
    // (rows (n_in_a+v)*n_occupied + (n_in_a+w), w contiguous), so each is one
    // chunk copy of n_act*nmo^2 doubles.
    RowMajorMatrix Jmat(nact2, nmo2);
    RowMajorMatrix Kmat(nact2, nmo2);
    for (int v = 0; v < n_act; ++v) {
        const std::ptrdiff_t src =
            (static_cast<std::ptrdiff_t>(n_in_a + v) * n_occupied + n_in_a) * static_cast<std::ptrdiff_t>(nmo2);
        std::copy(J.data() + src, J.data() + src + static_cast<std::ptrdiff_t>(n_act) * nmo2,
                  Jmat.data() + static_cast<std::ptrdiff_t>(v) * n_act * nmo2);
        std::copy(K.data() + src, K.data() + src + static_cast<std::ptrdiff_t>(n_act) * nmo2,
                  Kmat.data() + static_cast<std::ptrdiff_t>(v) * n_act * nmo2);
    }

    // fock_general, legacy lines 281-290: sum_{tu} D(t,u)*(J - 0.5*K) over the
    // active-active slabs == one GEMV against the packed blocks.
    Vector dvec(nact2);
    for (int t = 0; t < n_act; ++t)
        for (int u = 0; u < n_act; ++u) dvec(t * n_act + u) = D_tu_avg(t, u);
    const Vector fg_flat = (Jmat - 0.5 * Kmat).transpose() * dvec;
    RowMajorMatrix fock_general_rm = fock_core_rm + Eigen::Map<const RowMajorMatrix>(fg_flat.data(), nmo, nmo);
    const Matrix fock_general = fock_general_rm;

    // A inactive columns, legacy lines 294-295.
    for (int r = 0; r < rot_dim; ++r)
        for (int j = 0; j < n_in_a; ++j) A(r, j) = 2.0 * (fock_general(r, j) + d_cmo(r, j) * off_diagonal_constant);

    // A active columns, term 1, legacy lines 299-303 (already a matrix product).
    {
        Matrix term = fock_core.block(0, n_in_a, rot_dim, n_act) * D_tu_avg;
        for (int r = 0; r < rot_dim; ++r)
            for (int u = 0; u < n_act; ++u) A(r, n_in_a + u) = term(r, u);
    }

    // A active columns, "vwrt,tuvw->ru" term, legacy lines 310-319. Packed as
    // one GEMM by folding t into the contracted row index:
    //   P[(t,v,w), r] = J(n_in_a+v, n_in_a+w, r, n_in_a+t)
    //   Q[(t,v,w), u] = D_tuvw_avg(t, u, v, w)
    //   acc = P^T * Q  -> (rot_dim, n_act)
    {
        RowMajorMatrix P(static_cast<Eigen::Index>(n_act) * nact2, rot_dim);
        RowMajorMatrix Q(static_cast<Eigen::Index>(n_act) * nact2, n_act);
        for (int t = 0; t < n_act; ++t) {
            for (int v = 0; v < n_act; ++v) {
                for (int w = 0; w < n_act; ++w) {
                    const int row = (t * n_act + v) * n_act + w;
                    for (int r = 0; r < rot_dim; ++r) P(row, r) = J(n_in_a + v, n_in_a + w, r, n_in_a + t);
                    for (int u = 0; u < n_act; ++u) Q(row, u) = D_tuvw_avg(t, u, v, w);
                }
            }
        }
        const Matrix acc = P.transpose() * Q;
        for (int r = 0; r < rot_dim; ++r)
            for (int u = 0; u < n_act; ++u) A(r, n_in_a + u) += acc(r, u);
    }

    // A active columns, photon term, legacy lines 323-328 (matrix product).
    {
        Matrix term = d_cmo.block(0, n_in_a, rot_dim, n_act) * Dpe_tu_avg;
        const double pref = -std::sqrt(omega / 2.0);
        for (int r = 0; r < rot_dim; ++r)
            for (int u = 0; u < n_act; ++u) A(r, n_in_a + u) += pref * term(r, u);
    }

    // temp2, legacy lines 331-333.
    const RowMajorMatrix temp2_rm = fock_general_rm + off_diagonal_constant * d_cmo_rm;

    // G inactive-inactive, legacy lines 336-340: per-slab.
    for (int i = 0; i < n_in_a; ++i) {
        for (int j = 0; j < n_in_a; ++j) {
            slab_mut(G, i, j, nmo) = 2.0 * slab(L, i, j, nmo);
            if (i == j) slab_mut(G, i, j, nmo) += 2.0 * temp2_rm;
        }
    }

    // G active-inactive, legacy lines 345-355: "tv,vjrs->tjrs". L's rows
    // n_in_a..n_occupied are contiguous, so the whole (v, j*r*s) operand maps
    // without a copy and the contraction is a single GEMM.
    {
        const Eigen::Map<const RowMajorMatrix> Lact(
            L.data() + static_cast<std::ptrdiff_t>(n_in_a) * n_in_a * nmo2, n_act,
            static_cast<Eigen::Index>(n_in_a) * nmo2);
        const RowMajorMatrix Gti = RowMajorMatrix(D_tu_avg) * Lact; // (n_act, n_in_a*nmo^2)
        for (int t = 0; t < n_act; ++t)
            for (int j = 0; j < n_in_a; ++j)
                slab_mut(G, n_in_a + t, j, nmo) =
                    Eigen::Map<const RowMajorMatrix>(Gti.data() + (static_cast<std::ptrdiff_t>(t) * n_in_a + j) * nmo2,
                                                     nmo, nmo);
    }

    // G inactive-active, legacy lines 359-362: slab transpose of the above.
    for (int j = 0; j < n_in_a; ++j)
        for (int t = 0; t < n_act; ++t)
            slab_mut(G, j, n_in_a + t, nmo) = slab(G, n_in_a + t, j, nmo).transpose();

    // G active-active, legacy lines 371-392 -- the dominant term
    // (O(n_act^4 * nmo^2)). Both contractions become GEMMs against the packed
    // (n_act^2, nmo^2) J/K blocks:
    //   "vwrs,tuvw->turs" : D1[(t,u),(v,w)] = D_tuvw_avg(t,u,v,w) -- which is
    //                       exactly D_tuvw_avg's RowMajor buffer reshaped, no
    //                       copy needed
    //   "vwrs,tvuw->turs" : D2[(t,u),(v,w)] = D_tuvw_avg(t,v,u,w) -- a genuine
    //                       index permutation, so materialized explicitly
    {
        const Eigen::Map<const RowMajorMatrix> D1(D_tuvw_avg.data(), nact2, nact2);
        RowMajorMatrix D2(nact2, nact2);
        for (int t = 0; t < n_act; ++t)
            for (int u = 0; u < n_act; ++u)
                for (int v = 0; v < n_act; ++v)
                    for (int w = 0; w < n_act; ++w) D2(t * n_act + u, v * n_act + w) = D_tuvw_avg(t, v, u, w);

        const RowMajorMatrix GJ = D1 * Jmat;           // (n_act^2, nmo^2)
        const RowMajorMatrix GK = D2 * Kmat;           // (n_act^2, nmo^2)
        const double pref_pe = -std::sqrt(omega / 2.0);
        for (int t = 0; t < n_act; ++t) {
            for (int u = 0; u < n_act; ++u) {
                const int tu = t * n_act + u;
                slab_mut(G, n_in_a + t, n_in_a + u, nmo) =
                    D_tu_avg(t, u) * fock_core_rm + (pref_pe * Dpe_tu_avg(t, u)) * d_cmo_rm +
                    Eigen::Map<const RowMajorMatrix>(GJ.data() + static_cast<std::ptrdiff_t>(tu) * nmo2, nmo, nmo) +
                    2.0 * Eigen::Map<const RowMajorMatrix>(GK.data() + static_cast<std::ptrdiff_t>(tu) * nmo2, nmo,
                                                            nmo);
            }
        }
    }

    return result;
}

GradientResult build_gradient(const Matrix& U, const Matrix& A, const Tensor4& G, const Dimensions& dims) {
    const int nmo = dims.nmo;
    const int n_occupied = dims.n_occupied;

    // B(r,k) = A(r,k) + sum_{s,l} G(k,l,r,s) * T(s,l), T(s,l) = U(s,l) - delta(s,l)
    // helper_PFCI.py:6190-6194
    Matrix B = Matrix::Zero(nmo, n_occupied);
    for (int r = 0; r < nmo; ++r) {
        for (int k = 0; k < n_occupied; ++k) {
            double acc = A(r, k);
            for (int s = 0; s < nmo; ++s) {
                for (int l = 0; l < n_occupied; ++l) {
                    const double T_sl = U(s, l) - (s == l ? 1.0 : 0.0);
                    acc += G(k, l, r, s) * T_sl;
                }
            }
            B(r, k) = acc;
        }
    }

    // A_tilde[:, :n_occupied] = einsum("rs,sk->rk", U.T, B) -- plain matrix product
    // helper_PFCI.py:6196-6198
    GradientResult result;
    result.A_tilde = U.transpose() * B;

    // gradient_tilde(r,k) = A_tilde(r,k) - (r < n_occupied ? A_tilde(k,r) : 0)
    // helper_PFCI.py:6201-6203 -- A_tilde's virtual-orbital columns (accessed
    // here as A_tilde(k,r) for r >= n_occupied) were never populated by the
    // Python and stayed 0, so the antisymmetrization only actually applies
    // within the occ-occ block.
    result.gradient_tilde = Matrix::Zero(nmo, n_occupied);
    for (int r = 0; r < nmo; ++r)
        for (int k = 0; k < n_occupied; ++k)
            result.gradient_tilde(r, k) = result.A_tilde(r, k) - (r < n_occupied ? result.A_tilde(k, r) : 0.0);

    return result;
}

Matrix embed_and_symmetrize_A_tilde(const Matrix& A_tilde_occ_cols, const Dimensions& dims) {
    Matrix A_tilde_full = Matrix::Zero(dims.nmo, dims.nmo);
    A_tilde_full.leftCols(dims.n_occupied) = A_tilde_occ_cols;
    return A_tilde_full + A_tilde_full.transpose();
}

GradientAndHessianResult build_gradient_and_hessian(const Matrix& A, const Tensor4& G, const Dimensions& dims) {
    const int n_occupied = dims.n_occupied;

    GradientAndHessianResult result;
    // gradient_tilde = A - A.T, helper_PFCI.py:6414
    result.gradient_tilde = A - A.transpose();

    // A3_tilde = A + A.T, helper_PFCI.py:6426
    const Matrix A3_tilde = A + A.transpose();

    // hessian_tilde(k,l,r,s) = G(k,l,r,s) - G(r,l,k,s) - G(k,s,r,l) + G(r,s,k,l)
    //   - 0.5*delta(k,l)*A3_tilde(r,s) - 0.5*delta(r,s)*A3_tilde(k,l)
    //   + 0.5*delta(r,l)*A3_tilde(k,s) + 0.5*delta(k,s)*A3_tilde(r,l)
    // helper_PFCI.py:6416-6440 (derived by tracing each .transpose(...) call
    // against the "rot_dim == n_occupied" assumption established above --
    // all four axes range over n_occupied here, so the r</s<n_occupied
    // slice guards in the Python are unconditionally true and dropped).
    result.hessian_tilde = Tensor4(n_occupied, n_occupied, n_occupied, n_occupied);
    Tensor4& H = result.hessian_tilde;
    for (int k = 0; k < n_occupied; ++k) {
        for (int l = 0; l < n_occupied; ++l) {
            for (int r = 0; r < n_occupied; ++r) {
                for (int s = 0; s < n_occupied; ++s) {
                    double val = G(k, l, r, s) - G(r, l, k, s) - G(k, s, r, l) + G(r, s, k, l);
                    val -= 0.5 * (k == l ? 1.0 : 0.0) * A3_tilde(r, s);
                    val -= 0.5 * (r == s ? 1.0 : 0.0) * A3_tilde(k, l);
                    val += 0.5 * (r == l ? 1.0 : 0.0) * A3_tilde(k, s);
                    val += 0.5 * (k == s ? 1.0 : 0.0) * A3_tilde(r, l);
                    H(k, l, r, s) = val;
                }
            }
        }
    }
    return result;
}

HessianDiagonalResult build_hessian_diagonal(const Matrix& U, const Tensor4& G, const Matrix& A_tilde,
                                              const Dimensions& dims) {
    const int nmo = dims.nmo;
    const int n_occupied = dims.n_occupied;
    const int n_virtual = dims.n_virtual;

    HessianDiagonalResult result;
    Matrix& hd = result.hessian_diagonal;
    hd = Matrix::Zero(nmo, n_occupied);

    // Virtual-occupied block: hessian_diagonal(n_occupied+a, k)
    //   = sum_{r,s} G(k,k,r,s)*U(r,n_occupied+a)*U(s,n_occupied+a)
    //     - A_tilde(n_occupied+a, n_occupied+a)  [always 0, see header doc comment]
    //     - A_tilde(k,k)
    // helper_PFCI.py:14419-14455 (temp1(k,r,s) = G(k,k,r,s) is a diagonal
    // extraction, folded directly into the r,s loop here rather than
    // materialized as a separate array).
    for (int a = 0; a < n_virtual; ++a) {
        const int va = n_occupied + a;
        for (int k = 0; k < n_occupied; ++k) {
            double acc = 0.0;
            for (int r = 0; r < nmo; ++r)
                for (int s = 0; s < nmo; ++s) acc += G(k, k, r, s) * U(r, va) * U(s, va);
            acc -= A_tilde(va, va);
            acc -= A_tilde(k, k);
            hd(va, k) = acc;
        }
    }

    // Occupied-occupied block, for p (row), q (col) both in [0, n_occupied):
    //   hessian_diagonal(p,q) = sum_rs G(q,q,r,s)*U(r,p)*U(s,p)      [temp4]
    //                         + sum_rs G(p,p,r,s)*U(r,q)*U(s,q)      [temp4.T]
    //                         - 2*sum_rs G(p,q,r,s)*U(s,p)*U(r,q)
    //                         - A_tilde(q,q) - A_tilde(p,p)
    //                         + (p==q ? 2*A_tilde(p,p) : 0)
    // helper_PFCI.py:14459-14492
    for (int p = 0; p < n_occupied; ++p) {
        for (int q = 0; q < n_occupied; ++q) {
            double acc = 0.0;
            for (int r = 0; r < nmo; ++r) {
                for (int s = 0; s < nmo; ++s) {
                    acc += G(q, q, r, s) * U(r, p) * U(s, p);
                    acc += G(p, p, r, s) * U(r, q) * U(s, q);
                    acc -= 2.0 * G(p, q, r, s) * U(s, p) * U(r, q);
                }
            }
            acc -= A_tilde(q, q);
            acc -= A_tilde(p, p);
            if (p == q) acc += 2.0 * A_tilde(p, p);
            hd(p, q) = acc;
        }
    }

    // Pack into reduced_hessian_diagonal -- see header doc comment for why
    // build_index_map's enumeration matches this function's own reduction
    // loop exactly.
    const auto index_map = build_index_map(dims);
    result.reduced_hessian_diagonal = Vector::Zero(static_cast<int>(index_map.size()));
    for (size_t i = 0; i < index_map.size(); ++i)
        result.reduced_hessian_diagonal(static_cast<int>(i)) = hd(index_map[i].first, index_map[i].second);

    return result;
}

} // namespace casscf
