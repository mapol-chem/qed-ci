#include "casscf/microiteration_ci_integrals_transform.hpp"

#include <cstddef>
#include <vector>

namespace casscf {

MicroiterationCiIntegralsResult microiteration_ci_integrals_transform(
    const Matrix& U, double E_core_ref, const Matrix& fock_core, const Tensor4& L, const Tensor4& J,
    const Tensor4& K, const Tensor4& active_twoeint_ref, const Matrix& d_cmo_ref, const Dimensions& dims) {
    const int nmo = dims.nmo;
    const int n_in_a = dims.n_in_a;
    const int n_act = dims.n_act_orb;

    const Matrix T = U - Matrix::Identity(nmo, nmo);

    MicroiterationCiIntegralsResult result;

    // helper_PFCI.py:8696-8709: E_core2.
    double e_core = E_core_ref;
    for (int i = 0; i < n_in_a; ++i)
        for (int r = 0; r < nmo; ++r) e_core += 4.0 * fock_core(i, r) * T(r, i);

    // temp1(i,j,r,s) = fock_core(r,s)*delta(i,j) + L(i,j,r,s), i,j<n_in_a, r,s<nmo.
    // temp2(r,i) = sum_{j<n_in_a,s<nmo} temp1(i,j,r,s) * T(s,j).
    for (int r = 0; r < nmo; ++r) {
        for (int i = 0; i < n_in_a; ++i) {
            double temp2_ri = 0.0;
            for (int j = 0; j < n_in_a; ++j) {
                for (int s = 0; s < nmo; ++s) {
                    const double temp1_ijrs = (i == j ? fock_core(r, s) : 0.0) + L(i, j, r, s);
                    temp2_ri += temp1_ijrs * T(s, j);
                }
            }
            e_core += 2.0 * T(r, i) * temp2_ri;
        }
    }
    result.E_core2 = e_core;

    // helper_PFCI.py:8711-8762: active_fock_core.
    Matrix active_fock_core = Matrix::Zero(n_act, n_act);
    // temp3(r,u) = sum_s fock_core(r,s)*U(s,n_in_a+u); active_fock_core(t,u) = sum_r U(r,n_in_a+t)*temp3(r,u).
    {
        Matrix temp3(nmo, n_act);
        for (int r = 0; r < nmo; ++r)
            for (int u = 0; u < n_act; ++u) {
                double acc = 0.0;
                for (int s = 0; s < nmo; ++s) acc += fock_core(r, s) * U(s, n_in_a + u);
                temp3(r, u) = acc;
            }
        for (int t = 0; t < n_act; ++t)
            for (int u = 0; u < n_act; ++u) {
                double acc = 0.0;
                for (int r = 0; r < nmo; ++r) acc += U(r, n_in_a + t) * temp3(r, u);
                active_fock_core(t, u) = acc;
            }
    }
    // temp4(t,u,r,i) = sum_s J(n_in_a+t,n_in_a+u,r,s)*U(s,i); active_fock_core += 2*sum_{r,i} U(r,i)*temp4(t,u,r,i);
    // active_fock_core -= 2*sum_i J(n_in_a+t,n_in_a+u,i,i).
    for (int t = 0; t < n_act; ++t) {
        for (int u = 0; u < n_act; ++u) {
            double acc = 0.0;
            for (int r = 0; r < nmo; ++r) {
                for (int i = 0; i < n_in_a; ++i) {
                    double temp4 = 0.0;
                    for (int s = 0; s < nmo; ++s) temp4 += J(n_in_a + t, n_in_a + u, r, s) * U(s, i);
                    acc += U(r, i) * temp4;
                }
            }
            active_fock_core(t, u) += 2.0 * acc;

            double diag_acc = 0.0;
            for (int i = 0; i < n_in_a; ++i) diag_acc += J(n_in_a + t, n_in_a + u, i, i);
            active_fock_core(t, u) -= 2.0 * diag_acc;
        }
    }
    // Same pattern with K, coefficients 1 instead of 2, and added (not subtracted) for the diagonal term.
    for (int t = 0; t < n_act; ++t) {
        for (int u = 0; u < n_act; ++u) {
            double acc = 0.0;
            for (int r = 0; r < nmo; ++r) {
                for (int i = 0; i < n_in_a; ++i) {
                    double temp4 = 0.0;
                    for (int s = 0; s < nmo; ++s) temp4 += K(n_in_a + t, n_in_a + u, r, s) * U(s, i);
                    acc += U(r, i) * temp4;
                }
            }
            active_fock_core(t, u) -= acc;

            double diag_acc = 0.0;
            for (int i = 0; i < n_in_a; ++i) diag_acc += K(n_in_a + t, n_in_a + u, i, i);
            active_fock_core(t, u) += diag_acc;
        }
    }
    // temp3b(r,u) = sum_{i<n_in_a,s<nmo} L(n_in_a+u,i,r,s) * T(s,i);
    // temp5(t,u) = sum_r T(r,n_in_a+t)*temp3b(r,u); active_fock_core += temp5 + temp5^T.
    {
        Matrix temp3b(nmo, n_act);
        for (int r = 0; r < nmo; ++r)
            for (int u = 0; u < n_act; ++u) {
                double acc = 0.0;
                for (int i = 0; i < n_in_a; ++i)
                    for (int s = 0; s < nmo; ++s) acc += L(n_in_a + u, i, r, s) * T(s, i);
                temp3b(r, u) = acc;
            }
        Matrix temp5(n_act, n_act);
        for (int t = 0; t < n_act; ++t)
            for (int u = 0; u < n_act; ++u) {
                double acc = 0.0;
                for (int r = 0; r < nmo; ++r) acc += T(r, n_in_a + t) * temp3b(r, u);
                temp5(t, u) = acc;
            }
        active_fock_core += temp5 + temp5.transpose();
    }
    result.active_fock_core = active_fock_core;

    // helper_PFCI.py:8764-8793: active_twoeint.
    // temp6(v,w,r,u) = sum_s J(n_in_a+v,n_in_a+w,r,s)*U(s,n_in_a+u);
    // temp7(v,w,t,u) = sum_r temp6(v,w,r,u)*U(r,n_in_a+t);
    // active_twoeint = -active_twoeint_ref + temp7 + temp7.transpose(2,3,0,1).
    Tensor4 active_twoeint(n_act, n_act, n_act, n_act);
    for (int v = 0; v < n_act; ++v)
        for (int w = 0; w < n_act; ++w)
            for (int t = 0; t < n_act; ++t)
                for (int u = 0; u < n_act; ++u) active_twoeint(v, w, t, u) = -active_twoeint_ref(v, w, t, u);
    {
        Tensor4 temp6(n_act, n_act, nmo, n_act);
        for (int v = 0; v < n_act; ++v)
            for (int w = 0; w < n_act; ++w)
                for (int r = 0; r < nmo; ++r)
                    for (int u = 0; u < n_act; ++u) {
                        double acc = 0.0;
                        for (int s = 0; s < nmo; ++s) acc += J(n_in_a + v, n_in_a + w, r, s) * U(s, n_in_a + u);
                        temp6(v, w, r, u) = acc;
                    }
        Tensor4 temp7(n_act, n_act, n_act, n_act);
        for (int v = 0; v < n_act; ++v)
            for (int w = 0; w < n_act; ++w)
                for (int t = 0; t < n_act; ++t)
                    for (int u = 0; u < n_act; ++u) {
                        double acc = 0.0;
                        for (int r = 0; r < nmo; ++r) acc += temp6(v, w, r, u) * U(r, n_in_a + t);
                        temp7(v, w, t, u) = acc;
                    }
        for (int a = 0; a < n_act; ++a)
            for (int b = 0; b < n_act; ++b)
                for (int c = 0; c < n_act; ++c)
                    for (int d = 0; d < n_act; ++d) active_twoeint(a, b, c, d) += temp7(a, b, c, d) + temp7(c, d, a, b);
    }
    // temp6b(t,v,r,w) = sum_s K(n_in_a+t,n_in_a+v,r,s)*T(s,n_in_a+w);
    // temp7b(t,u,v,w) = sum_r temp6b(t,v,r,w)*T(r,n_in_a+u);
    // active_twoeint += temp7b + temp7b.transpose(1,0,2,3) + temp7b.transpose(0,1,3,2) + temp7b.transpose(1,0,3,2).
    {
        Tensor4 temp6b(n_act, n_act, nmo, n_act);
        for (int t = 0; t < n_act; ++t)
            for (int v = 0; v < n_act; ++v)
                for (int r = 0; r < nmo; ++r)
                    for (int w = 0; w < n_act; ++w) {
                        double acc = 0.0;
                        for (int s = 0; s < nmo; ++s) acc += K(n_in_a + t, n_in_a + v, r, s) * T(s, n_in_a + w);
                        temp6b(t, v, r, w) = acc;
                    }
        Tensor4 temp7b(n_act, n_act, n_act, n_act);
        for (int t = 0; t < n_act; ++t)
            for (int u = 0; u < n_act; ++u)
                for (int v = 0; v < n_act; ++v)
                    for (int w = 0; w < n_act; ++w) {
                        double acc = 0.0;
                        for (int r = 0; r < nmo; ++r) acc += temp6b(t, v, r, w) * T(r, n_in_a + u);
                        temp7b(t, u, v, w) = acc;
                    }
        for (int a = 0; a < n_act; ++a)
            for (int b = 0; b < n_act; ++b)
                for (int c = 0; c < n_act; ++c)
                    for (int d = 0; d < n_act; ++d)
                        active_twoeint(a, b, c, d) +=
                            temp7b(a, b, c, d) + temp7b(b, a, c, d) + temp7b(a, b, d, c) + temp7b(b, a, d, c);
    }
    result.active_twoeint = active_twoeint;

    // helper_PFCI.py:8795-8796: d_cmo_out = U^T @ d_cmo_ref @ U.
    result.d_cmo = U.transpose() * d_cmo_ref * U;

    return result;
}

// ---------------------------------------------------------------------------
// FAST path. Transformed term by term from the legacy loop directly above --
// each block cites the legacy line range it replaces. See the header for the
// two categories of saving (exact algebra vs. verified ERI symmetry) and for
// why this one, unlike the other _fast twins, is only equivalent to its
// oracle for ERI-symmetric J/K.
// ---------------------------------------------------------------------------

namespace {

// (r,s) slab of a (d0,d1,nmo,nmo) RowMajor tensor at fixed (i,j).
inline Eigen::Map<const RowMajorMatrix> slab(const Tensor4& T, int i, int j, int nmo) {
    return Eigen::Map<const RowMajorMatrix>(T.data() + (static_cast<std::ptrdiff_t>(i) * T.dimension(1) + j) *
                                                           static_cast<std::ptrdiff_t>(nmo) * nmo,
                                            nmo, nmo);
}

} // namespace

MicroiterationCiIntegralsResult microiteration_ci_integrals_transform_fast(
    const Matrix& U, double E_core_ref, const Matrix& fock_core, const Tensor4& L, const Tensor4& J,
    const Tensor4& K, const Tensor4& active_twoeint_ref, const Matrix& d_cmo_ref, const Dimensions& dims) {
    const int nmo = dims.nmo;
    const int n_in_a = dims.n_in_a;
    const int n_act = dims.n_act_orb;
    const int nmo2 = nmo * nmo;

    const Matrix T = U - Matrix::Identity(nmo, nmo);

    // Column blocks used throughout. Materialized (not left as expressions) so
    // each is one contiguous operand for the GEMMs below.
    const Matrix Uocc = U.leftCols(n_in_a);
    const Matrix Uact = U.middleCols(n_in_a, n_act);
    const Matrix Tocc = T.leftCols(n_in_a);
    const Matrix Tact = T.middleCols(n_in_a, n_act);

    MicroiterationCiIntegralsResult result;

    // --- E_core2, legacy lines 17-35 ---------------------------------------
    // 4*sum_{i,r} fock_core(i,r)*T(r,i) == 4*trace(fock_core[occ,:] * Tocc).
    double e_core = E_core_ref + 4.0 * (fock_core.topRows(n_in_a) * Tocc).trace();

    // The delta(i,j) half of temp1 collapses to a single congruence trace:
    //   sum_{i,r,s} T(r,i) fock_core(r,s) T(s,i) == trace(Tocc^T fock_core Tocc).
    e_core += 2.0 * (Tocc.transpose() * fock_core * Tocc).trace();

    // The L half: sum_{i,j} Tocc(:,i)^T Lslab(i,j) Tocc(:,j) -- n_in_a^2
    // GEMVs instead of a quadruple scalar loop. (No symmetry used: L is
    // 4K - K^T - J, which carries none of the individual swap symmetries.)
    {
        double acc = 0.0;
        for (int i = 0; i < n_in_a; ++i)
            for (int j = 0; j < n_in_a; ++j)
                acc += Tocc.col(i).dot(slab(L, i, j, nmo) * Tocc.col(j));
        e_core += 2.0 * acc;
    }
    result.E_core2 = e_core;

    // --- active_fock_core, legacy lines 38-112 -----------------------------
    // Term 1, legacy lines 40-54: a plain congruence transform.
    Matrix active_fock_core = Uact.transpose() * fock_core * Uact;

    // Terms 2+3, legacy lines 57-91. EXACT-ALGEBRA saving (see header): the
    // sum over the inactive index factorizes into a density matrix, dropping a
    // factor of n_in_a, and the +2*J / -1*K terms merge into one GEVM.
    {
        const RowMajorMatrix D = Uocc * Uocc.transpose(); // (nmo, nmo), symmetric

        // Active-active (t,u) blocks of J/K packed as (n_act^2, nmo^2). For
        // fixed t the n_act rows are contiguous in the source, so each is one
        // chunk copy -- same packing as build_intermediates_fast.
        const int n_occupied = dims.n_occupied;
        RowMajorMatrix JKmat(static_cast<Eigen::Index>(n_act) * n_act, nmo2);
        for (int t = 0; t < n_act; ++t) {
            const std::ptrdiff_t src =
                (static_cast<std::ptrdiff_t>(n_in_a + t) * n_occupied + n_in_a) * static_cast<std::ptrdiff_t>(nmo2);
            Eigen::Map<const RowMajorMatrix> Jrows(J.data() + src, n_act, nmo2);
            Eigen::Map<const RowMajorMatrix> Krows(K.data() + src, n_act, nmo2);
            JKmat.middleRows(static_cast<Eigen::Index>(t) * n_act, n_act) = 2.0 * Jrows - Krows;
        }
        const Eigen::Map<const Vector> dvec(D.data(), nmo2);
        const Vector flat = JKmat * dvec;
        for (int t = 0; t < n_act; ++t)
            for (int u = 0; u < n_act; ++u) active_fock_core(t, u) += flat(t * n_act + u);
    }

    // The two diagonal corrections, legacy lines 69-71 and 87-89:
    //   -2*sum_i J(t,u,i,i) + sum_i K(t,u,i,i)
    for (int t = 0; t < n_act; ++t) {
        for (int u = 0; u < n_act; ++u) {
            double acc = 0.0;
            for (int i = 0; i < n_in_a; ++i)
                acc += -2.0 * J(n_in_a + t, n_in_a + u, i, i) + K(n_in_a + t, n_in_a + u, i, i);
            active_fock_core(t, u) += acc;
        }
    }

    // temp3b/temp5, legacy lines 94-111.
    {
        Matrix temp3b = Matrix::Zero(nmo, n_act);
        for (int u = 0; u < n_act; ++u)
            for (int i = 0; i < n_in_a; ++i)
                temp3b.col(u).noalias() += slab(L, n_in_a + u, i, nmo) * Tocc.col(i);
        const Matrix temp5 = Tact.transpose() * temp3b;
        active_fock_core += temp5 + temp5.transpose();
    }
    result.active_fock_core = active_fock_core;

    // --- active_twoeint, legacy lines 118-176 ------------------------------
    // Both parts are two-index transforms of one (r,s) slab per active pair,
    // each computed over a triangle only (ERI symmetry, see header).
    //   N(v,w) = Uact^T Jslab(v,w) Uact,  N(w,v) == N(v,w)
    //   M(t,v) = Tact^T Kslab(t,v) Tact,  M(v,t) == M(t,v)^T
    std::vector<Matrix> N(static_cast<std::size_t>(n_act) * n_act);
    std::vector<Matrix> M(static_cast<std::size_t>(n_act) * n_act);
    for (int v = 0; v < n_act; ++v) {
        for (int w = v; w < n_act; ++w) {
            const Matrix nvw = Uact.transpose() * slab(J, n_in_a + v, n_in_a + w, nmo) * Uact;
            N[static_cast<std::size_t>(v) * n_act + w] = nvw;
            if (w != v) N[static_cast<std::size_t>(w) * n_act + v] = nvw;
        }
    }
    for (int t = 0; t < n_act; ++t) {
        for (int v = t; v < n_act; ++v) {
            const Matrix mtv = Tact.transpose() * slab(K, n_in_a + t, n_in_a + v, nmo) * Tact;
            M[static_cast<std::size_t>(t) * n_act + v] = mtv;
            if (v != t) M[static_cast<std::size_t>(v) * n_act + t] = mtv.transpose();
        }
    }

    // Assembly, legacy lines 122/142-145 (J) and 169-174 (K):
    //   -ref + temp7(a,b,c,d) + temp7(c,d,a,b)
    //        + temp7b(a,b,c,d) + temp7b(b,a,c,d) + temp7b(a,b,d,c) + temp7b(b,a,d,c)
    // with temp7(a,b,c,d) = N(a,b)(c,d) and temp7b(a,b,c,d) = M(a,c)(b,d).
    Tensor4 active_twoeint(n_act, n_act, n_act, n_act);
    for (int a = 0; a < n_act; ++a) {
        for (int b = 0; b < n_act; ++b) {
            const Matrix& Nab = N[static_cast<std::size_t>(a) * n_act + b];
            for (int c = 0; c < n_act; ++c) {
                const Matrix& Mac = M[static_cast<std::size_t>(a) * n_act + c];
                const Matrix& Mbc = M[static_cast<std::size_t>(b) * n_act + c];
                for (int d = 0; d < n_act; ++d) {
                    const Matrix& Ncd = N[static_cast<std::size_t>(c) * n_act + d];
                    const Matrix& Mad = M[static_cast<std::size_t>(a) * n_act + d];
                    const Matrix& Mbd = M[static_cast<std::size_t>(b) * n_act + d];
                    active_twoeint(a, b, c, d) = -active_twoeint_ref(a, b, c, d) + Nab(c, d) + Ncd(a, b) +
                                                 Mac(b, d) + Mbc(a, d) + Mad(b, c) + Mbd(a, c);
                }
            }
        }
    }
    result.active_twoeint = active_twoeint;

    // d_cmo, legacy line 179.
    result.d_cmo = U.transpose() * d_cmo_ref * U;

    return result;
}

} // namespace casscf
