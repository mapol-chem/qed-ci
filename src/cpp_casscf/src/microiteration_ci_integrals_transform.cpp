#include "casscf/microiteration_ci_integrals_transform.hpp"

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

} // namespace casscf
