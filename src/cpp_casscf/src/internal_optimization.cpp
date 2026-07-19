#include "casscf/internal_optimization.hpp"

#include <cmath>

namespace casscf {

InternalTransformationResult internal_transformation(const Matrix& U, const Matrix& occupied_h1,
                                                       const Matrix& occupied_d_cmo,
                                                       const Tensor4& occupied_J, const Dimensions& dims) {
    const int n = dims.n_occupied;
    const Matrix U_occ = U.topLeftCorner(n, n);

    // helper_PFCI.py:6482-6495: the standard 4-index "quarter transformation"
    // cascade, one leg at a time.
    // temp1(k,l,m,s) = sum_n occupied_J(k,l,m,n) * U(n,s)
    Tensor4 temp1(n, n, n, n);
    for (int k = 0; k < n; ++k)
        for (int l = 0; l < n; ++l)
            for (int m = 0; m < n; ++m)
                for (int s = 0; s < n; ++s) {
                    double acc = 0.0;
                    for (int nn = 0; nn < n; ++nn) acc += occupied_J(k, l, m, nn) * U_occ(nn, s);
                    temp1(k, l, m, s) = acc;
                }
    // temp2(k,l,r,s) = sum_m temp1(k,l,m,s) * U(m,r)
    Tensor4 temp2(n, n, n, n);
    for (int k = 0; k < n; ++k)
        for (int l = 0; l < n; ++l)
            for (int r = 0; r < n; ++r)
                for (int s = 0; s < n; ++s) {
                    double acc = 0.0;
                    for (int m = 0; m < n; ++m) acc += temp1(k, l, m, s) * U_occ(m, r);
                    temp2(k, l, r, s) = acc;
                }
    // temp1(k,q,r,s) = sum_l temp2(k,l,r,s) * U(l,q)  -- reuse temp1's storage
    for (int k = 0; k < n; ++k)
        for (int q = 0; q < n; ++q)
            for (int r = 0; r < n; ++r)
                for (int s = 0; s < n; ++s) {
                    double acc = 0.0;
                    for (int l = 0; l < n; ++l) acc += temp2(k, l, r, s) * U_occ(l, q);
                    temp1(k, q, r, s) = acc;
                }
    // temp2(p,q,r,s) = sum_k temp1(k,q,r,s) * U(k,p) -- final transformed J
    for (int p = 0; p < n; ++p)
        for (int q = 0; q < n; ++q)
            for (int r = 0; r < n; ++r)
                for (int s = 0; s < n; ++s) {
                    double acc = 0.0;
                    for (int k = 0; k < n; ++k) acc += temp1(k, q, r, s) * U_occ(k, p);
                    temp2(p, q, r, s) = acc;
                }

    InternalTransformationResult result;
    result.J = temp2; // helper_PFCI.py:6496
    // K(i,j,k,l) = J_transformed(j,l,i,k), helper_PFCI.py:6497-6499
    // (occupied_twoeint2.transpose(1,3,0,2))
    result.K = Tensor4(n, n, n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k)
                for (int l = 0; l < n; ++l) result.K(i, j, k, l) = temp2(j, l, i, k);

    // h1 = U^T @ occupied_h1 @ U, d_cmo1 = U^T @ occupied_d_cmo @ U --
    // helper_PFCI.py:6502-6517, the two chained einsums multiplied out.
    result.h1 = U_occ.transpose() * occupied_h1 * U_occ;
    result.d_cmo1 = U_occ.transpose() * occupied_d_cmo * U_occ;

    return result;
}

InternalOptimizationEnergyResult internal_optimization_exact_energy(
    double E0, const Matrix& eigenvecs, const Matrix& occupied_h1, const Matrix& occupied_d_cmo,
    const Tensor4& occupied_J, const Tensor4& occupied_K, int hard_case, const StateAverageData& sad,
    const Dimensions& dims) {
    const int n_occ = dims.n_occupied;
    const int n_in_a = dims.n_in_a;
    const int n_act = dims.n_act_orb;

    // helper_PFCI.py:6740-6743: occupied_fock_core = occupied_h1 +
    // 2*einsum("jjrs->rs", J[:n_in_a,:n_in_a]) - einsum("jjrs->rs", K[:n_in_a,:n_in_a]).
    Matrix occupied_fock_core = occupied_h1;
    for (int r = 0; r < n_occ; ++r) {
        for (int s = 0; s < n_occ; ++s) {
            double acc = 0.0;
            for (int j = 0; j < n_in_a; ++j) acc += 2.0 * occupied_J(j, j, r, s) - occupied_K(j, j, r, s);
            occupied_fock_core(r, s) += acc;
        }
    }

    // helper_PFCI.py:6745-6747
    double E_core = 0.0;
    for (int j = 0; j < n_in_a; ++j) E_core += occupied_h1(j, j) + occupied_fock_core(j, j);

    // helper_PFCI.py:6749-6763: Frobenius inner products between the
    // active-active blocks and the state-averaged RDMs.
    const Matrix fock_active = occupied_fock_core.block(n_in_a, n_in_a, n_act, n_act);
    const double active_one_e_energy = fock_active.cwiseProduct(sad.D_tu_avg).sum();

    double active_two_e_energy = 0.0;
    for (int t = 0; t < n_act; ++t)
        for (int u = 0; u < n_act; ++u)
            for (int v = 0; v < n_act; ++v)
                for (int w = 0; w < n_act; ++w)
                    active_two_e_energy +=
                        occupied_J(n_in_a + t, n_in_a + u, n_in_a + v, n_in_a + w) * sad.D_tuvw_avg(t, u, v, w);
    active_two_e_energy *= 0.5;

    const Matrix d_cmo_active = occupied_d_cmo.block(n_in_a, n_in_a, n_act, n_act);
    const double active_one_pe_energy = -std::sqrt(sad.omega / 2.0) * d_cmo_active.cwiseProduct(sad.Dpe_tu_avg).sum();

    const double ci_dependent_energy = calculate_ci_dependent_energy(eigenvecs, occupied_d_cmo, sad.weight, sad.N_p,
                                                                       sad.num_det, sad.omega, sad.d_exp, n_in_a);

    const double sum_energy = active_one_e_energy + active_two_e_energy + active_one_pe_energy + E_core + sad.Enuc +
                               sad.d_c + ci_dependent_energy;

    InternalOptimizationEnergyResult result;
    result.energy_change = sum_energy - E0;
    result.accepted = (result.energy_change <= 0.0) || (hard_case == 2);

    if (result.accepted) {
        // helper_PFCI.py:6816-6828. occupied_J/K/h1/d_cmo/fock_core here are
        // the (n_occupied)^4-or-(n_occupied,n_occupied) patches the Python
        // writes into the occupied-occupied sub-block of the larger
        // persistent self.occupied_* arrays (first n_occupied of the last
        // two J/K axes, all of the h1/d_cmo/fock_core axes) -- the caller
        // owns applying that sub-block write against its own larger context.
        result.occupied_J = occupied_J;
        result.occupied_K = occupied_K;
        result.occupied_h1 = occupied_h1;
        result.occupied_d_cmo = occupied_d_cmo;
        result.occupied_fock_core = occupied_fock_core;
        result.E_core = E_core;

        // helper_PFCI.py:6854-6856: gkl2 = fock_core[act,act] - 0.5 *
        // einsum("kjjl->kl", occupied_J[act,act,act,act]). Since the
        // just-updated self.occupied_J agrees with the local occupied_J
        // parameter over the whole [0, n_occupied) range on all 4 axes (see
        // the sub-block-write comment above), this reads directly from the
        // local occupied_J.
        result.gkl2 = Matrix(n_act, n_act);
        for (int k = 0; k < n_act; ++k) {
            for (int l = 0; l < n_act; ++l) {
                double acc = 0.0;
                for (int j = 0; j < n_act; ++j)
                    acc += occupied_J(n_in_a + k, n_in_a + j, n_in_a + j, n_in_a + l);
                result.gkl2(k, l) = fock_active(k, l) - 0.5 * acc;
            }
        }
    }

    return result;
}

double internal_optimization_predicted_energy(const Vector& gradient_ai, const Matrix& hessian_ai,
                                                const Vector& Rai) {
    return 2.0 * gradient_ai.dot(Rai) + Rai.dot(hessian_ai * Rai);
}

double step_control(double ratio, double trust_radius) {
    if (ratio >= 0.0 && ratio <= 0.25) trust_radius = 0.7 * trust_radius;
    if (ratio > 0.75) {
        trust_radius = 1.2 * trust_radius;
        if (trust_radius > 0.75) trust_radius = 0.75;
    }
    return trust_radius;
}

} // namespace casscf
