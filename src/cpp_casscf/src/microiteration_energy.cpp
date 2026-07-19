#include "casscf/microiteration_energy.hpp"

#include "casscf/orbital_sigma.hpp"

namespace casscf {

double microiteration_exact_energy(const Matrix& U, const Matrix& A, const Tensor4& G, const Dimensions& dims) {
    const int nmo = dims.nmo;
    const int n_occupied = dims.n_occupied;
    const Matrix T = U - Matrix::Identity(nmo, nmo);

    double e = 0.0;
    for (int r = 0; r < nmo; ++r)
        for (int k = 0; k < n_occupied; ++k) e += 2.0 * T(r, k) * A(r, k);

    for (int K = 0; K < n_occupied; ++K)
        for (int L = 0; L < n_occupied; ++L)
            for (int R = 0; R < nmo; ++R)
                for (int S = 0; S < nmo; ++S) e += G(K, L, R, S) * T(R, K) * T(S, L);

    return e;
}

double microiteration_predicted_energy2(const Matrix& U, const Vector& reduced_gradient, const Matrix& A_tilde,
                                         const Tensor4& G, const Vector& step, const Dimensions& dims) {
    double energy = 2.0 * reduced_gradient.dot(step);
    const Vector sigma_reduced = orbital_sigma3(U, A_tilde, G, step, dims);
    energy += sigma_reduced.dot(step);
    return energy;
}

} // namespace casscf
