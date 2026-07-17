#include "casscf/orbital_rotation.hpp"

#include <Eigen/Eigenvalues>
#include <cmath>

namespace casscf {

Matrix build_unitary_matrix(const Matrix& Rai, const Matrix& Rvi, const Matrix& Rva,
                             const Dimensions& dims) {
    const int nmo = dims.nmo;
    const int n_in_a = dims.n_in_a;
    const int n_occupied = dims.n_occupied;
    const int n_act_orb = dims.n_act_orb;
    const int n_virtual = dims.n_virtual;

    Matrix R = Matrix::Zero(nmo, nmo);
    R.block(n_in_a, 0, n_act_orb, n_in_a) = Rai;
    R.block(0, n_in_a, n_in_a, n_act_orb) = -Rai.transpose();
    R.block(n_occupied, 0, n_virtual, n_in_a) = Rvi;
    R.block(0, n_occupied, n_in_a, n_virtual) = -Rvi.transpose();
    R.block(n_occupied, n_in_a, n_virtual, n_act_orb) = Rva;
    R.block(n_in_a, n_occupied, n_act_orb, n_virtual) = -Rva.transpose();

    Matrix R1 = -R * R;
    Eigen::SelfAdjointEigenSolver<Matrix> eig(R1);
    Vector tau_square = eig.eigenvalues();
    const Matrix& W = eig.eigenvectors();

    Vector cosine_array(nmo);
    Vector sine_product_array(nmo);
    for (int i = 0; i < nmo; ++i) {
        double t2 = tau_square[i] < 0.0 ? 0.0 : tau_square[i];
        double tau = std::sqrt(t2);
        cosine_array[i] = std::cos(tau);
        if (tau > 1e-15) {
            sine_product_array[i] = std::sin(tau) / tau;
        } else {
            // Small-angle series, matching the Python fallback exactly
            // (1 - tau^2/6 + tau^4/120), written here in terms of tau^2 = t2
            // to avoid a redundant sqrt/square round trip.
            sine_product_array[i] = 1.0 - t2 / 6.0 + (t2 * t2) / 120.0;
        }
    }

    Matrix U_delta = W * cosine_array.asDiagonal() * W.transpose();
    U_delta += W * sine_product_array.asDiagonal() * (W.transpose() * R);
    return U_delta;
}

} // namespace casscf
