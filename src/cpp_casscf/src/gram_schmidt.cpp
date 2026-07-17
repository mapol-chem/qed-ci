#include "casscf/gram_schmidt.hpp"

#include <cmath>

namespace casscf {

int gram_schmidt_orthogonalize(Matrix& Q) {
    const int rows = static_cast<int>(Q.rows());
    int L = 0;
    for (int k = 0; k < rows; ++k) {
        if (L > 0) {
            for (int i = 0; i < L; ++i) {
                const double dotval = Q.row(i).dot(Q.row(k));
                Q.row(k) -= dotval * Q.row(i);
            }
            // reorthogonalization
            for (int i = 0; i < L; ++i) {
                const double dotval = Q.row(i).dot(Q.row(k));
                Q.row(k) -= dotval * Q.row(i);
            }
        }
        const double normval = Q.row(k).norm();
        if (normval > 1e-20) {
            Q.row(L) = Q.row(k) / normval;
            ++L;
        }
    }
    return L;
}

void gram_schmidt_add(Matrix& Q, int rows, int rows2) {
    for (int k = rows; k < rows + rows2; ++k) {
        for (int i = 0; i < k; ++i) {
            const double dotval = Q.row(i).dot(Q.row(k));
            Q.row(k) -= dotval * Q.row(i);
        }
        // reorthogonalization
        for (int i = 0; i < k; ++i) {
            const double dotval = Q.row(i).dot(Q.row(k));
            Q.row(k) -= dotval * Q.row(i);
        }
        const double normval = Q.row(k).norm();
        if (normval > 1e-20) {
            Q.row(k) /= normval;
        }
    }
}

} // namespace casscf
