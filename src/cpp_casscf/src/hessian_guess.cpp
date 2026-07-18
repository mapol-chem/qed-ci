#include "casscf/hessian_guess.hpp"

#include <utility>

namespace casscf {

namespace {

// U(:,col_r) . (G(k,l,:,:) @ U(:,col_s)) -- matches the Python's
// temp1=G[k,l,:,:]; temp2=U[:,col_s]; temp3=U[:,col_r];
// temp4=dot(temp1,temp2); result=dot(temp3,temp4) pattern, reused for all
// four G-based terms in hessian_guess with different (k,l,col_s,col_r).
double contract_G_slice(const Tensor4& G, int k, int l, const Matrix& U, int col_s, int col_r) {
    double acc = 0.0;
    const int nmo = static_cast<int>(U.rows());
    for (int p = 0; p < nmo; ++p) {
        double gv = 0.0;
        for (int q = 0; q < nmo; ++q) gv += G(k, l, p, q) * U(q, col_s);
        acc += U(p, col_r) * gv;
    }
    return acc;
}

} // namespace

OrbitalHessianGuessProvider::OrbitalHessianGuessProvider(Matrix U, Matrix sym_A_tilde,
                                                           Vector reduced_gradient, Tensor4 G,
                                                           std::vector<std::pair<int, int>> index_map,
                                                           int n_occupied)
    : U_(std::move(U)),
      sym_A_tilde_(std::move(sym_A_tilde)),
      reduced_gradient_(std::move(reduced_gradient)),
      G_(std::move(G)),
      index_map_(std::move(index_map)),
      n_occupied_(n_occupied) {}

void OrbitalHessianGuessProvider::guess_block(const std::vector<int>& indices, Matrix& hessian_block,
                                               Vector& gradient_block) const {
    const int dim1 = static_cast<int>(indices.size());
    hessian_block = Matrix::Zero(dim1, dim1);
    gradient_block = Vector::Zero(dim1);

    // helper_PFCI.py:16665-16712
    for (int i = 0; i < dim1; ++i) {
        const int index1 = indices[i];
        const int r = index_map_[index1].first;
        const int k = index_map_[index1].second;
        gradient_block(i) = reduced_gradient_(index1);

        for (int j = 0; j < dim1; ++j) {
            const int index2 = indices[j];
            const int s = index_map_[index2].first;
            const int l = index_map_[index2].second;

            double a = contract_G_slice(G_, k, l, U_, s, r);
            a -= 0.5 * (k == l ? 1.0 : 0.0) * sym_A_tilde_(r, s);
            a -= 0.5 * (r == s ? 1.0 : 0.0) * sym_A_tilde_(k, l);

            if (r < n_occupied_) {
                a -= contract_G_slice(G_, r, l, U_, s, k);
                a += 0.5 * (r == l ? 1.0 : 0.0) * sym_A_tilde_(k, s);
            }
            if (s < n_occupied_) {
                a -= contract_G_slice(G_, k, s, U_, l, r);
                a += 0.5 * (k == s ? 1.0 : 0.0) * sym_A_tilde_(r, l);
            }
            if (r < n_occupied_ && s < n_occupied_) {
                a += contract_G_slice(G_, r, s, U_, l, k);
            }

            hessian_block(i, j) = a;
        }
    }
}

} // namespace casscf
