#pragma once

#include "casscf/davidson_augmented_hessian_solver.hpp"
#include "casscf/tensor_types.hpp"
#include "casscf/types.hpp"

#include <utility>
#include <vector>

namespace casscf {

// Faithful port of hessian_guess / build_orbital_hessian_guess,
// helper_PFCI.py:16614-16712 (the latter a @nb.njit-compiled staticmethod).
// Computes individual elements of the full (matrix-free) reduced orbital
// Hessian at specific index_map-pair coordinates on demand, rather than
// materializing the whole index_map_size x index_map_size matrix -- this is
// what lets DavidsonAugmentedHessianSolver build a small "guess subspace"
// Hessian block cheaply. The index *selection* (ranking by |gradient_i /
// diagonal_i|, helper_PFCI.py:15527-15535) already lives in
// DavidsonAugmentedHessianSolver itself (idx_hessian_); this class only
// answers "what is the Hessian element at these specific coordinates."
//
// G: (n_occupied, n_occupied, nmo, nmo), the same G produced by
// build_intermediates for the current U.
// U: (nmo, nmo), the current orbital rotation (self.U2).
// sym_A_tilde: (nmo, nmo) = A_tilde_full + A_tilde_full.transpose(), where
// A_tilde_full is build_gradient's A_tilde embedded back into its full
// (nmo, nmo) shape (virtual-orbital columns zero) -- see
// embed_and_symmetrize_A_tilde in intermediates.hpp, which is exactly
// helper_PFCI.py:15544's `sym_A_tilde = A_tilde + A_tilde.T`.
// reduced_gradient: (index_map_size).
// index_map: from build_index_map (tensor_types.hpp) -- pairs are (s, r)
// with s the larger index, r the smaller, matching
// `r = index_map[i][0]; k = index_map[i][1]` in the Python (so this port's
// "r" below is the Python's "r"/first/larger, "k" is the Python's
// "k"/second/smaller -- kept as (r, k) throughout to mirror the source).
class OrbitalHessianGuessProvider final : public HessianGuessProvider {
public:
    OrbitalHessianGuessProvider(Matrix U, Matrix sym_A_tilde, Vector reduced_gradient, Tensor4 G,
                                 std::vector<std::pair<int, int>> index_map, int n_occupied);

    void guess_block(const std::vector<int>& indices, Matrix& hessian_block,
                      Vector& gradient_block) const override;

private:
    Matrix U_;
    Matrix sym_A_tilde_;
    Vector reduced_gradient_;
    Tensor4 G_;
    std::vector<std::pair<int, int>> index_map_;
    int n_occupied_;
};

} // namespace casscf
