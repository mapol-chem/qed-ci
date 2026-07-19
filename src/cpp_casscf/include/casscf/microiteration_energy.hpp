#pragma once

#include "casscf/tensor_types.hpp"
#include "casscf/types.hpp"

namespace casscf {

// Faithful port of microiteration_exact_energy, helper_PFCI.py:8800-8826.
// U, A: (nmo, nmo); G: (n_occupied, n_occupied, nmo, nmo), as produced by
// build_intermediates (full block).
//
// The Python computes this via `T_rk = (U - I)[:, :n_occupied].flatten()`,
// `B = 2*A_rk + np.dot(T_rk, G1)` (G1 the same (nmo*n_occupied)^2 flattened
// matrix used by orbital_sigma3's caller), `E = np.dot(T_rk, B)` -- worked
// out by hand (same method as orbital_sigma3) to the closed form below:
// `E = 2 * sum_{r<nmo,k<n_occupied} T(r,k)*A(r,k)
//      + sum_{K,L<n_occupied, R,S<nmo} G(K,L,R,S) * T(R,K) * T(S,L)`,
// where T = U - I. Implemented directly as explicit loops on the raw G
// tensor rather than replicating the reshape/dot route, since this
// particular derivation is a plain double contraction with no ambiguity
// once decoded (unlike orbital_sigma3, nothing here needed keeping terms
// artificially separate for transcription safety).
double microiteration_exact_energy(const Matrix& U, const Matrix& A, const Tensor4& G, const Dimensions& dims);

// Faithful port of microiteration_predicted_energy2, helper_PFCI.py:9455-9467
// -- the active predicted-energy function in microiteration_optimization6
// (microiteration_predicted_energy, helper_PFCI.py:9449-9453, the plain
// dense-Hessian quadratic form, is referenced only in a commented-out line
// there and never actually called; not ported).
//
// SUBSTITUTION: the Python calls `self.orbital_sigma(U, A_tilde, G, step,
// ...)`, which dispatches to `build_sigma_reduced4` -- a *different*
// function from `build_sigma_reduced7` (orbital_sigma3.hpp/.cpp), but one
// that computes the exact same matrix-free Hessian-vector product via a
// single (nmo*n_occupied, nmo*n_occupied) matrix multiply instead of the
// three smaller G_ij/G_ti/G_tu block multiplies -- confirmed by hand-
// deriving build_sigma_reduced4's middle step and finding it reduces to
// `sum_{a,bp<n_occupied} temp1(a,bp)*G(d,bp,c,a)` for every d (not just
// d < n_in_a), which is exactly what orbital_sigma3's own d < n_in_a case
// combines to -- i.e. the two Python functions are two different
// implementation strategies for the same operation on the same
// (physically-symmetric) G, not two different formulas. Reusing the
// already-ported, already-cross-validated orbital_sigma3 here avoids
// re-porting a second ~150-line near-duplicate kernel, matching this
// module's existing precedent of not re-porting the LSTRS bisection loop
// a second time where the Python has it duplicated (see
// lstrs_bisection_core.hpp).
double microiteration_predicted_energy2(const Matrix& U, const Vector& reduced_gradient, const Matrix& A_tilde,
                                         const Tensor4& G, const Vector& step, const Dimensions& dims);

} // namespace casscf
