#pragma once

// This file provides TWO implementations of the same matrix-free orbital
// Hessian-vector product (helper_PFCI.py:8285-8316, 8533-8686), the hottest
// path in the whole solver stack (called once per Hessian-vector product
// from GLTR, DavidsonDrivenLstrsSolver, and BfgsOperator alike):
//
//   * orbital_sigma3(...)        -- LEGACY, explicit nested-loop transcription
//     (see orbital_sigma.cpp). Kept deliberately: it is the closest
//     line-for-line correspondence to the Python's index algebra and is the
//     intended reference for the future TAMM retarget (per the developer,
//     the loop-based construction of every intermediate/orbital transform in
//     this port is kept as legacy precisely because loops translate to
//     TAMM's indexed-tensor API more directly than fused matmul chains).
//     Also serves as the correctness oracle for the fast path below.
//
//   * OrbitalSigmaOperator / orbital_sigma3_fast(...) -- the FAST path.
//     Same math, restructured into BLAS-backed Eigen matrix products, with
//     the G-derived block matrices (G_ij/G_ti/G_tu, the Python's own
//     performance decomposition) precomputed ONCE in the operator's
//     constructor and reused across every apply() -- so that within a single
//     GLTR / Davidson solve (many Hessian-vector products against the same
//     fixed U/A_tilde/G) only the per-vector matmuls are paid each apply,
//     not the O(nmo^2 * n_occupied^2) block materialization. This is what
//     makes larger side-by-side-vs-Python test cases tractable.
//
// The fast path is NOT a re-derivation-by-eye of the Python (which
// orbital_sigma.cpp's doc comment explains was deliberately avoided for the
// loop version): it is the exact matmul structure of the independently
// written reference implementation in test_orbital_sigma.cpp, which has been
// cross-validated against the loop version on random problems since this
// module was first written. test_orbital_sigma.cpp additionally checks the
// fast path here directly against the loop version.

#include "casscf/tensor_types.hpp"
#include "casscf/types.hpp"

#include <utility>
#include <vector>

namespace casscf {

// Faithful, explicit-loop port of orbital_sigma3 -> build_sigma_reduced7,
// helper_PFCI.py:8285-8316, 8533-8686 -- the matrix-free "apply the full
// orbital Hessian to a reduced-space vector" operation used (directly, or
// via get_bfgs_mv's B_0 base term) by every trust-region solve in
// microiteration_optimization6. LEGACY / TAMM-reference / fast-path oracle
// -- see this file's top comment. Prefer OrbitalSigmaOperator for repeated
// application against a fixed (U, A_tilde, G).
//
// Restricted to a single vector (the Python's num_states/pointer generality,
// used for batched bordered-eigenproblem solves, isn't needed here -- this
// module's HessianOperator interface (hessian_operator.hpp) already applies
// one vector at a time, with the "+1" border handled separately by
// BorderedHessianOperator).
//
// R_reduced: index_map_size-dim vector (the non-redundant rotation
// parameters, same packing as build_index_map). U: (nmo, nmo). A_tilde:
// (nmo, nmo), as produced by build_gradient. G: (n_occupied, n_occupied,
// nmo, nmo), as produced by build_intermediates (full block).
//
// DERIVATION NOTE (read before touching this function): the Python computes
// this via ~10 chained .transpose()/.reshape()/np.dot() calls on 2D/3D
// numpy views of the (num_states, nmo, n_occupied)-shaped R_total/temp1/
// temp2/sigma_total arrays and three reshaped "blocks" of G (G_ij, G_ti,
// G_tu). Composing that many reshape/transpose calls by eye is exactly the
// kind of thing this module's other functions deliberately avoid (see
// intermediates.hpp's doc comment) -- so this was instead worked out by
// hand, one line at a time, using the same "result[i] = original[j] where
// j[axes[k]] = i[k]" transpose-decoding method used for
// internal_transformation's K formula, then implemented as the explicit
// intermediate arrays below (R_total -> temp1 -> W -> sigma_total, mirroring
// the Python's own variable structure so it can be re-checked side by side
// with the source rather than as one large fused expression). The one place
// two of the Python's terms were algebraically combined (temp2's
// n_in_a-block and n_act-block terms, both of the form
// `sum_a sum_b temp1[a,b']*G[d,b',*,a]` over disjoint-but-complementary
// ranges of b') is flagged at that point in orbital_sigma.cpp; everything
// else is kept as separate terms exactly as the Python computes them, not
// algebraically simplified, to minimize transcription risk.
//
// Cross-validated in test_orbital_sigma.cpp against a second, independently
// written implementation that materializes G_ij/G_ti/G_tu literally (rather
// than reading directly from G) and keeps every term separate -- i.e. two
// structurally different transcriptions of the same Python source, checked
// to agree on random small problems.
Vector orbital_sigma3(const Matrix& U, const Matrix& A_tilde, const Tensor4& G, const Vector& R_reduced,
                       const Dimensions& dims);

// FAST path -- see this file's top comment. Precomputes, once, the
// U/A3_tilde and the G_ij/G_ti/G_tu block matrices the per-vector product
// contracts against (the same G.transpose(3,1,2,0)-then-block-reshape
// decomposition helper_PFCI.py:11084-11097 builds for performance), so that
// apply() is pure BLAS-backed Eigen matmuls with no per-call block
// materialization. Construct one per fixed (U, A_tilde, G) -- i.e. once per
// microiteration outer pass / BFGS reference point -- and reuse across every
// Hessian-vector product in that solve.
//
// Numerically identical to orbital_sigma3(...) above (same math, different
// evaluation order); test_orbital_sigma.cpp asserts agreement to ~1e-10.
class OrbitalSigmaOperator {
public:
    OrbitalSigmaOperator(const Matrix& U, const Matrix& A_tilde, const Tensor4& G, const Dimensions& dims);

    // B * R_reduced. R_reduced/return are index_map_size-dim, same packing
    // as orbital_sigma3(...).
    Vector apply(const Vector& R_reduced) const;

private:
    Dimensions dims_;
    std::vector<std::pair<int, int>> index_map_;
    Matrix U_;
    Matrix A3_tilde_; // A_tilde + A_tilde^T, helper_PFCI.py:8620
    // G blocks, laid out exactly as test_orbital_sigma.cpp's reference:
    //   G_ij_[(a,b),(c,d)] = G[d,b,c,a]            a,c<nmo; b,d<n_in_a
    //   G_ti_[(a,b),(c,d)] = G[d,n_in_a+b,c,a]     a,c<nmo; b<n_act; d<n_in_a
    //   G_tu_[(a,b),(c,d)] = G[n_in_a+d,n_in_a+b,c,a]  a,c<nmo; b,d<n_act
    Matrix G_ij_;
    Matrix G_ti_;
    Matrix G_tu_;
};

// Convenience one-shot: builds an OrbitalSigmaOperator and applies it once.
// For a single Hessian-vector product; use OrbitalSigmaOperator directly
// (and reuse it) when applying repeatedly against the same (U, A_tilde, G).
inline Vector orbital_sigma3_fast(const Matrix& U, const Matrix& A_tilde, const Tensor4& G, const Vector& R_reduced,
                                   const Dimensions& dims) {
    return OrbitalSigmaOperator(U, A_tilde, G, dims).apply(R_reduced);
}

} // namespace casscf
