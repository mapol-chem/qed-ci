#pragma once

// FLAGGED FOR FUTURE ACCELERATION: this is the hottest path in the whole
// solver stack (called once per Hessian-vector product from GLTR,
// DavidsonDrivenLstrsSolver, and BfgsOperator alike). Currently explicit
// O(n^4)-ish nested loops, chosen for transcription safety over a function
// this intricate (see the derivation note below) -- worth revisiting for
// BLAS-backed Eigen matrix ops (or the G_ij/G_ti/G_tu block decomposition
// the Python itself uses for performance) once correctness is fully
// settled. Also flagged for re-review by a stronger reasoning model, same
// as bfgs_operator.hpp.

#include "casscf/tensor_types.hpp"
#include "casscf/types.hpp"

namespace casscf {

// Faithful port of orbital_sigma3 -> build_sigma_reduced7,
// helper_PFCI.py:8285-8316, 8533-8686 -- the matrix-free "apply the full
// orbital Hessian to a reduced-space vector" operation used (directly, or
// via get_bfgs_mv's B_0 base term) by every trust-region solve in
// microiteration_optimization6.
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

} // namespace casscf
