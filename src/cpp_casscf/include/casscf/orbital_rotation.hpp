#pragma once

#include "casscf/types.hpp"

namespace casscf {

// Port of build_unitary_matrix, helper_PFCI.py:6131-6164: builds the
// orthogonal matrix exp(R) for the antisymmetric rotation generator R
// assembled from three off-diagonal blocks (active-inactive, virtual-
// inactive, virtual-active). Computed via eigendecomposition of -R^2 rather
// than a generic matrix exponential: R is real antisymmetric, so -R^2 is
// symmetric PSD and shares eigenvectors with R, which lets cos/sin of the
// rotation angles be applied directly in that eigenbasis.
//
// Rai: n_act_orb x n_in_a, Rvi: n_virtual x n_in_a, Rva: n_virtual x n_act_orb
// (the three non-redundant off-diagonal blocks of R; the diagonal and the
// active-active/inactive-inactive/virtual-virtual blocks are always zero).
Matrix build_unitary_matrix(const Matrix& Rai, const Matrix& Rvi, const Matrix& Rva,
                             const Dimensions& dims);

} // namespace casscf
