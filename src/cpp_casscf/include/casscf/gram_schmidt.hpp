#pragma once

#include "casscf/types.hpp"

namespace casscf {

// Port of gram_schmidt_orthogonalization, ci_solver.c:3509-3545. Modified
// Gram-Schmidt (with one reorthogonalization pass) over the rows of Q,
// compacted in place: row k is orthogonalized against all already-accepted
// rows [0, L), and only kept (written back at row L, normalized, L
// incremented) if its residual norm exceeds 1e-20. NOTE: faithfully
// reproduces the source's behavior of not truncating the matrix when rows
// are dropped -- trailing rows beyond the returned count hold stale
// (unnormalized) data, exactly as in the Python/C original. Returns the
// number of surviving orthonormal rows (== Q.rows() in the overwhelmingly
// common case of no exact linear dependence).
int gram_schmidt_orthogonalize(Matrix& Q);

// Port of gram_schmidt_add, ci_solver.c:3549-3583. Orthogonalizes rows
// [rows, rows + rows2) of Q against the first `rows` rows (assumed already
// orthonormal) and against each other, appending them in place. Unlike
// gram_schmidt_orthogonalize, this does not compact on near-zero norm --
// matching the source, which leaves such a row un-normalized rather than
// dropping it (the caller unconditionally treats all rows2 vectors as
// added).
void gram_schmidt_add(Matrix& Q, int rows, int rows2);

} // namespace casscf
