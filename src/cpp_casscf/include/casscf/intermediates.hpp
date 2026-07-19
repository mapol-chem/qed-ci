#pragma once

#include "casscf/tensor_types.hpp"
#include "casscf/types.hpp"

namespace casscf {

// Bundles the state-averaged RDMs and physical constants that
// build_intermediates(_internal), calculate_ci_dependent_energy, and
// internal_optimization_exact_energy / the (future) microiteration
// equivalent all need -- these are recomputed once per macroiteration (by
// whatever eventually implements CiStateAverageSolver) and then read
// (never mutated) by everything downstream in that macroiteration. A new,
// purely additive grouping -- the already-validated build_intermediates*/
// calculate_ci_dependent_energy signatures are deliberately left exactly as
// tested rather than retrofitted onto this, to avoid touching passing code.
struct StateAverageData {
    Matrix D_tu_avg;     // (n_act_orb, n_act_orb)
    Tensor4 D_tuvw_avg;  // (n_act_orb, n_act_orb, n_act_orb, n_act_orb)
    Matrix Dpe_tu_avg;   // (n_act_orb, n_act_orb)
    Vector weight;       // (davidson_roots) -- state-average weights
    int N_p = 0;
    int num_det = 0;
    double omega = 0.0;
    double Enuc = 0.0;
    double d_c = 0.0;
    double d_exp = 0.0;
};

// Faithful port of calculate_off_diagonal_photon_constant, helper_PFCI.py:6105-6151.
// eigenvecs: one row per state-averaged CI root (davidson_roots rows), each
// row the photon-number-basis-stacked CI vector of length (N_p+1)*num_det
// (row i is exactly self's `eigenvecs[i]` in the Python, flat, not yet
// reshaped). weight: per-root state-average weights (self.weight).
//
// The Python reshapes each root's flat vector to (N_p+1, num_det) then
// transposes to (num_det, N_p+1) before taking column dot products between
// adjacent photon-number blocks m and m-1 (or m and m+1) -- since a
// numpy .reshape((N_p+1, num_det)) is C-order, column m of the transposed
// array is exactly the contiguous length-num_det segment of the flat row
// starting at offset m*num_det. So this reduces to plain segment dot
// products on the flat row, no actual reshape/transpose needed here.
double calculate_off_diagonal_photon_constant(const Matrix& eigenvecs, const Vector& weight,
                                               int N_p, int num_det, double omega);

// Faithful port of calculate_ci_dependent_energy, helper_PFCI.py:6047-6113.
// Structurally the same per-root, per-photon-block loop as
// calculate_off_diagonal_photon_constant above (same block() segment-dot
// pattern), but with a `(d_exp - d_diag)` factor, no sign flip, and an
// extra `photon_energy` accumulator folded into the same loop -- ported
// separately rather than sharing code with that function since the two
// differ in more than a sign (this one is not simply "the negation of"
// that one). occupied_d_cmo: (n_occupied, n_occupied) or larger, only its
// (n_in_a, n_in_a) top-left block is read (matches
// occupied_d_cmo[:n_in_a, :n_in_a] in the Python).
double calculate_ci_dependent_energy(const Matrix& eigenvecs, const Matrix& occupied_d_cmo,
                                      const Vector& weight, int N_p, int num_det, double omega,
                                      double d_exp, int n_in_a);

struct SmallBlockIntermediates {
    Matrix A;   // (n_occupied, n_occupied)
    Tensor4 G;  // (n_occupied, n_occupied, n_occupied, n_occupied)
};

// Faithful port of build_intermediates_internal, helper_PFCI.py:5569-5747
// (rot_dim == n_occupied always, since this is only ever called that way
// from internal_optimization3). off_diagonal_constant is
// calculate_off_diagonal_photon_constant's result, passed in rather than
// recomputed here (it's a pure function of eigenvecs alone -- an interface
// simplification, not a change to what's computed).
//
// occupied_fock_core, occupied_d_cmo: (n_occupied, n_occupied).
// occupied_J, occupied_K: (n_occupied, n_occupied, n_occupied, n_occupied).
// D_tu_avg, Dpe_tu_avg: (n_act_orb, n_act_orb). D_tuvw_avg: (n_act_orb x4).
//
// Contraction terms are written as explicit index loops rather than chained
// Eigen::Tensor contract()/shuffle() calls -- slower to write, but each term
// maps directly onto its einsum string with no risk of a silent axis-order
// bug from composing tensor-library primitives incorrectly. Dimensions here
// are small (orbital-space), so this isn't a performance concern; this
// module's stated purpose is a TAMM-portable *reference* implementation,
// not a performance-tuned one.
SmallBlockIntermediates build_intermediates_internal(const Matrix& occupied_fock_core,
                                                       const Matrix& occupied_d_cmo,
                                                       const Tensor4& occupied_J,
                                                       const Tensor4& occupied_K,
                                                       const Matrix& D_tu_avg,
                                                       const Tensor4& D_tuvw_avg,
                                                       const Matrix& Dpe_tu_avg,
                                                       double off_diagonal_constant,
                                                       double omega, const Dimensions& dims);

struct FullBlockIntermediates {
    Matrix A;          // (nmo, nmo)
    Tensor4 G;         // (n_occupied, n_occupied, nmo, nmo)
    Matrix fock_core;  // (nmo, nmo) -- side output, not consumed by build_gradient itself but
                        // needed by the (not-yet-ported) energy bookkeeping around it
};

// Faithful port of build_intermediates, helper_PFCI.py:5748-5929, with
// rot_dim hardcoded to nmo (full_space == True): that's the only value ever
// used -- every active call site (microiteration_optimization6) passes
// full_space=True, and the other call sites that pass full_space are all
// unreachable dead code (microiteration_optimization5 / ah_orbital_optimization,
// neither has any caller -- confirmed by grep).
//
// IMPORTANT: J and K here are (n_occupied, n_occupied, nmo, nmo), NOT
// (nmo, nmo, nmo, nmo) -- self.J/self.K only ever store ERI blocks with
// their first two indices restricted to occupied orbitals (see the
// commented-out shape declaration at helper_PFCI.py:5480-5481 and
// c_full_transformation_macroiteration's output array shapes). This is
// smaller than (and NOT simply a differently-sliced view of)
// build_intermediates_internal's occupied_J/occupied_K, which are the
// fully-occupied-restricted (n_occupied)^4 block.
//
// off_diagonal_constant: same interface simplification as
// build_intermediates_internal -- computed by the caller via
// calculate_off_diagonal_photon_constant, passed in rather than recomputed.
FullBlockIntermediates build_intermediates(const Matrix& H_spatial2, const Matrix& d_cmo,
                                            const Tensor4& J, const Tensor4& K,
                                            const Matrix& D_tu_avg, const Tensor4& D_tuvw_avg,
                                            const Matrix& Dpe_tu_avg, double off_diagonal_constant,
                                            double omega, const Dimensions& dims);

struct GradientResult {
    Matrix A_tilde;         // (nmo, n_occupied) -- A_tilde[:, :n_occupied] in the Python
    Matrix gradient_tilde;  // (nmo, n_occupied)
};

// Faithful port of build_gradient, helper_PFCI.py:6188-6203, full_space==True
// branch only (the only one ever called, from microiteration_optimization6).
// The Python's A_tilde is (nmo, nmo) but only ever has its first n_occupied
// columns assigned (`A_tilde[:, :n_occupied] = ...`); the rest stay zero from
// initialization and are read as such by gradient_tilde's antisymmetrization
// term below. This returns A_tilde already restricted to that populated
// (nmo, n_occupied) shape rather than carrying the always-zero remainder.
GradientResult build_gradient(const Matrix& U, const Matrix& A, const Tensor4& G, const Dimensions& dims);

// Embeds build_gradient's (nmo, n_occupied) A_tilde back into the full
// (nmo, nmo) matrix it's a slice of (virtual-orbital columns zero, matching
// the Python's own A_tilde array -- see build_gradient's doc comment), then
// returns A_tilde_full + A_tilde_full.transpose(). Matches
// `sym_A_tilde = A_tilde + A_tilde.T` at helper_PFCI.py:15544, the input
// OrbitalHessianGuessProvider (hessian_guess.hpp) expects.
Matrix embed_and_symmetrize_A_tilde(const Matrix& A_tilde_occ_cols, const Dimensions& dims);

struct GradientAndHessianResult {
    Matrix gradient_tilde;  // (n_occupied, n_occupied)
    Tensor4 hessian_tilde;  // (n_occupied, n_occupied, n_occupied, n_occupied)
};

// Faithful port of build_gradient_and_hessian, helper_PFCI.py:6282-6441,
// full_space==False branch only (helper_PFCI.py:6412-6441) -- the only one
// ever called, from internal_optimization3 (which always passes A/G already
// restricted to rot_dim == n_occupied; the einsum labels in this branch
// reuse eye(n_occupied) for BOTH the (k,l) and (r,s) axis pairs, which is
// only self-consistent when rot_dim == n_occupied, confirming this branch's
// implicit assumption matches its one actual call site). Deliberately
// excludes the full_space==True branch (helper_PFCI.py:6285-6411): besides
// never being called, it's dominated by ~O(n^6) nested-loop debug/allclose
// validation code with no bearing on the returned values (see
// build_intermediates_internal's doc comment on the same pattern).
GradientAndHessianResult build_gradient_and_hessian(const Matrix& A, const Tensor4& G, const Dimensions& dims);

struct HessianDiagonalResult {
    Matrix hessian_diagonal;          // (nmo, n_occupied)
    Vector reduced_hessian_diagonal;  // (index_map_size)
};

// Faithful port of build_hessian_diagonal, helper_PFCI.py:14416-14513.
// A_tilde: (nmo, nmo), as populated by build_gradient. NOTE: build_gradient
// only ever fills A_tilde's first n_occupied columns -- this is called
// immediately after build_gradient with nothing populating the rest in
// between (helper_PFCI.py:11031/11042) -- so A_tilde's virtual-orbital
// columns (n_occupied:nmo) are always exactly zero on the only active call
// path. This makes the diag(A_tilde[n_occupied:nmo, n_occupied:nmo]) term
// below always numerically zero -- ported faithfully anyway (computed
// literally, not dropped), since it's cheap and safer than silently
// assuming it away.
//
// The reduction into reduced_hessian_diagonal reuses build_index_map: this
// function's own reduction loop (helper_PFCI.py:14497-14510, `for k in
// range(n_occupied): for r in range(k+1,nmo)`) is exactly build_index_map's
// enumeration restricted to its outer index < n_occupied -- and
// build_index_map's third (both-virtual) skip condition never actually
// removes anything else beyond that restriction, since whenever its outer
// index r >= n_occupied, the inner index s > r is automatically also
// >= n_occupied, so that pair was always going to be skipped anyway. So the
// two loops visit exactly the same set of (s, r) pairs, in the same order.
HessianDiagonalResult build_hessian_diagonal(const Matrix& U, const Tensor4& G, const Matrix& A_tilde,
                                              const Dimensions& dims);

} // namespace casscf
