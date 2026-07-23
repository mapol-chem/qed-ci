#pragma once

#include <cstddef>

// extern "C" declarations for the plain-C CASSCF backend (ci_solver.c /
// orbital.c, one level up in qed-ci/src/) that this Python codebase already
// calls via ctypes -- see cpp_casscf/README.md's "CiStateAverageSolver"/
// "IntegralTransformer" sections for which of these back which C++
// interface, and CMakeLists.txt for how these two .c files get compiled
// and linked (casscf_c_backend).
//
// Transcribed directly from ci_solver.h / orbital.h (NOT from the Python
// ctypes `argtypes` declarations, which use the same shapes but sometimes
// looser types -- e.g. `get_graph`'s first two parameters are `size_t` in
// the real C header, not `int`; ctypes' `c_size_t`/`c_int32` distinction is
// easy to lose sight of if only the Python wrapper is read). All arrays are
// C-contiguous (row-major) flat buffers, matching numpy's default layout
// and Eigen::Tensor<..., RowMajor>/a row-major Eigen::Matrix -- NOT
// Eigen::MatrixXd's default column-major storage; callers must marshal
// through a row-major temporary for any 2D `double*` argument that is
// itself an Eigen::MatrixXd (see integral_transformer.cpp for the pattern).
extern "C" {

// CI print level (0=Silent, 1=Normal, 2=Debug, 3=Trace; matches
// casscf::PrintLevel). Governs ci_solver.c's own stdout (per-iteration tables,
// timing, memory, spin checkpoint). The C++ driver syncs this to context.log.level
// so --print-level also quiets the C backend, which otherwise defaults to Trace.
void set_ci_print_level(int level);
int get_ci_print_level(void);

// ci_solver.c -- CI graph/string-table setup (computed once per active-space
// definition, reused for a whole CASSCF run; see CasscfCiSetup, not yet
// ported as of this header's writing).
void get_graph(size_t N, size_t n_o, int* Y);

// Maps a lexical alpha/beta string INDEX to its occupation BIT-STRING, using
// the graph `Y` built by get_graph. Returns `size_t` in ci_solver.h (the
// ctypes wrapper c_index_to_string, helper_PFCI.py:497, hides this). Used by
// the root analysis (root_analysis.hpp) to recover which orbitals a given
// determinant index occupies.
size_t index_to_string(int index, int N, int n_o, int* Y);

void get_string(double* h1e, double* h2e, double* H_diag, int* b_array, int* table,
                 int* table_creation, int* table_annihilation, int N_p, int num_alpha, int nmo, int N,
                 int n_o, int n_in_a, double E_core, double omega, double Enuc, double dc,
                 double target_spin);

void build_H_diag_cas_spin(double* h1e, double* h2e, double* H_diag, int N_p, int num_alpha, int nmo,
                             int n_act_a, int n_act_orb, int n_in_a, double E_core, double omega,
                             double Enuc, double dc, int* Y, double target_spin);

void build_S_diag(double* S_diag, int num_alpha, int nmo, int N_ac, int n_o_ac, int n_o_in,
                    double shift);

// Applies the S^2 operator to `c_vectors`, accumulating into `c1_vectors`
// (scaled by `scale`). Backs check_total_spin (helper_PFCI.py:5169-5188) via
// the ctypes wrapper c_sigma_s_square -- see root_analysis.hpp.
void build_sigma_s_square(double* c_vectors, double* c1_vectors, double* S_diag, int* b_array,
                            int* table1, int num_links, int n_o_ac, int num_alpha, int num_state,
                            int N_p, double scale);

// ci_solver.c -- per-macroiteration CI Davidson solve + active-space RDMs
// (CasscfCiStateAverageSolver, not yet ported as of this header's writing).
void get_roots(double* h1e, double* h2e, double* d_cmo, double* Hdiag, double* Sdiag,
                double* Sdiag_projection, double* eigenvals, double* eigenvecs, int* table,
                int* table_creation, int* table_annihilation, int* b_array, int* constint,
                double* constdouble, int* index_Hdiag, bool casscf, double target_spin);

void build_active_rdm(double* eigvec, double* D_tu, double* D_tuvw, int* table, int N_ac, int n_o_ac,
                        int num_photon, int state_p1, int state_p2, double weight);

void build_active_photon_electron_one_rdm(double* eigvec, double* Dpe_tu, int* table, int N_ac,
                                            int n_o_ac, int num_photon, int state_p1, int state_p2,
                                            double weight);

// orbital.c -- integral transformations under an orbital rotation
// (CasscfIntegralTransformer, integral_transformer.hpp/.cpp).
void full_transformation_macroiteration(double* U, double* h2e, double* J, double* K,
                                          int* index_map_pq, int* index_map_kl, int nmo,
                                          int n_occupied);

void full_transformation_internal_optimization(double* U, double* J, double* K, double* h,
                                                  double* d_cmo, double* J1, double* K1, double* h1,
                                                  double* d_cmo1, int* index_map_ab, int* index_map_kl,
                                                  int nmo, int n_occupied);

} // extern "C"
