#pragma once

#include "casscf/ci_setup.hpp"
#include "casscf/tensor_types.hpp"
#include "casscf/types.hpp"

#include <cstddef>
#include <iosfwd>
#include <string>
#include <vector>

namespace casscf {

// Port of the post-CI-solve root analysis, helper_PFCI.py:1959-2077 -- the
// "ACTIVE PART OF DETERMINANTS THAT HAVE THE MOST IMPORTANT CONTRIBUTIONS"
// block. For every CI root it reports <S^2> (with a singlet/triplet/quintet
// label and a running per-label counter), then walks that root's coefficients
// in descending |c| order, decoding each CI index back into the alpha/beta
// active-orbital occupation lists and the photon-number block it belongs to.
//
// Three near-identical copies of this block exist in the Python
// (helper_PFCI.py:1959, 2618, 5457). This ports the FIRST (the __init__
// post-CI-solve one, the only one carrying the excitation-rank accumulation
// -- confirmed by grep: "excitation ranke" appears at exactly one line,
// helper_PFCI.py:2072). The other two are strictly smaller subsets of the
// same logic and can reuse this if they are ever needed.
//
// Index decoding, transcribed literally from helper_PFCI.py:2003-2032:
//     Idet     = position % num_det
//     photon_p = (position - Idet) / num_det
//     Ib       = Idet % num_alpha        <- BETA  from the remainder
//     Ia       = Idet / num_alpha        <- ALPHA from the quotient
// (that alpha/beta assignment is the Python's, and is not the order the
// names suggest -- kept exactly as written rather than "corrected".)

// One determinant's contribution to one root.
struct RootAnalysisDeterminant {
    double amplitude = 0.0; // c_i, signed
    int position = 0;       // index into the full (num_det * (N_p+1)) CI vector
    // Active-space orbital indices SHIFTED by n_in_a, matching the Python's
    // alphalist2/betalist2 (helper_PFCI.py:2058-2061). The commented-out
    // inactive-orbital prepend there is deliberately not ported.
    std::vector<int> alpha;
    std::vector<int> beta;
    int photon = 0;
    // Only meaningful for state 0: the Python computes this inside `if i==0`
    // and leaves it at its pre-loop 0 for every other state
    // (helper_PFCI.py:2015, 2044-2047). Reproduced literally.
    int excitation_rank = 0;
};

enum class SpinLabel { None, Singlet, Triplet, Quintet };

struct RootAnalysisState {
    int state = 0;
    double energy = 0.0;
    double total_spin = 0.0;
    SpinLabel spin_label = SpinLabel::None;
    // The running singlet_count/triplet_count the Python prints alongside the
    // label (helper_PFCI.py:1982-1990). 0 when spin_label is None/Quintet --
    // the Python prints no counter for quintets.
    int spin_label_count = 0;
    // ALL H_dim determinants, sorted by descending |amplitude| -- the Python
    // loops over the full range and only gates PRINTING to the first 11
    // (`if j <= 10`, helper_PFCI.py:2064). The excitation-rank accumulation
    // below runs over all of them, so truncating here would change results.
    std::vector<RootAnalysisDeterminant> determinants;
};

struct RootAnalysisResult {
    std::vector<RootAnalysisState> states;
};

// Mirrors self.casci_config_count_by_rank / self.casci_sum_squared_weight_by_rank
// (helper_PFCI.py:3315-3319), which are allocated once per run and ACCUMULATED
// into by this block. Passed in/out for that reason rather than returned
// fresh. Both are sized n_act_el+1.
struct ExcitationRankAccumulators {
    std::vector<double> config_count_by_rank;
    std::vector<double> sum_squared_weight_by_rank;

    explicit ExcitationRankAccumulators(int n_act_el)
        : config_count_by_rank(static_cast<size_t>(n_act_el) + 1, 0.0),
          sum_squared_weight_by_rank(static_cast<size_t>(n_act_el) + 1, 0.0) {}
};

// Port of Determinant.obtBits2ObtIndexList (helper_PFCI.py:991-1003):
// the occupied-orbital indices of a bit-string, ascending.
std::vector<int> obt_bits_to_obt_index_list(size_t bits);

// Port of check_total_spin (helper_PFCI.py:5169-5188): normalizes `v` and
// returns <v|S^2|v>. `v` is a single CI vector (length H_dim).
double check_total_spin(const Vector& v, const CasscfCiSetup& setup, int n_act_a, int n_act_orb,
                        int N_p);

// The analysis proper. `eigenvecs` is (n_roots, H_dim) and `eigenvals` is
// (n_roots), matching CiStateAverageResult. `accumulators` is updated in
// place from state 0 only, exactly as the Python does.
RootAnalysisResult analyze_roots(const Matrix& eigenvecs, const Vector& eigenvals,
                                 const CasscfCiSetup& setup, const Dimensions& dims, int n_act_a,
                                 int N_p, ExcitationRankAccumulators& accumulators);

// Reproduces the Python's stdout formatting for the block, including the
// `if j <= 10` print cap. Kept separate from analyze_roots so the analysis is
// testable without scraping text.
//
// Documented deviation: Python prints `eigenvals[i]` and `total_spin` with
// str()/repr() float formatting, which C++ iostreams cannot reproduce
// character-for-character in general. This uses %.12g, so those two fields
// may differ in trailing digits from the Python log while every other field
// (the %20.12lf amplitude, %9.3d position, %4.1d photon, orbital lists,
// excitation rank) matches exactly.
void print_root_analysis(const RootAnalysisResult& result, const Dimensions& dims,
                         std::ostream& os);

// Python-list formatting ("[0, 1, 2]") for the alpha/beta orbital lists, so
// the printed line matches the Python's own list repr.
std::string format_index_list(const std::vector<int>& v);

} // namespace casscf
