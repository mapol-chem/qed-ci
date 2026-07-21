// Tests for root_analysis.hpp/.cpp -- the port of the post-CI-solve
// "ACTIVE PART OF DETERMINANTS THAT HAVE THE MOST IMPORTANT CONTRIBUTIONS"
// block (helper_PFCI.py:1959-2077).
//
// Uses the same deliberately hand-solvable active space as
// test_ci_state_average_solver.cpp: 2 active orbitals, n_act_a == 1 (so one
// alpha + one beta electron), H_spatial2 == diag(0, 1), J == K == 0, N_p == 0.
// That gives num_alpha == C(2,1) == 2, num_det == 4, H_dim == 4, and a CI
// index -> determinant map that can be written out by hand:
//
//     position | Ia=pos/2 | Ib=pos%2 | alpha string | beta string
//     ---------+----------+----------+--------------+-------------
//        0     |    0     |    0     |    [0]       |    [0]
//        1     |    0     |    1     |    [0]       |    [1]
//        2     |    1     |    0     |    [1]       |    [0]
//        3     |    1     |    1     |    [1]       |    [1]
//
// (recall the Python's own convention, ported literally: BETA comes from the
// remainder Idet % num_alpha and ALPHA from the quotient Idet / num_alpha.)
//
// check_total_spin is exercised against the REAL compiled ci_solver.c
// build_sigma_s_square, using cases whose <S^2> is unambiguous from first
// principles and needs no knowledge of the C code's phase conventions:
//   - a closed-shell determinant (both electrons in the same spatial
//     orbital) is a pure singlet          -> <S^2> = 0
//   - a single open-shell determinant with one alpha and one beta electron
//     in DIFFERENT spatial orbitals       -> <S^2> = 1
//   - the two normalized combinations of those two open-shell determinants
//     are the singlet and the m_s=0 triplet, so their <S^2> values are 0 and
//     2 in some order (which one gets which depends on the backend's
//     determinant phase convention, so the test asserts the SET, not the
//     assignment).
#include "casscf/ci_setup.hpp"
#include "casscf/root_analysis.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <sstream>

using namespace casscf;

namespace {

int failures = 0;

void expect_near(double actual, double expected, double tol, const char* label) {
    if (std::abs(actual - expected) > tol) {
        std::printf("FAIL: %s -- expected %.10f, got %.10f\n", label, expected, actual);
        ++failures;
    } else {
        std::printf("PASS: %s (%.10f ~= %.10f)\n", label, actual, expected);
    }
}

void expect_int(long actual, long expected, const char* label) {
    if (actual != expected) {
        std::printf("FAIL: %s -- expected %ld, got %ld\n", label, expected, actual);
        ++failures;
    } else {
        std::printf("PASS: %s (%ld)\n", label, actual);
    }
}

void expect_str(const std::string& actual, const std::string& expected, const char* label) {
    if (actual != expected) {
        std::printf("FAIL: %s -- expected \"%s\", got \"%s\"\n", label, expected.c_str(),
                    actual.c_str());
        ++failures;
    } else {
        std::printf("PASS: %s (\"%s\")\n", label, actual.c_str());
    }
}

Dimensions make_dims() {
    Dimensions dims;
    dims.n_in_a = 0;
    dims.n_act_orb = 2;
    dims.n_virtual = 0;
    dims.nmo = 2;
    dims.n_occupied = 2;
    return dims;
}

CasscfPhysicalConstants make_constants() {
    CasscfPhysicalConstants constants;
    constants.N_p = 0;
    constants.num_det = 4;
    constants.omega = 0.0;
    constants.Enuc = 0.0;
    constants.d_c = 0.0;
    constants.d_exp = 0.0;
    constants.weight = Vector::Constant(1, 1.0);
    return constants;
}

CasscfCiConfig make_config() {
    CasscfCiConfig config;
    config.n_act_a = 1;
    config.davidson_roots = 1;
    config.davidson_threshold = 1e-10;
    config.davidson_indim = 2;
    config.davidson_maxdim = 4;
    config.davidson_maxiter = 200;
    config.target_spin = -1.0;
    return config;
}

} // namespace

int main() {
    const double tol = 1e-8;

    // --- obt_bits_to_obt_index_list: pure bit arithmetic, hand-checked. ---
    {
        auto empty = obt_bits_to_obt_index_list(0);
        expect_int(static_cast<long>(empty.size()), 0, "bits 0 -> empty list");

        auto one = obt_bits_to_obt_index_list(1); // 0b0001
        expect_int(static_cast<long>(one.size()), 1, "bits 0b0001 -> size 1");
        expect_int(one.empty() ? -1 : one[0], 0, "bits 0b0001 -> [0]");

        auto two = obt_bits_to_obt_index_list(2); // 0b0010
        expect_int(two.empty() ? -1 : two[0], 1, "bits 0b0010 -> [1]");

        auto mixed = obt_bits_to_obt_index_list(11); // 0b1011 -> orbitals 0,1,3
        expect_str(format_index_list(mixed), "[0, 1, 3]", "bits 0b1011 -> [0, 1, 3]");

        // High bit, well past the small active spaces used elsewhere.
        auto high = obt_bits_to_obt_index_list(size_t(1) << 20);
        expect_int(high.empty() ? -1 : high[0], 20, "bits 1<<20 -> [20]");
    }

    // --- format_index_list ---
    {
        expect_str(format_index_list({}), "[]", "format empty list");
        expect_str(format_index_list({7}), "[7]", "format single-element list");
        expect_str(format_index_list({0, 2, 5}), "[0, 2, 5]", "format multi-element list");
    }

    Dimensions dims = make_dims();
    Matrix H_spatial2 = Matrix::Zero(dims.nmo, dims.nmo);
    H_spatial2(0, 0) = 0.0;
    H_spatial2(1, 1) = 1.0;

    CasscfCiConfig config = make_config();
    CasscfPhysicalConstants constants = make_constants();

    Tensor4 J(dims.n_occupied, dims.n_occupied, dims.nmo, dims.nmo);
    J.setZero();
    Tensor4 K(dims.n_occupied, dims.n_occupied, dims.nmo, dims.nmo);
    K.setZero();

    CasscfCiSetup setup(dims, config, constants, H_spatial2, J, K, /*E_core=*/0.0);
    expect_int(setup.H_dim(), 4, "H_dim == 4");
    expect_int(setup.num_alpha(), 2, "num_alpha == C(2,1) == 2");

    // --- check_total_spin against the real build_sigma_s_square. ---
    {
        // position 0 == alpha [0], beta [0]: closed shell -> pure singlet.
        Vector closed_shell = Vector::Zero(4);
        closed_shell(0) = 1.0;
        expect_near(check_total_spin(closed_shell, setup, config.n_act_a, dims.n_act_orb,
                                     constants.N_p),
                    0.0, 1e-9, "check_total_spin: closed-shell determinant -> <S^2> = 0");

        // position 1 == alpha [0], beta [1]: single open-shell determinant
        // with S_z = 0 -> <S^2> = 1.
        Vector open_shell = Vector::Zero(4);
        open_shell(1) = 1.0;
        expect_near(check_total_spin(open_shell, setup, config.n_act_a, dims.n_act_orb,
                                     constants.N_p),
                    1.0, 1e-9, "check_total_spin: single open-shell determinant -> <S^2> = 1");

        // The two normalized combinations are the singlet and m_s=0 triplet.
        const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
        Vector plus = Vector::Zero(4);
        plus(1) = inv_sqrt2;
        plus(2) = inv_sqrt2;
        Vector minus = Vector::Zero(4);
        minus(1) = inv_sqrt2;
        minus(2) = -inv_sqrt2;

        double s_plus = check_total_spin(plus, setup, config.n_act_a, dims.n_act_orb, constants.N_p);
        double s_minus =
            check_total_spin(minus, setup, config.n_act_a, dims.n_act_orb, constants.N_p);
        double lo = std::min(s_plus, s_minus);
        double hi = std::max(s_plus, s_minus);
        expect_near(lo, 0.0, 1e-9, "check_total_spin: open-shell combinations include <S^2> = 0");
        expect_near(hi, 2.0, 1e-9, "check_total_spin: open-shell combinations include <S^2> = 2");

        // check_total_spin normalizes internally, so scaling must not matter.
        Vector scaled = closed_shell * 37.0;
        expect_near(
            check_total_spin(scaled, setup, config.n_act_a, dims.n_act_orb, constants.N_p), 0.0,
            1e-9, "check_total_spin: normalizes its input (scale-invariant)");
    }

    // --- analyze_roots: one root, hand-built coefficients. ---
    {
        // Deliberately unsorted magnitudes so the descending-|c| ordering is
        // actually exercised: |c| = 0.1, 0.9, 0.3, 0.2 at positions 0..3, so
        // the expected order is positions 1, 2, 3, 0.
        Matrix eigenvecs(1, 4);
        eigenvecs(0, 0) = 0.1;
        eigenvecs(0, 1) = -0.9; // largest magnitude -> becomes the reference
        eigenvecs(0, 2) = 0.3;
        eigenvecs(0, 3) = 0.2;
        Vector eigenvals(1);
        eigenvals(0) = -1.25;

        ExcitationRankAccumulators acc(/*n_act_el=*/2);
        RootAnalysisResult result =
            analyze_roots(eigenvecs, eigenvals, setup, dims, config.n_act_a, constants.N_p, acc);

        expect_int(static_cast<long>(result.states.size()), 1, "analyze_roots: 1 state");
        const RootAnalysisState& st = result.states[0];
        expect_near(st.energy, -1.25, tol, "analyze_roots: energy passed through");
        expect_int(static_cast<long>(st.determinants.size()), 4,
                   "analyze_roots: all H_dim determinants retained (not just the printed 11)");

        // Descending |c|: positions 1 (0.9), 2 (0.3), 3 (0.2), 0 (0.1).
        const int expected_positions[4] = {1, 2, 3, 0};
        const double expected_amplitudes[4] = {-0.9, 0.3, 0.2, 0.1};
        for (int j = 0; j < 4; ++j) {
            char label[128];
            std::snprintf(label, sizeof(label), "analyze_roots: rank-%d position", j);
            expect_int(st.determinants[j].position, expected_positions[j], label);
            std::snprintf(label, sizeof(label), "analyze_roots: rank-%d amplitude (signed)", j);
            expect_near(st.determinants[j].amplitude, expected_amplitudes[j], tol, label);
        }

        // Decoding, per the table in this file's header comment. n_in_a == 0
        // here, so alphalist2/betalist2 equal the raw active indices.
        expect_str(format_index_list(st.determinants[0].alpha), "[0]", "position 1 -> alpha [0]");
        expect_str(format_index_list(st.determinants[0].beta), "[1]", "position 1 -> beta [1]");
        expect_str(format_index_list(st.determinants[1].alpha), "[1]", "position 2 -> alpha [1]");
        expect_str(format_index_list(st.determinants[1].beta), "[0]", "position 2 -> beta [0]");
        expect_str(format_index_list(st.determinants[2].alpha), "[1]", "position 3 -> alpha [1]");
        expect_str(format_index_list(st.determinants[2].beta), "[1]", "position 3 -> beta [1]");
        expect_str(format_index_list(st.determinants[3].alpha), "[0]", "position 0 -> alpha [0]");
        expect_str(format_index_list(st.determinants[3].beta), "[0]", "position 0 -> beta [0]");

        for (int j = 0; j < 4; ++j) {
            expect_int(st.determinants[j].photon, 0, "analyze_roots: photon block 0 (N_p == 0)");
        }

        // Excitation rank is measured against the LARGEST contribution of
        // state 0, i.e. position 1 == (alpha [0], beta [1]).
        //   position 1: alpha [0] vs [0] = 0, beta [1] vs [1] = 0 -> rank 0
        //   position 2: alpha [1] vs [0] = 1, beta [0] vs [1] = 1 -> rank 2
        //   position 3: alpha [1] vs [0] = 1, beta [1] vs [1] = 0 -> rank 1
        //   position 0: alpha [0] vs [0] = 0, beta [0] vs [1] = 1 -> rank 1
        const int expected_ranks[4] = {0, 2, 1, 1};
        for (int j = 0; j < 4; ++j) {
            char label[128];
            std::snprintf(label, sizeof(label), "analyze_roots: rank-%d excitation_rank", j);
            expect_int(st.determinants[j].excitation_rank, expected_ranks[j], label);
        }

        // Accumulators: one determinant at rank 0, two at rank 1, one at rank 2.
        expect_near(acc.config_count_by_rank[0], 1.0, tol, "accumulator: count[rank 0] == 1");
        expect_near(acc.config_count_by_rank[1], 2.0, tol, "accumulator: count[rank 1] == 2");
        expect_near(acc.config_count_by_rank[2], 1.0, tol, "accumulator: count[rank 2] == 1");
        // Squared weights: rank 0 -> 0.9^2; rank 1 -> 0.2^2 + 0.1^2; rank 2 -> 0.3^2.
        expect_near(acc.sum_squared_weight_by_rank[0], 0.81, tol, "accumulator: w2[rank 0]");
        expect_near(acc.sum_squared_weight_by_rank[1], 0.05, tol, "accumulator: w2[rank 1]");
        expect_near(acc.sum_squared_weight_by_rank[2], 0.09, tol, "accumulator: w2[rank 2]");

        // The Python ACCUMULATES into these across calls rather than
        // overwriting -- a second pass must double every entry.
        analyze_roots(eigenvecs, eigenvals, setup, dims, config.n_act_a, constants.N_p, acc);
        expect_near(acc.config_count_by_rank[1], 4.0, tol,
                    "accumulator: second call accumulates rather than overwrites");
    }

    // --- Multi-root: spin labelling, counters, and the state-0-only rule. ---
    {
        Matrix eigenvecs = Matrix::Zero(3, 4);
        eigenvecs(0, 0) = 1.0; // closed shell  -> <S^2> = 0 -> singlet 1
        eigenvecs(1, 1) = 1.0; // open shell    -> <S^2> = 1 -> no label
        // Row 2: whichever open-shell combination is the triplet; try (+) and
        // fall back to (-) so this does not depend on the phase convention.
        const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
        Vector plus = Vector::Zero(4);
        plus(1) = inv_sqrt2;
        plus(2) = inv_sqrt2;
        const bool plus_is_triplet =
            std::abs(check_total_spin(plus, setup, config.n_act_a, dims.n_act_orb, constants.N_p) -
                     2.0) < 1e-9;
        eigenvecs(2, 1) = inv_sqrt2;
        eigenvecs(2, 2) = plus_is_triplet ? inv_sqrt2 : -inv_sqrt2;

        Vector eigenvals(3);
        eigenvals << 0.0, 1.0, 1.0;

        ExcitationRankAccumulators acc(/*n_act_el=*/2);
        RootAnalysisResult result =
            analyze_roots(eigenvecs, eigenvals, setup, dims, config.n_act_a, constants.N_p, acc);

        expect_int(static_cast<long>(result.states.size()), 3, "multi-root: 3 states");
        expect_int(static_cast<long>(result.states[0].spin_label), static_cast<long>(SpinLabel::Singlet),
                   "multi-root: state 0 labelled singlet");
        expect_int(result.states[0].spin_label_count, 1, "multi-root: singlet counter == 1");
        expect_int(static_cast<long>(result.states[1].spin_label), static_cast<long>(SpinLabel::None),
                   "multi-root: state 1 (<S^2>=1) gets no label");
        expect_int(static_cast<long>(result.states[2].spin_label), static_cast<long>(SpinLabel::Triplet),
                   "multi-root: state 2 labelled triplet");
        expect_int(result.states[2].spin_label_count, 1, "multi-root: triplet counter == 1");

        // helper_PFCI.py:2015 sets excitation_rank = 0 before the j loop and
        // only updates it inside `if i == 0`, so every non-zero state reports 0.
        long nonzero_rank_in_excited_states = 0;
        for (size_t s = 1; s < result.states.size(); ++s) {
            for (const auto& d : result.states[s].determinants) {
                if (d.excitation_rank != 0) ++nonzero_rank_in_excited_states;
            }
        }
        expect_int(nonzero_rank_in_excited_states, 0,
                   "multi-root: excitation_rank stays 0 for states > 0 (matches Python)");

        // Only state 0 contributes to the accumulators: 4 determinants total.
        double total_count = 0.0;
        for (double c : acc.config_count_by_rank) total_count += c;
        expect_near(total_count, 4.0, tol,
                    "multi-root: only state 0 contributes to the rank accumulators");
    }

    // --- n_in_a shift: alphalist2/betalist2 offset into full-MO numbering. ---
    {
        Dimensions shifted = make_dims();
        shifted.n_in_a = 3;
        shifted.nmo = 5;
        shifted.n_occupied = 5;

        Matrix eigenvecs = Matrix::Zero(1, 4);
        eigenvecs(0, 3) = 1.0; // alpha [1], beta [1] -> shifted to [4], [4]
        Vector eigenvals(1);
        eigenvals(0) = 0.0;

        ExcitationRankAccumulators acc(/*n_act_el=*/2);
        RootAnalysisResult result = analyze_roots(eigenvecs, eigenvals, setup, shifted,
                                                  config.n_act_a, constants.N_p, acc);
        expect_str(format_index_list(result.states[0].determinants[0].alpha), "[4]",
                   "n_in_a shift: alpha index offset by n_in_a");
        expect_str(format_index_list(result.states[0].determinants[0].beta), "[4]",
                   "n_in_a shift: beta index offset by n_in_a");
    }

    // --- print_root_analysis: header text and the `j <= 10` print cap. ---
    {
        Matrix eigenvecs = Matrix::Zero(1, 4);
        eigenvecs(0, 0) = 1.0;
        Vector eigenvals(1);
        eigenvals(0) = -0.5;
        ExcitationRankAccumulators acc(/*n_act_el=*/2);
        RootAnalysisResult result =
            analyze_roots(eigenvecs, eigenvals, setup, dims, config.n_act_a, constants.N_p, acc);

        std::ostringstream os;
        print_root_analysis(result, dims, os);
        const std::string out = os.str();

        expect_int(out.find("ACTIVE PART OF DETERMINANTS THAT HAVE THE MOST IMPORTANT "
                            "CONTRIBUTIONS") != std::string::npos,
                   1, "print: emits the Python's header line");
        expect_int(out.find("singlet 1") != std::string::npos, 1, "print: emits the singlet label");
        expect_int(out.find("excitation ranke") != std::string::npos, 1,
                   "print: keeps the Python's 'excitation ranke' spelling");
        // 4 determinants here, all under the cap -> 4 amplitude lines.
        long alpha_lines = 0;
        for (size_t p = out.find("alpha "); p != std::string::npos; p = out.find("alpha ", p + 1)) {
            ++alpha_lines;
        }
        expect_int(alpha_lines, 4, "print: one line per determinant when under the 11-line cap");
    }

    if (failures == 0) {
        std::printf("\nAll root analysis tests passed.\n");
        return 0;
    }
    std::printf("\n%d root analysis test(s) FAILED.\n", failures);
    return 1;
}
