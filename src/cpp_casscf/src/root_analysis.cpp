#include "casscf/root_analysis.hpp"

#include "casscf/ci_orbital_backend.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <ostream>

namespace casscf {
namespace {

// ci_solver.c's index_to_string/build_sigma_s_square take non-const int*/double*
// even though they only read these buffers -- same const_cast pattern already
// used in ci_state_average_solver.cpp.
int* mutable_ptr(const std::vector<int32_t>& v) { return const_cast<int32_t*>(v.data()); }
double* mutable_ptr(const Vector& v) { return const_cast<double*>(v.data()); }

} // namespace

std::vector<int> obt_bits_to_obt_index_list(size_t bits) {
    // helper_PFCI.py:991-1003, transcribed directly.
    std::vector<int> obts;
    int i = 0;
    while (bits != 0) {
        if ((bits & 1u) == 1u) {
            obts.push_back(i);
        }
        bits >>= 1;
        ++i;
    }
    return obts;
}

double check_total_spin(const Vector& v, const CasscfCiSetup& setup, int n_act_a, int n_act_orb,
                        int N_p) {
    // helper_PFCI.py:5169-5188.
    const double norm = std::sqrt(v.dot(v));
    Vector v1 = v / norm;
    Vector newS = Vector::Zero(v.size());

    // num_links0 = n_act_a * (n_act_orb - n_act_a) + n_act_a  (helper_PFCI.py:5173)
    const int num_links0 = n_act_a * (n_act_orb - n_act_a) + n_act_a;

    build_sigma_s_square(v1.data(), newS.data(), mutable_ptr(setup.S_diag()),
                         mutable_ptr(setup.b_array()), mutable_ptr(setup.table()), num_links0,
                         n_act_orb, setup.num_alpha(), /*num_state=*/1, N_p, /*scale=*/1.0);

    return v1.dot(newS);
}

RootAnalysisResult analyze_roots(const Matrix& eigenvecs, const Vector& eigenvals,
                                 const CasscfCiSetup& setup, const Dimensions& dims, int n_act_a,
                                 int N_p, ExcitationRankAccumulators& accumulators) {
    const int n_act_orb = dims.n_act_orb;
    const int n_in_a = dims.n_in_a;
    const int num_alpha = setup.num_alpha();
    const int num_det = num_alpha * num_alpha; // helper_PFCI.py:1417
    const int n_states = static_cast<int>(eigenvecs.rows());
    const int H_dim = static_cast<int>(eigenvecs.cols());

    // helper_PFCI.py:1963-1967 -- a FRESH graph is built here, sized for the
    // active space, rather than reusing self.Y. Same call, same arguments,
    // so it is numerically identical to setup.Y(); rebuilt anyway to keep this
    // function self-contained and to match the Python line-for-line.
    std::vector<int32_t> Y(static_cast<size_t>(n_act_a) * (n_act_orb - n_act_a + 1) * 3, 0);
    get_graph(static_cast<size_t>(n_act_a), static_cast<size_t>(n_act_orb), Y.data());

    RootAnalysisResult result;
    result.states.reserve(static_cast<size_t>(n_states));

    int singlet_count = 0;
    int triplet_count = 0;

    // Reference (state-0, largest-|c|) occupation lists the excitation rank is
    // measured against. Only ever written in the i==0 pass, exactly as the
    // Python's a_ref/b_ref are (helper_PFCI.py:2017-2019).
    std::vector<int> a_ref;
    std::vector<int> b_ref;

    for (int i = 0; i < n_states; ++i) {
        RootAnalysisState state;
        state.state = i;
        state.energy = eigenvals(i);

        state.total_spin = check_total_spin(Vector(eigenvecs.row(i)), setup, n_act_a, n_act_orb, N_p);

        // helper_PFCI.py:1982-1990. Note these are exact-ish equality tests
        // against 0/2/6 with a 1e-5 window; anything else gets no label.
        if (std::abs(state.total_spin) < 1e-5) {
            state.spin_label = SpinLabel::Singlet;
            state.spin_label_count = ++singlet_count;
        } else if (std::abs(state.total_spin - 2.0) < 1e-5) {
            state.spin_label = SpinLabel::Triplet;
            state.spin_label_count = ++triplet_count;
        } else if (std::abs(state.total_spin - 6.0) < 1e-5) {
            state.spin_label = SpinLabel::Quintet; // the Python prints no counter here
        }

        // index = np.argsort(np.abs(eigenvecs[i, :])) -- ASCENDING by |c|, so
        // index[H_dim - 1 - j] is the j-th LARGEST (helper_PFCI.py:1998).
        //
        // Deviation: np.argsort's default quicksort is not stable, so exactly
        // tied |c| values may be ordered differently than in the Python. A
        // stable sort is used here to keep this deterministic; ties only
        // reorder degenerate-magnitude determinants and never change the
        // multiset of reported amplitudes.
        std::vector<int> index(static_cast<size_t>(H_dim));
        std::iota(index.begin(), index.end(), 0);
        std::stable_sort(index.begin(), index.end(), [&](int lhs, int rhs) {
            return std::abs(eigenvecs(i, lhs)) < std::abs(eigenvecs(i, rhs));
        });

        // Decodes one CI position into (alpha list, beta list, photon block).
        auto decode = [&](int position, std::vector<int>& alpha, std::vector<int>& beta,
                          int& photon) {
            const int Idet = position % num_det;
            photon = (position - Idet) / num_det;
            const int Ib = Idet % num_alpha;
            const int Ia = Idet / num_alpha;
            const size_t a = index_to_string(Ia, n_act_a, n_act_orb, Y.data());
            const size_t b = index_to_string(Ib, n_act_a, n_act_orb, Y.data());
            alpha = obt_bits_to_obt_index_list(a);
            beta = obt_bits_to_obt_index_list(b);
        };

        if (i == 0) {
            // helper_PFCI.py:2001-2019: the reference is the single largest
            // contribution of state 0.
            int photon0 = 0;
            decode(index[static_cast<size_t>(H_dim) - 1], a_ref, b_ref, photon0);
        }

        state.determinants.reserve(static_cast<size_t>(H_dim));
        for (int j = 0; j < H_dim; ++j) {
            const int position = index[static_cast<size_t>(H_dim - j - 1)];

            RootAnalysisDeterminant det;
            det.position = position;
            det.amplitude = eigenvecs(i, position);
            decode(position, det.alpha, det.beta, det.photon);

            if (i == 0) {
                // helper_PFCI.py:2038-2052. np.sum(a_ref != a_curr) is an
                // ELEMENTWISE comparison of two equal-length occupied-orbital
                // lists (both have n_act_a entries, since particle number is
                // conserved), i.e. it counts positions at which the sorted
                // occupation lists differ -- NOT a set difference.
                int a_diff = 0;
                for (size_t k = 0; k < a_ref.size() && k < det.alpha.size(); ++k) {
                    if (a_ref[k] != det.alpha[k]) ++a_diff;
                }
                int b_diff = 0;
                for (size_t k = 0; k < b_ref.size() && k < det.beta.size(); ++k) {
                    if (b_ref[k] != det.beta[k]) ++b_diff;
                }
                det.excitation_rank = a_diff + b_diff;

                const size_t rank = static_cast<size_t>(det.excitation_rank);
                if (rank < accumulators.config_count_by_rank.size()) {
                    accumulators.config_count_by_rank[rank] += 1.0;
                    accumulators.sum_squared_weight_by_rank[rank] += det.amplitude * det.amplitude;
                }
            }

            // alphalist2/betalist2 -- shift active indices into full-MO
            // numbering (helper_PFCI.py:2058-2061).
            for (int& x : det.alpha) x += n_in_a;
            for (int& x : det.beta) x += n_in_a;

            state.determinants.push_back(std::move(det));
        }

        result.states.push_back(std::move(state));
    }

    return result;
}

std::string format_index_list(const std::vector<int>& v) {
    std::string s = "[";
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) s += ", ";
        s += std::to_string(v[i]);
    }
    s += "]";
    return s;
}

void print_root_analysis(const RootAnalysisResult& result, const Dimensions& dims,
                         std::ostream& os) {
    (void)dims;
    char buf[256];

    os << "\nACTIVE PART OF DETERMINANTS THAT HAVE THE MOST IMPORTANT CONTRIBUTIONS\n";

    for (const RootAnalysisState& state : result.states) {
        std::snprintf(buf, sizeof(buf), "state %d energy = %.12g <S^2>= %.12g", state.state,
                      state.energy, state.total_spin);
        os << buf;
        switch (state.spin_label) {
        case SpinLabel::Singlet: os << "\tsinglet " << state.spin_label_count; break;
        case SpinLabel::Triplet: os << "\ttriplet " << state.spin_label_count; break;
        case SpinLabel::Quintet: os << "\tquintet"; break;
        case SpinLabel::None: break;
        }
        os << "\n";

        os << "        amplitude       position          most important determinants"
              "              number of photon\n";

        // helper_PFCI.py:2063-2077 -- only the first 11 (j <= 10) are printed.
        const size_t n_print = std::min<size_t>(state.determinants.size(), 11);
        for (size_t j = 0; j < n_print; ++j) {
            const RootAnalysisDeterminant& d = state.determinants[j];
            std::snprintf(buf, sizeof(buf), "%20.12lf %9.3d", d.amplitude, d.position);
            os << buf << " alpha " << format_index_list(d.alpha) << "    beta "
               << format_index_list(d.beta);
            std::snprintf(buf, sizeof(buf), " %4.1d", d.photon);
            // "excitation rank" (corrected): the Python's analyze_roots block
            // (helper_PFCI.py:2093) misspells this label "excitation ranke";
            // its own SA-CASSCF block (helper_PFCI.py:2773) spells it correctly.
            // This is a deliberate cosmetic deviation from the ported line's typo.
            os << buf << " photon excitation rank " << d.excitation_rank << "\n";
        }
    }
}

} // namespace casscf
