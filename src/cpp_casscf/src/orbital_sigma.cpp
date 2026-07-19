#include "casscf/orbital_sigma.hpp"

namespace casscf {

Vector orbital_sigma3(const Matrix& U, const Matrix& A_tilde, const Tensor4& G, const Vector& R_reduced,
                       const Dimensions& dims) {
    const int nmo = dims.nmo;
    const int n_occupied = dims.n_occupied;
    const int n_in_a = dims.n_in_a;
    const int n_act = dims.n_act_orb;

    const auto index_map = build_index_map(dims);

    // Step 1 (helper_PFCI.py:8556-8560): embed the reduced vector into the
    // full (nmo, n_occupied) rotation-parameter matrix via index_map.
    Matrix R_total = Matrix::Zero(nmo, n_occupied);
    for (std::size_t j = 0; j < index_map.size(); ++j) {
        R_total(index_map[j].first, index_map[j].second) = R_reduced(static_cast<int>(j));
    }

    // Step 2 (helper_PFCI.py:8566-8576): temp1 = U @ R_total (contracted
    // over the full nmo axis) minus the same contraction with U's roles
    // swapped between the two arguments of R_total.
    Matrix temp1(nmo, n_occupied);
    for (int r = 0; r < nmo; ++r) {
        for (int k = 0; k < n_occupied; ++k) {
            double val = 0.0;
            for (int p = 0; p < nmo; ++p) val += U(r, p) * R_total(p, k);
            for (int l = 0; l < n_occupied; ++l) val -= U(r, l) * R_total(k, l);
            temp1(r, k) = val;
        }
    }

    // Step 3 (helper_PFCI.py:8580-8608): contract temp1 against G through
    // the (materialized-in-the-Python-as-G_ij/G_ti/G_tu, read directly from
    // G here) inactive/active blocks. The d < n_in_a case is the one place
    // two of the Python's separate terms (sigma_i's two np.dot calls) are
    // combined: both have the identical form
    // `sum_a sum_b' temp1(a,b')*G(d,b',c,a)`, over b' < n_in_a and
    // b' in [n_in_a, n_occupied) respectively -- two disjoint ranges whose
    // union is exactly [0, n_occupied), so summing them separately or
    // together is the same value; kept as one loop here purely because the
    // range split serves no purpose once read directly off G (it existed in
    // the Python only because G_ij/G_ti were separate materialized
    // matrices). The d >= n_in_a case does NOT admit the same combination
    // (the two terms have genuinely different G index patterns) and is kept
    // as two separate accumulations, exactly as the Python computes them.
    Matrix W(nmo, n_occupied);
    for (int c = 0; c < nmo; ++c) {
        for (int d = 0; d < n_in_a; ++d) {
            double val = 0.0;
            for (int a = 0; a < nmo; ++a)
                for (int bp = 0; bp < n_occupied; ++bp) val += temp1(a, bp) * G(d, bp, c, a);
            W(c, d) = val;
        }
        for (int dd = 0; dd < n_act; ++dd) {
            const int d = n_in_a + dd;
            double val = 0.0;
            for (int a = 0; a < nmo; ++a)
                for (int b = 0; b < n_in_a; ++b) val += temp1(a, b) * G(b, d, a, c);
            for (int a = 0; a < nmo; ++a)
                for (int b = 0; b < n_act; ++b) val += temp1(a, n_in_a + b) * G(d, n_in_a + b, c, a);
            W(c, d) = val;
        }
    }

    // Step 4 (helper_PFCI.py:8612-8618): wrap W back through U, plus the
    // r < n_occupied correction term.
    Matrix sigma_total(nmo, n_occupied);
    for (int r = 0; r < nmo; ++r) {
        for (int k = 0; k < n_occupied; ++k) {
            double val = 0.0;
            for (int c = 0; c < nmo; ++c) val += W(c, k) * U(c, r);
            if (r < n_occupied)
                for (int c = 0; c < nmo; ++c) val -= W(c, r) * U(c, k);
            sigma_total(r, k) = val;
        }
    }

    // Step 5 (helper_PFCI.py:8620-8680): the four A3_tilde correction
    // terms, each kept separate exactly as the Python computes them.
    const Matrix A3_tilde = A_tilde + A_tilde.transpose();
    for (int r = 0; r < nmo; ++r) {
        for (int k = 0; k < n_occupied; ++k) {
            double val = sigma_total(r, k);

            double term_a = 0.0;
            for (int p = 0; p < nmo; ++p) term_a += A3_tilde(r, p) * R_total(p, k);
            val -= 0.5 * term_a;

            if (r < n_occupied) {
                double term_b = 0.0;
                for (int p = 0; p < nmo; ++p) term_b += A3_tilde(k, p) * R_total(p, r);
                val += 0.5 * term_b;
            }

            double term_c = 0.0;
            for (int a = 0; a < n_occupied; ++a) term_c += R_total(r, a) * A3_tilde(k, a);
            val -= 0.5 * term_c;

            double term_d = 0.0;
            for (int q = 0; q < n_occupied; ++q) term_d += R_total(k, q) * A3_tilde(r, q);
            val += 0.5 * term_d;

            sigma_total(r, k) = val;
        }
    }

    // helper_PFCI.py:8682-8686: reduce back down via index_map.
    Vector sigma_reduced(static_cast<int>(index_map.size()));
    for (std::size_t j = 0; j < index_map.size(); ++j) {
        sigma_reduced(static_cast<int>(j)) = sigma_total(index_map[j].first, index_map[j].second);
    }
    return sigma_reduced;
}

} // namespace casscf
