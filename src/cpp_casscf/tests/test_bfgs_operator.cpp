// Tests for BfgsOperator (get_bfgs_mv + the damped-BFGS history update,
// helper_PFCI.py:10857-10906, 11128-11176).
#include "casscf/bfgs_operator.hpp"
#include "casscf/orbital_sigma.hpp"

#include <cmath>
#include <cstdio>

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

void expect_vector_near(const Vector& actual, const Vector& expected, double tol, const char* label) {
    expect_near((actual - expected).norm(), 0.0, tol, label);
}

Dimensions small_dims() {
    Dimensions dims;
    dims.n_in_a = 1;
    dims.n_act_orb = 1;
    dims.n_virtual = 1;
    dims.nmo = 3;
    dims.n_occupied = 2;
    return dims;
}

} // namespace

int main() {
    const double tol = 1e-9;
    const Dimensions dims = small_dims();
    const int n = dims.index_map_size();

    Matrix U(3, 3);
    U << 1.0, 0.1, -0.1,
        -0.1, 1.0, 0.2,
        0.05, -0.2, 1.0;
    Matrix A_tilde(3, 3);
    A_tilde << 0.4, 0.1, -0.1,
              -0.1, 0.3, 0.2,
               0.1, 0.0, 0.5;
    Tensor4 G(2, 2, 3, 3);
    double v0 = 0.02;
    for (int k = 0; k < 2; ++k)
        for (int l = 0; l < 2; ++l)
            for (int r = 0; r < 3; ++r)
                for (int s = 0; s < 3; ++s) {
                    G(k, l, r, s) = v0;
                    v0 += 0.005;
                }

    // --- apply() with empty history == plain orbital_sigma3. ---
    {
        BfgsOperator op(U, A_tilde, G, dims);
        Vector v(n);
        for (int i = 0; i < n; ++i) v(i) = 0.1 * (i + 1);
        Vector result = op.apply(v);
        Vector expected = orbital_sigma3(U, A_tilde, G, v, dims);
        expect_vector_near(result, expected, tol, "apply(): empty history == orbital_sigma3 directly");
    }

    // --- apply() with one hand-crafted history entry: check the recursive
    //     update formula directly (sigma = B0*v + y*(y.v*rho_y) -
    //     Bs*(s.(B0*v)*rho_Bs)), not via update() (so this test doesn't
    //     depend on update()'s own correctness). ---
    {
        BfgsOperator op(U, A_tilde, G, dims);
        // Can't set history_ directly (private) -- use update() with a
        // hand-chosen (s,y) pair where the curvature condition holds
        // (ys_dot >= 0.1*sBs_dot), so final_y == y_vec and final_rho_y ==
        // 1/ys_dot exactly (no damping), keeping the entry's values
        // predictable for the hand-check below.
        Vector s(n), y(n);
        for (int i = 0; i < n; ++i) {
            s(i) = 0.3 - 0.05 * i;
            y(i) = 0.5 + 0.1 * i; // large y -> large ys_dot, avoids damping
        }
        const Vector B0_s = orbital_sigma3(U, A_tilde, G, s, dims);
        const double ys_dot = y.dot(s);
        const double sBs_dot = s.dot(B0_s);
        if (ys_dot < 0.1 * sBs_dot) {
            std::printf("SKIP: hand-crafted (s,y) unexpectedly triggers damping -- adjust test data\n");
            ++failures;
        } else {
            op.update(s, y);
            const double rho_y = 1.0 / ys_dot;
            const double rho_Bs = 1.0 / sBs_dot;

            Vector v(n);
            for (int i = 0; i < n; ++i) v(i) = -0.2 + 0.07 * i;
            const Vector B0_v = orbital_sigma3(U, A_tilde, G, v, dims);
            const Vector expected = B0_v + y * (y.dot(v) * rho_y) - B0_s * (s.dot(B0_v) * rho_Bs);
            const Vector result = op.apply(v);
            expect_vector_near(result, expected, tol, "apply(): one undamped history entry matches hand formula");
        }
    }

    // --- update(): no damping needed (ys_dot >= 0.1*sBs_dot) -- history
    //     entry should store y_vec/rho_y = 1/ys_dot exactly. ---
    {
        BfgsOperator op(U, A_tilde, G, dims);
        Vector s(n), y(n);
        for (int i = 0; i < n; ++i) {
            s(i) = 0.2 + 0.01 * i;
            y(i) = 0.6 - 0.02 * i;
        }
        const Vector Bs_vec = orbital_sigma3(U, A_tilde, G, s, dims);
        const double ys_dot = y.dot(s);
        const double sBs_dot = s.dot(Bs_vec);
        if (ys_dot < 0.1 * sBs_dot) {
            std::printf("SKIP: expected no-damping case actually triggers damping -- adjust test data\n");
            ++failures;
        } else {
            op.update(s, y);
            const BfgsHistoryEntry& e = op.history().back();
            expect_vector_near(e.s, s, tol, "update() no-damping: stored s == input s_vec");
            expect_vector_near(e.y, y, tol, "update() no-damping: stored y == y_vec (undamped)");
            expect_near(e.rho_y, 1.0 / ys_dot, tol, "update() no-damping: rho_y == 1/y.s");
            expect_near(e.rho_Bs, 1.0 / sBs_dot, tol, "update() no-damping: rho_Bs == 1/s.Bs");
        }
    }

    // --- update(): damping needed (force ys_dot < 0.1*sBs_dot). The
    //     damping condition `ys_dot < 0.1*sBs_dot` doesn't require sBs_dot
    //     to be positive -- it's satisfied just as well by making ys_dot
    //     substantially more negative than 0.1*sBs_dot, which y = -s
    //     achieves regardless of orbital_sigma3's sign structure (which
    //     isn't hand-predictable from G alone): y.s = -s.s < 0 always,
    //     while 0.1*sBs_dot is a much smaller-magnitude number here. ---
    {
        BfgsOperator op(U, A_tilde, G, dims);
        Vector s(n);
        for (int i = 0; i < n; ++i) s(i) = 0.2 + 0.01 * i;
        const Vector Bs_vec = orbital_sigma3(U, A_tilde, G, s, dims);
        const double sBs_dot = s.dot(Bs_vec);

        Vector y = -s;
        const double ys_dot = y.dot(s);

        if (!(ys_dot < 0.1 * sBs_dot)) {
            std::printf("SKIP: expected damping case doesn't actually trigger damping -- adjust test data "
                        "(sBs_dot=%.6f, ys_dot=%.9f)\n",
                        sBs_dot, ys_dot);
            ++failures;
        } else {
            op.update(s, y);
            const BfgsHistoryEntry& e = op.history().back();

            const double damping_sigma = 0.1;
            const double theta = ((1.0 - damping_sigma) * sBs_dot) / (sBs_dot - ys_dot);
            const Vector expected_y = theta * y + (1.0 - theta) * Bs_vec;
            const double expected_rho_y = 1.0 / (damping_sigma * sBs_dot);

            expect_vector_near(e.y, expected_y, tol, "update() damping: stored y == Powell-damped y_bar");
            expect_near(e.rho_y, expected_rho_y, tol, "update() damping: rho_y == 1/(sigma*sBs_dot)");
            expect_near(e.rho_Bs, 1.0 / sBs_dot, tol, "update() damping: rho_Bs == 1/s.Bs (unaffected by damping)");
        }
    }

    // --- history cap: pushing more than m_history entries pops the oldest. ---
    {
        const int m_history = 3;
        BfgsOperator op(U, A_tilde, G, dims, m_history);
        Vector s0(n), y0(n);
        for (int i = 0; i < n; ++i) {
            s0(i) = 0.5;
            y0(i) = 0.5;
        }
        for (int t = 0; t < m_history + 2; ++t) {
            Vector s = s0 * (1.0 + 0.1 * t);
            Vector y = y0 * (1.0 + 0.1 * t);
            op.update(s, y);
        }
        expect_near(static_cast<double>(op.history().size()), static_cast<double>(m_history), 0.0,
                    "history cap: size stays at m_history after exceeding it");
        // The oldest surviving entry should be from t=2 (0-indexed; entries
        // from t=0,1 popped), i.e. s == s0*1.2.
        expect_vector_near(op.history().front().s, s0 * 1.2, tol, "history cap: oldest surviving entry is t=2");
    }

    // --- reset_reference(): clears history and adopts new tensors. ---
    {
        BfgsOperator op(U, A_tilde, G, dims);
        Vector s(n), y(n);
        for (int i = 0; i < n; ++i) {
            s(i) = 0.3;
            y(i) = 0.5;
        }
        op.update(s, y);
        if (op.history().empty()) {
            std::printf("FAIL: reset_reference() setup: history unexpectedly empty before reset\n");
            ++failures;
        }
        Matrix U2 = Matrix::Identity(3, 3);
        op.reset_reference(U2, A_tilde, G);
        expect_near(static_cast<double>(op.history().size()), 0.0, 0.0, "reset_reference(): history cleared");
        expect_vector_near(Vector::Map(op.U_zero().data(), op.U_zero().size()),
                            Vector::Map(U2.data(), U2.size()), tol, "reset_reference(): U_zero updated");
    }

    if (failures == 0) {
        std::printf("\nAll tests passed.\n");
        return 0;
    }
    std::printf("\n%d test(s) failed.\n", failures);
    return 1;
}
